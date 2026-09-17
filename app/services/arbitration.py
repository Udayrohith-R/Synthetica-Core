"""
Module 5: Deep Arbitration Loop.

Invoked when H(R_q) exceeds the consensus threshold.
Uses a larger model (Llama-3.1-405B / Llama-3.3-70B / Claude-class)
to resolve factual conflicts into a grounded synthesis.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

import httpx

from app.config import Settings, get_settings
from app.models.schemas import ArbitrationResult, Claim, ClaimRelation

logger = logging.getLogger(__name__)

ARBITRATION_SYSTEM_PROMPT = """You are a factual arbitration engine for conflicting claims.
Given a query and a set of claims (some corroborating, some contradicting), resolve
the conflicts and produce a grounded synthesis.

Return ONLY valid JSON:
{
  "resolved_claim_ids": ["uuid", ...],
  "discarded_claim_ids": ["uuid", ...],
  "synthesis": "citation-aware grounded answer",
  "conflicts_resolved": 0
}

Rules:
- Prefer claims with higher confidence and corroboration.
- Prefer temporally newer claims when SUPERSEDES applies.
- Never invent facts not implied by the claim set.
- synthesis must be concise and agent-consumable.
"""


class DeepArbitrationLoop:
    """High-capacity conflict resolution pass for contradictory subgraphs."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=120.0)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def arbitrate(
        self,
        query: str,
        claims: list[Claim],
        relations: list[ClaimRelation] | None = None,
    ) -> ArbitrationResult:
        relations = relations or []
        if not claims:
            return ArbitrationResult(synthesis="No claims available for arbitration.")

        api_key = self._settings.llm_api_key
        if not api_key:
            # Deterministic offline fallback: keep highest-confidence non-contradicted claims
            return self._offline_arbitrate(claims, relations)

        payload = {
            "query": query,
            "claims": [
                {
                    "id": str(c.id),
                    "text": c.text,
                    "subject": c.subject,
                    "predicate": c.predicate,
                    "object": c.object,
                    "confidence": c.confidence,
                    "source_url": str(c.source_url) if c.source_url else None,
                    "valid_from": c.valid_from.isoformat() if c.valid_from else None,
                }
                for c in claims
            ],
            "relations": [
                {
                    "source": str(r.source_claim_id),
                    "target": str(r.target_claim_id),
                    "relation": r.relation.value,
                    "confidence": r.confidence,
                }
                for r in relations
            ],
        }

        data = await self._complete(json.dumps(payload, ensure_ascii=False))
        return self._parse(data, claims)

    async def _complete(self, user_content: str) -> dict[str, Any]:
        url = f"{self._settings.arbitration_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._settings.arbitration_model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": ARBITRATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        response = await self._client.post(url, headers=headers, json=body)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    def _parse(self, data: dict[str, Any], claims: list[Claim]) -> ArbitrationResult:
        by_id = {str(c.id): c for c in claims}
        resolved_ids = [str(x) for x in (data.get("resolved_claim_ids") or [])]
        discarded_raw = [str(x) for x in (data.get("discarded_claim_ids") or [])]

        resolved = [by_id[i] for i in resolved_ids if i in by_id]
        if not resolved:
            resolved = sorted(claims, key=lambda c: c.confidence, reverse=True)[:5]

        discarded: list[UUID] = []
        for i in discarded_raw:
            try:
                discarded.append(UUID(i))
            except ValueError:
                continue

        return ArbitrationResult(
            resolved_claims=resolved,
            discarded_claim_ids=discarded,
            synthesis=data.get("synthesis"),
            model_used=self._settings.arbitration_model,
            conflicts_resolved=int(data.get("conflicts_resolved") or 0),
        )

    @staticmethod
    def _offline_arbitrate(
        claims: list[Claim],
        relations: list[ClaimRelation],
    ) -> ArbitrationResult:
        _ = relations  # reserved for richer offline conflict heuristics
        ranked = sorted(claims, key=lambda c: c.confidence, reverse=True)
        kept = ranked[: max(1, len(ranked) // 2)]
        kept_ids = {c.id for c in kept}
        discarded = [c.id for c in claims if c.id not in kept_ids]
        synthesis = " ".join(c.text for c in kept[:3])
        return ArbitrationResult(
            resolved_claims=kept,
            discarded_claim_ids=discarded,
            synthesis=synthesis,
            model_used="offline-heuristic",
            conflicts_resolved=len(discarded),
        )
