"""
Module 2: Zero-Latency Fact Extractor.

Async Llama-3.1-8B-Instruct extraction (Groq / Together / vLLM via httpx)
that turns Exa search snippets into strict ``AtomicClaim`` objects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse
from uuid import UUID, uuid4

import httpx

from app.config import Settings, get_settings
from app.models.schemas import AtomicClaim, ExtractorTier, SearchResult
from app.services.claim_cache import extraction_cache
from app.services.claim_relevance import select_query_window

logger = logging.getLogger(__name__)

ATOMIC_CLAIM_SYSTEM_PROMPT = """You are a zero-latency factual claim extraction engine.

Your ONLY job is to parse the provided document text into Atomic Claims:
non-decomposable, single-assertion factual statements. Each claim must stand alone
as exactly one fact — never compound ("X and Y"), never hedged multi-part prose.

Return ONLY a JSON object (no markdown, no commentary) with this exact schema:
{
  "claims": [
    {
      "statement": "<one atomic factual assertion>",
      "confidence_score": <float 0.0-1.0>,
      "entities": ["<Entity Name>", "..."]
    }
  ]
}

Hard rules:
1. Atomicity: one subject–predicate–object assertion per statement.
2. Grounding: every statement MUST be entailed by the source text; never invent facts.
3. Precision over recall: omit uncertain, opinion, or promotional language.
4. Entities: list surface-form named entities explicitly mentioned in that statement.
5. confidence_score reflects textual support strength (direct quote ≈ 0.9+, weak ≈ 0.5).
6. If the text contains no extractable facts, return {"claims": []}.
7. Do NOT wrap JSON in markdown code fences. Do NOT add trailing commentary.
8. When a research_query is provided, PRIORITIZE atomic claims that directly
   answer it (correct entity, metric, and time). Skip tangential quotes.
"""

_MD_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*([\s\S]*?)\s*```\s*$",
    re.MULTILINE,
)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


@dataclass(frozen=True, slots=True)
class ExtractionSnippet:
    """Raw Exa (or other) snippet ready for concurrent atomic extraction."""

    text: str
    document_url: str
    source_domain: str
    document_id: str | None = None


class ClaimExtractor:
    """
    Asynchronous Atomic Claim extractor backed by an 8B LLM.

    Talks to any OpenAI-compatible Chat Completions endpoint (Groq, Together AI,
    or local vLLM) over ``httpx.AsyncClient``. Concurrent page batches use
    ``asyncio.gather`` with a bounded semaphore for sub-second multi-page runs.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        max_concurrency: int = 3,
        max_text_chars: int = 12_000,
        max_retries: int = 4,
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=60.0)
        # Keep concurrency low to stay under Groq free-tier rate limits.
        self._max_concurrency = max(1, max_concurrency)
        self._max_text_chars = max_text_chars
        self._max_retries = max(1, max_retries)

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned by this instance."""
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(
        self,
        text: str,
        *,
        document_url: str,
        source_domain: str,
        tier: ExtractorTier = ExtractorTier.FAST,
        research_query: str | None = None,
        use_cache: bool = True,
    ) -> list[AtomicClaim]:
        """
        Parse a single raw text snippet into ``AtomicClaim`` objects.

        Parameters
        ----------
        text:
            Raw document / Exa snippet body.
        document_url:
            Absolute citation URL for every emitted claim.
        source_domain:
            Publisher hostname (e.g. ``sec.gov``).
        tier:
            Model cascade tier (``FAST`` = 8B, ``DENSE`` = 70B).
        research_query:
            Optional focus query — biases extraction toward answering claims.
        use_cache:
            Reuse prior extractions for the same URL/body/model when True.
        """
        if not (text or "").strip():
            return []

        domain = (source_domain or "").strip() or domain_from_url(document_url)
        url = (document_url or "").strip()
        if not url:
            raise ValueError("document_url is required for citation-anchored claims")
        if not domain:
            raise ValueError("source_domain could not be inferred; pass it explicitly")

        window = text
        if research_query:
            window = select_query_window(
                text,
                research_query,
                max_chars=min(self._max_text_chars, self._settings.research_max_chars),
            )
        else:
            window = text[: self._max_text_chars]

        model = self._model_for_tier(tier)
        cache_key = extraction_cache.make_key(
            url=url,
            text=window,
            model=model,
            max_chars=self._max_text_chars,
        )
        if use_cache:
            cached = extraction_cache.get(cache_key)
            if cached is not None:
                logger.info("extract.cache_hit url=%s claims=%d", url, len(cached))
                return cached

        focus_block = ""
        if research_query:
            focus_block = f"research_query: {research_query.strip()}\n"

        user_content = (
            f"{focus_block}"
            f"source_domain: {domain}\n"
            f"document_url: {url}\n\n"
            f"document_text:\n{window}"
        )
        raw = await self._complete(user_content, model=model)
        payload = parse_llm_json(raw)
        claims = self._to_atomic_claims(
            payload,
            document_url=url,
            source_domain=domain,
        )
        if use_cache:
            extraction_cache.put(cache_key, claims)
        return claims

    async def extract_many(
        self,
        snippets: Iterable[ExtractionSnippet],
        *,
        tier: ExtractorTier = ExtractorTier.FAST,
        max_concurrency: int | None = None,
    ) -> list[AtomicClaim]:
        """
        Extract claims from 5–10 (or more) pages concurrently via ``asyncio.gather``.
        """
        items = [s for s in snippets if (s.text or "").strip()]
        if not items:
            return []

        limit = max(1, max_concurrency or self._max_concurrency)
        semaphore = asyncio.Semaphore(limit)

        async def _one(snippet: ExtractionSnippet) -> list[AtomicClaim]:
            async with semaphore:
                try:
                    return await self.extract(
                        snippet.text,
                        document_url=snippet.document_url,
                        source_domain=snippet.source_domain,
                        tier=tier,
                    )
                except Exception:
                    logger.exception(
                        "Atomic claim extraction failed for %s",
                        snippet.document_url,
                    )
                    return []

        batches = await asyncio.gather(*[_one(s) for s in items])
        claims: list[AtomicClaim] = []
        for batch in batches:
            claims.extend(batch)
        return claims

    async def extract_from_documents(
        self,
        documents: list[SearchResult],
        *,
        tier: ExtractorTier = ExtractorTier.FAST,
        research_query: str | None = None,
    ) -> list[AtomicClaim]:
        """Convenience wrapper: map Exa ``SearchResult`` hits → concurrent extraction."""
        snippets: list[ExtractionSnippet] = []
        for doc in documents:
            url = str(doc.url) if doc.url else ""
            if not url:
                continue
            domain = domain_from_url(url)
            text = doc.text or ""
            if doc.chunks and not text:
                text = "\n".join(ch.text for ch in doc.chunks)
            if not text.strip():
                continue
            snippets.append(
                ExtractionSnippet(
                    text=text,
                    document_url=url,
                    source_domain=domain,
                    document_id=doc.id,
                )
            )
        return await self.extract_many_focused(
            snippets,
            tier=tier,
            research_query=research_query,
        )

    async def extract_many_focused(
        self,
        snippets: Iterable[ExtractionSnippet],
        *,
        tier: ExtractorTier = ExtractorTier.FAST,
        research_query: str | None = None,
        max_concurrency: int | None = None,
    ) -> list[AtomicClaim]:
        """Like ``extract_many`` but passes ``research_query`` into each extract call."""
        items = [s for s in snippets if (s.text or "").strip()]
        if not items:
            return []

        limit = max(1, max_concurrency or self._max_concurrency)
        semaphore = asyncio.Semaphore(limit)

        async def _one(snippet: ExtractionSnippet) -> list[AtomicClaim]:
            async with semaphore:
                try:
                    return await self.extract(
                        snippet.text,
                        document_url=snippet.document_url,
                        source_domain=snippet.source_domain,
                        tier=tier,
                        research_query=research_query,
                    )
                except Exception:
                    logger.exception(
                        "Atomic claim extraction failed for %s",
                        snippet.document_url,
                    )
                    return []

        batches = await asyncio.gather(*[_one(s) for s in items])
        claims: list[AtomicClaim] = []
        for batch in batches:
            claims.extend(batch)
        return claims

    async def extract_claims_batch(
        self,
        documents: list[SearchResult],
        *,
        tier: ExtractorTier = ExtractorTier.FAST,
        max_concurrency: int | None = None,
        research_query: str | None = None,
        max_docs: int | None = None,
    ) -> list[AtomicClaim]:
        """
        Parallel claim extraction across documents (default: Llama-3.1-8B FAST tier).

        Latency knobs: ``max_docs`` caps pages extracted; ``research_query`` enables
        query-window truncation + focused prompting + extraction cache.
        """
        docs = list(documents)
        if max_docs is not None:
            docs = docs[: max(0, max_docs)]

        snippets: list[ExtractionSnippet] = []
        for doc in docs:
            url = str(doc.url) if doc.url else ""
            text = doc.text or ""
            if doc.chunks and not text:
                text = "\n".join(ch.text for ch in doc.chunks)
            if not url or not text.strip():
                continue
            snippets.append(
                ExtractionSnippet(
                    text=text,
                    document_url=url,
                    source_domain=domain_from_url(url),
                    document_id=doc.id,
                )
            )
        return await self.extract_many_focused(
            snippets,
            tier=tier,
            research_query=research_query,
            max_concurrency=max_concurrency,
        )

    # ------------------------------------------------------------------
    # LLM transport
    # ------------------------------------------------------------------

    def _model_for_tier(self, tier: ExtractorTier) -> str:
        if tier == ExtractorTier.DENSE:
            return self._settings.llm_model_dense
        return self._settings.llm_model_fast

    async def _complete(self, user_content: str, *, model: str) -> str:
        api_key = self._settings.llm_api_key
        base = self._settings.llm_base_url.rstrip("/")
        if not api_key and not _is_local_endpoint(base):
            raise RuntimeError(
                "Neither GROQ_API_KEY nor TOGETHER_API_KEY is configured "
                "(required unless LLM_BASE_URL points at a local vLLM server)"
            )

        url = f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body: dict[str, Any] = {
            "model": model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": ATOMIC_CLAIM_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        # Groq / Together support json_object; local vLLM may ignore or reject it.
        if not _is_local_endpoint(base):
            body["response_format"] = {"type": "json_object"}

        response: httpx.Response | None = None
        for attempt in range(self._max_retries):
            response = await self._client.post(url, headers=headers, json=body)
            if response.status_code == 400 and "response_format" in body:
                body.pop("response_format", None)
                response = await self._client.post(url, headers=headers, json=body)

            if response.status_code != 429:
                break

            # Exponential backoff + honor Retry-After when present.
            retry_after = response.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else (0.75 * (2**attempt))
            except ValueError:
                delay = 0.75 * (2**attempt)
            delay = min(max(delay, 0.5), 20.0)
            logger.warning(
                "LLM rate-limited (429); retry %d/%d after %.1fs",
                attempt + 1,
                self._max_retries,
                delay,
            )
            await asyncio.sleep(delay)

        assert response is not None
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            # Some OpenAI-compatible servers return content parts
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    # ------------------------------------------------------------------
    # Claim materialization
    # ------------------------------------------------------------------

    def _to_atomic_claims(
        self,
        payload: dict[str, Any] | list[Any],
        *,
        document_url: str,
        source_domain: str,
    ) -> list[AtomicClaim]:
        if isinstance(payload, list):
            raw_claims = payload
        else:
            raw_claims = payload.get("claims") or payload.get("atomic_claims") or []

        if not isinstance(raw_claims, list):
            logger.warning("LLM payload missing claims list: %s", type(raw_claims))
            return []

        claims: list[AtomicClaim] = []
        for item in raw_claims:
            claim = self._coerce_atomic_claim(
                item,
                document_url=document_url,
                source_domain=source_domain,
            )
            if claim is not None:
                claims.append(claim)
        return claims

    @staticmethod
    def _coerce_atomic_claim(
        item: Any,
        *,
        document_url: str,
        source_domain: str,
    ) -> AtomicClaim | None:
        if not isinstance(item, dict):
            return None

        statement = (
            item.get("statement")
            or item.get("text")
            or item.get("claim")
            or ""
        )
        statement = str(statement).strip()
        if not statement:
            return None

        raw_entities = item.get("entities") or []
        entities: list[str] = []
        if isinstance(raw_entities, list):
            for ent in raw_entities:
                if isinstance(ent, str) and ent.strip():
                    entities.append(ent.strip())
                elif isinstance(ent, dict) and ent.get("name"):
                    entities.append(str(ent["name"]).strip())

        confidence = item.get("confidence_score", item.get("confidence", 0.7))
        try:
            confidence_f = float(confidence)
        except (TypeError, ValueError):
            confidence_f = 0.7
        confidence_f = max(0.0, min(1.0, confidence_f))

        claim_id = item.get("claim_id") or str(uuid4())
        try:
            return AtomicClaim(
                claim_id=claim_id,
                statement=statement,
                confidence_score=confidence_f,
                entities=entities,
                source_domain=source_domain,
                citation_anchor=document_url,
                embedding=item.get("embedding"),
                consensus_score=float(item.get("consensus_score", 0.0) or 0.0),
            )
        except Exception:
            logger.debug("Skipping malformed claim payload: %s", item, exc_info=True)
            return None


# Public alias matching the research-pipeline naming in architecture docs.
Extractor = ClaimExtractor


# ---------------------------------------------------------------------------
# JSON repair helpers
# ---------------------------------------------------------------------------


def parse_llm_json(raw: str) -> dict[str, Any] | list[Any]:
    """
    Parse LLM output into JSON with progressive repair fallbacks.

    Handles markdown fences, leading/trailing prose, and trailing commas.
    """
    if raw is None:
        raise ValueError("empty LLM response")

    text = str(raw).strip()
    if not text:
        raise ValueError("empty LLM response")

    candidates = _json_candidates(text)
    last_error: Exception | None = None
    for candidate in candidates:
        for variant in (candidate, _repair_trailing_commas(candidate)):
            try:
                parsed = json.loads(variant)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except json.JSONDecodeError as exc:
                last_error = exc
                continue

    raise ValueError(
        f"Failed to parse LLM JSON after repairs: {last_error}"
    ) from last_error


def _json_candidates(text: str) -> list[str]:
    """Generate increasingly aggressive candidate strings for JSON parsing."""
    candidates: list[str] = [text]

    fenced = _MD_FENCE_RE.match(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    # Strip any leading/trailing fences even if nested in prose
    stripped_fences = re.sub(r"```(?:json|JSON)?", "", text).replace("```", "").strip()
    if stripped_fences not in candidates:
        candidates.append(stripped_fences)

    obj_match = _JSON_OBJECT_RE.search(text)
    if obj_match:
        candidates.append(obj_match.group(0))

    arr_match = _JSON_ARRAY_RE.search(text)
    if arr_match:
        candidates.append(arr_match.group(0))
        repaired_arr = _repair_or_empty_array(arr_match.group(0))
        try:
            parsed_arr = json.loads(repaired_arr)
            if isinstance(parsed_arr, list):
                candidates.append(json.dumps({"claims": parsed_arr}))
        except json.JSONDecodeError:
            pass

    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _repair_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common LLM JSON defect)."""
    prev = None
    repaired = text
    # Iterate — nested trailing commas may need more than one pass
    while prev != repaired:
        prev = repaired
        repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    return repaired


def _repair_or_empty_array(text: str) -> str:
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        repaired = _repair_trailing_commas(text)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            return "[]"


def domain_from_url(url: str) -> str:
    """Extract a bare hostname from a URL for ``source_domain``."""
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _is_local_endpoint(base_url: str) -> bool:
    host = urlparse(base_url).hostname or ""
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def atomic_claims_to_legacy_claims(
    atomic_claims: list[AtomicClaim],
    *,
    document_id: str | None = None,
) -> list[Any]:
    """
    Adapter for downstream modules still typed on the legacy ``Claim`` model.

    Imported lazily to avoid circular imports at module load time.
    """
    from app.models.schemas import Claim, Entity

    legacy: list[Claim] = []
    for ac in atomic_claims:
        legacy.append(
            Claim(
                id=UUID(ac.claim_id),
                text=ac.statement,
                confidence=ac.confidence_score,
                entities=[Entity(name=name) for name in ac.entities],
                source_url=ac.citation_anchor,
                source_title=ac.source_domain,
                document_id=document_id,
            )
        )
    return legacy
