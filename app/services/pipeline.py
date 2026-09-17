"""
End-to-end Synthetica-Core pipeline orchestrator.

Agent Swarm → SSM+BM25 → Fact Extractor → 4D Graph → CAMMR/Entropy
  → Fast Path  OR  Deep Arbitration → Grounded Synthesis
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.config import Settings, get_settings
from app.models.schemas import (
    CitationAnchor,
    Claim,
    Entity,
    GraphContext,
    PipelinePath,
    QueryRequest,
    QueryResponse,
    SourceRef,
)
from app.services.arbitration import DeepArbitrationLoop
from app.services.cammr_engine import CAMMREngine
from app.services.extractor import ClaimExtractor, atomic_claims_to_legacy_claims
from app.services.graph_service import GraphService
from app.services.ssm_retrieval import StateSpaceRetriever

logger = logging.getLogger(__name__)


class SyntheticaPipeline:
    """Wires Modules 1–5 into a single async request path."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: StateSpaceRetriever | None = None,
        extractor: ClaimExtractor | None = None,
        graph: GraphService | None = None,
        cammr: CAMMREngine | None = None,
        arbitration: DeepArbitrationLoop | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or StateSpaceRetriever(self.settings)
        self.extractor = extractor or ClaimExtractor(self.settings)
        self.graph = graph or GraphService(self.settings)
        self.cammr = cammr or CAMMREngine(
            lam=self.settings.cammr_lambda,
            entropy_threshold=self.settings.entropy_threshold,
        )
        self.arbitration = arbitration or DeepArbitrationLoop(self.settings)
        self._graph_ready = False

    async def startup(self) -> None:
        try:
            await self.graph.connect()
            await self.graph.init_schema()
            self._graph_ready = True
        except Exception:
            logger.warning("Neo4j unavailable; graph writes will be skipped", exc_info=True)
            self._graph_ready = False

    async def shutdown(self) -> None:
        await self.extractor.aclose()
        await self.arbitration.aclose()
        await self.graph.close()

    async def run(self, request: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()

        # Module 1 — Agent State-Space Retrieval
        agent_state = request.agent_state
        if agent_state is None:
            query = request.query
        else:
            query = agent_state.query or request.query

        documents = await self.retriever.retrieve(
            query=query,
            agent_state=agent_state,
            num_results=request.num_results,
        )

        # Module 2 — Zero-Latency Fact Extractor (AtomicClaim → legacy Claim)
        claims: list[Claim] = []
        atomic_claims = []
        if request.extract_claims and documents:
            try:
                atomic_claims = await self.extractor.extract_from_documents(
                    documents,
                    tier=request.extractor_tier,
                )
                url_to_doc_id = {
                    str(doc.url): doc.id for doc in documents if doc.url
                }
                for atomic in atomic_claims:
                    claims.extend(
                        atomic_claims_to_legacy_claims(
                            [atomic],
                            document_id=url_to_doc_id.get(atomic.citation_anchor),
                        )
                    )
            except Exception:
                logger.exception("Fact extraction failed; continuing without claims")

        entities = self._unique_entities(claims)
        triples = [t for c in claims for t in c.triples]

        # Module 3 — 4D Spatiotemporal Graph (AtomicClaim ingest)
        relations = []
        graph_context: GraphContext | None = None
        if request.build_graph and self._graph_ready and atomic_claims:
            try:
                ingest_result = await self.graph.ingest_claims(atomic_claims)
                relations = ingest_result.relations
            except Exception:
                logger.exception("Graph ingest failed")

        # Module 4 — CAMMR + H(R_q)
        cammr_result = self.cammr.rerank(
            query,
            documents,
            claims,
            relations,
            top_k=request.num_results,
            force_arbitration=request.force_arbitration,
            entropy_threshold=request.entropy_threshold,
        )

        arbitration = None
        synthesis: str | None = None
        final_claims = cammr_result.selected_claims

        # Entropy gate
        if cammr_result.path == PipelinePath.DEEP_ARBITRATION:
            # Module 5 — Deep Arbitration Loop
            arbitration = await self.arbitration.arbitrate(
                query,
                cammr_result.selected_claims,
                relations,
            )
            final_claims = arbitration.resolved_claims
            synthesis = arbitration.synthesis
        else:
            synthesis = self._fast_path_synthesis(final_claims)

        citations = [
            CitationAnchor(
                claim=c,
                source=SourceRef(
                    url=c.source_url,
                    title=c.source_title,
                )
                if c.source_url or c.source_title
                else None,
            )
            for c in final_claims
        ]

        if self._graph_ready and (entities or final_claims):
            try:
                entity_names = [e.name for e in entities] or [
                    name
                    for claim in final_claims
                    for name in (ent.name for ent in claim.entities)
                ]
                if entity_names:
                    graph_context = await self.graph.get_subgraph_for_query(entity_names)
                else:
                    graph_context = await self.graph.fetch_query_subgraph(
                        [c.id for c in final_claims]
                    )
                if graph_context is not None:
                    graph_context.subgraph_entropy = cammr_result.subgraph_entropy
            except Exception:
                logger.exception("Subgraph fetch failed")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return QueryResponse(
            query=query,
            path=cammr_result.path,
            results=cammr_result.ranked,
            claims=final_claims,
            entities=entities,
            triples=triples,
            citations=citations,
            graph_context=graph_context,
            subgraph_entropy=cammr_result.subgraph_entropy,
            arbitration=arbitration,
            synthesis=synthesis,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _unique_entities(claims: list[Claim]) -> list[Entity]:
        seen: set[str] = set()
        out: list[Entity] = []
        for claim in claims:
            for ent in claim.entities:
                key = ent.name.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(ent)
        return out

    @staticmethod
    def _fast_path_synthesis(claims: list[Claim]) -> str | None:
        if not claims:
            return None
        top = sorted(claims, key=lambda c: c.confidence, reverse=True)[:5]
        return " ".join(c.text for c in top)

    def describe(self) -> dict[str, Any]:
        return {
            "modules": [
                "ssm_retrieval",
                "fact_extractor",
                "knowledge_graph",
                "cammr_entropy",
                "deep_arbitration",
            ],
            "graph_ready": self._graph_ready,
            "entropy_threshold": self.settings.entropy_threshold,
            "retriever": self.retriever.describe(),
        }
