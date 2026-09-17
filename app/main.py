"""FastAPI application entrypoint for Synthetica-Core."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, Request

from app import __version__
from app.config import Settings, get_settings
from app.models.schemas import (
    AtomicClaim,
    ClaimRelationType,
    ExtractorTier,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ResearchQueryRequest,
    SearchResult,
    SynthesisResponse,
)
from app.services.cammr_engine import CAMMREngine
from app.services.claim_cache import extraction_cache
from app.services.claim_relevance import (
    boost_numeric_consensus,
    filter_claims_for_query,
)
from app.services.conflict_resolver import build_conflict_aware_answer
from app.services.exa_service import ExaService
from app.services.extractor import Extractor
from app.services.graph_service import GraphService, IngestResult
from app.services.pipeline import SyntheticaPipeline
from app.utils.entropy import calculate_subgraph_entropy, tokenize

logger = logging.getLogger(__name__)

# CAMMR slate size returned in SynthesisResponse.
_CAMMR_SHOWN_TOP_K = 8


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Wire Modules 1–5 on startup; tear down clients on shutdown."""
    settings = get_settings()

    exa = ExaService(settings)
    extractor = Extractor(
        settings,
        max_concurrency=3,
        max_text_chars=settings.research_max_chars,
    )
    graph = GraphService(settings)
    cammr = CAMMREngine(
        lam=settings.cammr_lambda,
        entropy_threshold=settings.entropy_threshold,
    )
    extraction_cache.ttl_seconds = settings.claim_cache_ttl_seconds
    pipeline = SyntheticaPipeline(
        settings,
        extractor=extractor,
        graph=graph,
        cammr=cammr,
    )

    graph_ready = False
    try:
        await graph.connect()
        await graph.init_schema()
        graph_ready = True
    except Exception:
        logger.warning(
            "Neo4j unavailable; /v1/research/query will skip graph writes",
            exc_info=True,
        )

    # Keep pipeline.startup() for legacy /v1/query; share the same graph client.
    pipeline._graph_ready = graph_ready  # noqa: SLF001 — intentional shared readiness

    app.state.settings = settings
    app.state.exa = exa
    app.state.extractor = extractor
    app.state.graph = graph
    app.state.cammr = cammr
    app.state.graph_ready = graph_ready
    app.state.pipeline = pipeline

    yield

    await extractor.aclose()
    await pipeline.arbitration.aclose()
    await graph.close()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "High-throughput search middleware: SSM+BM25 retrieval, "
            "model-cascade fact extraction, 4D Neo4j graph, CAMMR re-ranking, "
            "and entropy-gated deep arbitration."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def healthcheck() -> HealthResponse:
        """Liveness / readiness probe."""
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            version=__version__,
            environment=settings.app_env,
        )

    @app.get("/", tags=["system"])
    async def root() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "docs": "/docs",
            "health": "/health",
            "research": "/v1/research/query",
        }

    @app.get("/v1/architecture", tags=["system"])
    async def architecture(request: Request) -> dict:
        """Describe the live five-module pipeline."""
        pipeline: SyntheticaPipeline = request.app.state.pipeline
        return pipeline.describe()

    @app.post("/v1/query", response_model=QueryResponse, tags=["pipeline"])
    async def query(body: QueryRequest, request: Request) -> QueryResponse:
        """
        End-to-end agent query.

        Routes to fast-path citation JSON when H(R_q) ≤ threshold,
        otherwise enters Module 5 deep arbitration.
        """
        pipeline: SyntheticaPipeline = request.app.state.pipeline
        return await pipeline.run(body)

    @app.post(
        "/v1/research/query",
        response_model=SynthesisResponse,
        tags=["research"],
    )
    async def research_query(
        body: ResearchQueryRequest,
        request: Request,
    ) -> SynthesisResponse:
        """
        End-to-end research orchestration:

        Exa (top-5) → Llama-3.1-8B batch extract → Neo4j ingest / contradictions
        → H(R_q) → CAMMR re-rank → ``SynthesisResponse``.
        """
        return await _run_research_pipeline(body, request)

    return app


async def _run_research_pipeline(
    body: ResearchQueryRequest,
    request: Request,
) -> SynthesisResponse:
    """
    Cooked research path:

    A) query-conditioned filter + numeric consensus
    B) conflict-aware grounded answer
    C) latency knobs — top-k docs, query windows, extraction cache
    """
    settings: Settings = request.app.state.settings
    exa: ExaService = request.app.state.exa
    extractor: Extractor = request.app.state.extractor
    graph: GraphService = request.app.state.graph
    cammr: CAMMREngine = request.app.state.cammr
    graph_ready: bool = bool(request.app.state.graph_ready)

    session_id = str(uuid4())
    t_total = time.perf_counter()
    step_ms: dict[str, float] = {}

    logger.info(
        "research.query.start session_id=%s agent_id=%s query=%r",
        session_id,
        body.agent_id,
        body.query[:120],
    )

    # ------------------------------------------------------------------
    # 1) Exa neural search — capped top-N (latency)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    documents: list[SearchResult] = await exa.search(
        body.query,
        num_results=settings.research_top_n,
        include_text=True,
    )
    step_ms["exa_search"] = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "research.step.exa_search session_id=%s docs=%d duration_ms=%.2f",
        session_id,
        len(documents),
        step_ms["exa_search"],
    )

    # ------------------------------------------------------------------
    # 2) Parallel 8B extraction — top docs only + query window + cache
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    atomic_claims: list[AtomicClaim] = []
    if documents:
        atomic_claims = await extractor.extract_claims_batch(
            documents,
            tier=ExtractorTier.FAST,
            research_query=body.query,
            max_docs=settings.research_extract_docs,
        )
    step_ms["extract_claims_batch"] = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "research.step.extract_claims_batch session_id=%s claims=%d "
        "cache=%s duration_ms=%.2f",
        session_id,
        len(atomic_claims),
        extraction_cache.stats(),
        step_ms["extract_claims_batch"],
    )

    # Credibility gate
    atomic_claims = [
        c
        for c in atomic_claims
        if c.confidence_score >= body.required_credibility
    ]

    # ------------------------------------------------------------------
    # 2b) Query-conditioned filter (kill Intel/AMD on NVIDIA queries)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    filtered = filter_claims_for_query(
        atomic_claims,
        body.query,
        min_score=settings.research_min_relevance,
        require_primary_entity=True,
    )
    atomic_claims = filtered.kept
    step_ms["query_filter"] = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "research.step.query_filter session_id=%s kept=%d rejected=%d duration_ms=%.2f",
        session_id,
        len(filtered.kept),
        len(filtered.rejected),
        step_ms["query_filter"],
    )

    # ------------------------------------------------------------------
    # 3) Neo4j ingest + contradiction detection
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    ingest_result = IngestResult()
    if graph_ready and atomic_claims:
        try:
            ingest_result = await graph.ingest_claims(atomic_claims)
        except Exception:
            logger.exception(
                "research.step.ingest_claims failed session_id=%s",
                session_id,
            )
    step_ms["ingest_claims"] = (time.perf_counter() - t0) * 1000.0
    contradiction_count = sum(
        1
        for rel in ingest_result.relations
        if rel.relation == ClaimRelationType.CONTRADICTS
    )
    _apply_graph_consensus(atomic_claims, ingest_result)
    # Numeric cross-domain corroboration (A) — $10.32B == $10.32 billion
    boost_numeric_consensus(atomic_claims)
    logger.info(
        "research.step.ingest_claims session_id=%s ingested=%d relations=%d "
        "contradictions=%d duration_ms=%.2f",
        session_id,
        ingest_result.ingested,
        len(ingest_result.relations),
        contradiction_count,
        step_ms["ingest_claims"],
    )

    # Single consensus write (skip pre+post double round-trip for latency)
    if graph_ready and atomic_claims:
        try:
            await graph.update_consensus_scores(atomic_claims)
        except Exception:
            logger.exception(
                "research.step.update_consensus failed session_id=%s",
                session_id,
            )

    # ------------------------------------------------------------------
    # 4) Subgraph information entropy H(R_q)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    entropy_score = calculate_subgraph_entropy(atomic_claims, contradiction_count)
    step_ms["subgraph_entropy"] = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "research.step.subgraph_entropy session_id=%s H_Rq=%.4f duration_ms=%.2f",
        session_id,
        entropy_score,
        step_ms["subgraph_entropy"],
    )

    # ------------------------------------------------------------------
    # 5) CAMMR — relevance-heavy after filter (λ↑, μ keeps consensus)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    query_embedding = _hash_embed(
        body.query,
        dim=max(settings.matryoshka_dim_list or [256]),
    )
    candidate_payloads = [
        _claim_to_candidate_dict(claim, query_dim=int(query_embedding.size))
        for claim in atomic_claims
    ]
    ranked_claims = cammr.compute_cammr_reranking(
        query_embedding,
        candidate_payloads,
        top_k=_CAMMR_SHOWN_TOP_K,
        lambda_param=0.50,
        mu_param=0.30,
        phi_param=0.15,
    )
    step_ms["cammr_reranking"] = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "research.step.cammr_reranking session_id=%s ranked=%d duration_ms=%.2f",
        session_id,
        len(ranked_claims),
        step_ms["cammr_reranking"],
    )

    # ------------------------------------------------------------------
    # 6) Conflict-aware grounded answer (B)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    grounded = build_conflict_aware_answer(
        body.query,
        ranked_claims,
        rejected_by_filter=filtered.rejected,
    )
    step_ms["conflict_resolve"] = (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------------------
    # 7) SynthesisResponse
    # ------------------------------------------------------------------
    factual_consensus_index = _aggregate_consensus(ranked_claims)
    if grounded.confidence > 0:
        factual_consensus_index = max(factual_consensus_index, grounded.confidence * 0.85)

    status = _resolve_status(
        entropy_score=entropy_score,
        threshold=settings.entropy_threshold,
        claims=ranked_claims,
        required_credibility=body.required_credibility,
        grounded=grounded,
    )

    total_ms = (time.perf_counter() - t_total) * 1000.0
    logger.info(
        "research.query.complete session_id=%s status=%s claims=%d "
        "consensus=%.4f entropy=%.4f act_safe=%s total_ms=%.2f steps_ms=%s",
        session_id,
        status,
        len(ranked_claims),
        factual_consensus_index,
        entropy_score,
        grounded.act_safe,
        total_ms,
        {k: round(v, 2) for k, v in step_ms.items()},
    )

    return SynthesisResponse(
        session_id=session_id,
        status=status,
        factual_consensus_index=float(min(1.0, factual_consensus_index)),
        claims=ranked_claims,
        entropy_score=float(entropy_score),
        grounded_answer=grounded,
        latency_ms=total_ms,
        step_ms={k: round(v, 2) for k, v in step_ms.items()},
    )


def _apply_graph_consensus(
    claims: list[AtomicClaim],
    ingest_result: IngestResult,
) -> None:
    """Elevate ``consensus_score`` from corroboration vs contradiction mass."""
    if not claims:
        return

    corr: dict[str, float] = defaultdict(float)
    contra: dict[str, float] = defaultdict(float)
    for rel in ingest_result.relations:
        src = str(rel.source_claim_id)
        tgt = str(rel.target_claim_id)
        mass = float(rel.confidence)
        if rel.relation == ClaimRelationType.CORROBORATES:
            corr[src] += mass
            corr[tgt] += mass
        elif rel.relation == ClaimRelationType.CONTRADICTS:
            contra[src] += mass
            contra[tgt] += mass

    # Distinct publisher domains mentioning overlapping entities → soft boost
    domain_by_entity: dict[str, set[str]] = defaultdict(set)
    for claim in claims:
        for ent in claim.entities:
            key = ent.strip().lower()
            if key:
                domain_by_entity[key].add(claim.source_domain.lower())

    for claim in claims:
        c_mass = corr[claim.claim_id]
        d_mass = contra[claim.claim_id]
        total = c_mass + d_mass
        if total > 0:
            base = c_mass / total
        else:
            base = claim.confidence_score

        cross_domain = 0.0
        if claim.entities:
            spans = [
                len(domain_by_entity[e.strip().lower()])
                for e in claim.entities
                if e.strip() and e.strip().lower() in domain_by_entity
            ]
            if spans:
                # More than one domain covering an entity ⇒ corroboration signal
                cross_domain = min(1.0, (max(spans) - 1) / 3.0)

        claim.consensus_score = float(max(0.0, min(1.0, 0.7 * base + 0.3 * cross_domain)))


def _claim_to_candidate_dict(claim: AtomicClaim, *, query_dim: int) -> dict[str, Any]:
    """Serialize an AtomicClaim into the CAMMR candidate dict shape."""
    embedding = claim.embedding
    if not embedding:
        embedding = _hash_embed(claim.statement, dim=query_dim).tolist()
    return {
        "claim_id": claim.claim_id,
        "statement": claim.statement,
        "confidence_score": claim.confidence_score,
        "entities": list(claim.entities),
        "source_domain": claim.source_domain,
        "citation_anchor": claim.citation_anchor,
        "embedding": embedding,
        "consensus_score": claim.consensus_score,
    }


def _hash_embed(text: str, *, dim: int) -> np.ndarray:
    """Deterministic bag-of-tokens embedding (no external model required)."""
    vec = np.zeros(dim, dtype=np.float64)
    for token in tokenize(text):
        idx = hash(token) % dim
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _aggregate_consensus(claims: list[AtomicClaim]) -> float:
    if not claims:
        return 0.0
    return float(sum(c.consensus_score for c in claims) / len(claims))


def _resolve_status(
    *,
    entropy_score: float,
    threshold: float,
    claims: list[AtomicClaim],
    required_credibility: float,
    grounded: Any | None = None,
) -> str:
    """COMPLETED on low-entropy consensus or a high-confidence grounded answer."""
    if not claims:
        return "REQUIRES_CLARIFICATION"
    if (
        grounded is not None
        and getattr(grounded, "act_safe", False)
        and getattr(grounded, "answer", None)
    ):
        return "COMPLETED"
    if entropy_score > threshold:
        return "REQUIRES_CLARIFICATION"
    if all(c.confidence_score < required_credibility for c in claims):
        return "REQUIRES_CLARIFICATION"
    return "COMPLETED"


app = create_app()


if __name__ == "__main__":
    import uvicorn

    _settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=_settings.host,
        port=_settings.port,
        reload=_settings.debug,
    )
