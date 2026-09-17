"""
Module 1: Agent State-Space Retrieval (SSM + BM25).

Accepts raw AST, stack traces, and non-human agent context state tensors.
Combines fast lexical BM25 with Matryoshka vector projection.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import numpy as np

from app.config import Settings, get_settings
from app.models.schemas import AgentStateTensor, DocumentChunk, SearchResult
from app.services.exa_service import ExaService
from app.utils.bm25 import BM25Index
from app.utils.entropy import tokenize

logger = logging.getLogger(__name__)


def matryoshka_project(
    vector: list[float] | np.ndarray,
    dims: list[int],
) -> dict[int, np.ndarray]:
    """
    Nested truncation projection for Matryoshka embeddings.

    Returns {dim: unit-normalized prefix} for each requested dimension.
    """
    vec = np.asarray(vector, dtype=np.float64).ravel()
    if vec.size == 0:
        return {d: np.zeros(d, dtype=np.float64) for d in dims}

    out: dict[int, np.ndarray] = {}
    for d in dims:
        if d <= 0:
            continue
        prefix = vec[:d] if vec.size >= d else np.pad(vec, (0, d - vec.size))
        norm = np.linalg.norm(prefix)
        out[d] = prefix / norm if norm > 0 else prefix
    return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class StateSpaceRetriever:
    """
    Hybrid lexical (BM25) + Matryoshka vector retriever for agent swarms.

    External web/corpus hits come from Exa; local corpus + agent context
    are scored via the in-process BM25 index. A Rust SIMD backend can replace
    BM25Index without changing this facade.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        exa: ExaService | None = None,
        bm25: BM25Index | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._exa = exa or ExaService(self._settings)
        self._bm25 = bm25 or BM25Index()
        self._corpus: dict[str, SearchResult] = {}

    def index_documents(self, documents: list[SearchResult]) -> int:
        """Add documents to the local lexical index."""
        for doc in documents:
            text = f"{doc.title or ''}\n{doc.text or ''}".strip()
            self._bm25.add(doc.id, text)
            self._corpus[doc.id] = doc
        return len(documents)

    def build_query_from_agent_state(self, state: AgentStateTensor) -> str:
        """Fold AST / stack / query into a single retrieval string."""
        parts = [state.query]
        if state.stack_trace:
            parts.append(f"STACK:\n{state.stack_trace}")
        if state.ast_dump:
            # Keep AST compact — first 4k chars for lexical signal
            parts.append(f"AST:\n{state.ast_dump[:4000]}")
        if state.agent_id:
            parts.append(f"agent:{state.agent_id}")
        return "\n\n".join(parts)

    async def retrieve(
        self,
        *,
        query: str,
        agent_state: AgentStateTensor | None = None,
        num_results: int = 10,
        use_exa: bool = True,
        alpha: float = 0.55,
    ) -> list[SearchResult]:
        """
        Hybrid retrieval over agent context + optional Exa neural search.

        alpha blends BM25 vs Matryoshka/vector channel: score = α·bm25 + (1-α)·vec
        """
        effective_query = (
            self.build_query_from_agent_state(agent_state)
            if agent_state is not None
            else query
        )

        lexical_hits = self._bm25.search(effective_query, top_k=num_results * 2)
        lexical_map = {doc_id: score for doc_id, score in lexical_hits}

        remote: list[SearchResult] = []
        if use_exa and self._settings.exa_api_key:
            try:
                remote = await self._exa.search(
                    effective_query,
                    num_results=num_results,
                    include_text=True,
                )
            except Exception:
                logger.exception("Exa retrieval failed; continuing with local index")

        # Merge local + remote corpora for scoring
        candidates: dict[str, SearchResult] = {}
        for doc_id, _ in lexical_hits:
            if doc_id in self._corpus:
                candidates[doc_id] = self._corpus[doc_id]
        for doc in remote:
            candidates[doc.id] = doc
            if doc.id not in lexical_map:
                # Score remote docs against local BM25 if indexed; else use provider score
                text = f"{doc.title or ''}\n{doc.text or ''}"
                tmp = BM25Index()
                tmp.add(doc.id, text)
                # Fallback: token overlap proxy when doc isn't in the shared index
                lexical_map[doc.id] = float(doc.score or self._overlap_score(effective_query, text))

        query_vec = self._query_vector(agent_state, effective_query)
        dims = self._settings.matryoshka_dim_list
        q_proj = matryoshka_project(query_vec, dims)
        # Use largest available Matryoshka dim for scoring
        primary_dim = max(dims) if dims else len(query_vec) or 64
        q_primary = q_proj.get(primary_dim, np.zeros(primary_dim))

        bm25_values = np.array(
            [lexical_map.get(doc_id, 0.0) for doc_id in candidates],
            dtype=np.float64,
        )
        bm25_norm = self._minmax(bm25_values)

        scored: list[SearchResult] = []
        for idx, (doc_id, doc) in enumerate(candidates.items()):
            doc_vec = self._document_vector(doc, target_dim=primary_dim)
            d_proj = matryoshka_project(doc_vec, [primary_dim])[primary_dim]
            vec_score = cosine_similarity(q_primary, d_proj)
            hybrid = alpha * float(bm25_norm[idx]) + (1.0 - alpha) * vec_score

            enriched = doc.model_copy(
                update={
                    "bm25_score": float(lexical_map.get(doc_id, 0.0)),
                    "vector_score": float(vec_score),
                    "score": float(hybrid),
                    "chunks": doc.chunks or self._chunk_document(doc),
                }
            )
            scored.append(enriched)

        scored.sort(key=lambda d: d.score or 0.0, reverse=True)
        return scored[:num_results]

    def _query_vector(
        self,
        agent_state: AgentStateTensor | None,
        query: str,
    ) -> np.ndarray:
        if agent_state and agent_state.state_tensor:
            return np.asarray(agent_state.state_tensor, dtype=np.float64)
        # Deterministic hashed bag-of-tokens embedding (no external model required)
        return self._hash_embed(query, dim=max(self._settings.matryoshka_dim_list or [256]))

    def _document_vector(self, doc: SearchResult, *, target_dim: int) -> np.ndarray:
        meta_vec = doc.metadata.get("embedding") if doc.metadata else None
        if isinstance(meta_vec, list) and meta_vec:
            return np.asarray(meta_vec, dtype=np.float64)
        return self._hash_embed(f"{doc.title or ''} {doc.text or ''}", dim=target_dim)

    @staticmethod
    def _hash_embed(text: str, *, dim: int) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float64)
        for token in tokenize(text):
            idx = hash(token) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def _overlap_score(query: str, text: str) -> float:
        q = set(tokenize(query))
        t = set(tokenize(text))
        if not q or not t:
            return 0.0
        return len(q & t) / len(q)

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        vmin, vmax = float(values.min()), float(values.max())
        if vmax <= vmin:
            return np.zeros_like(values) if vmax == 0 else np.ones_like(values)
        return (values - vmin) / (vmax - vmin)

    @staticmethod
    def _chunk_document(doc: SearchResult, *, chunk_size: int = 1200) -> list[DocumentChunk]:
        text = doc.text or ""
        if not text:
            return []
        chunks: list[DocumentChunk] = []
        for i in range(0, len(text), chunk_size):
            piece = text[i : i + chunk_size]
            chunks.append(
                DocumentChunk(
                    id=uuid4(),
                    document_id=doc.id,
                    ordinal=len(chunks),
                    text=piece,
                    start_offset=i,
                    end_offset=i + len(piece),
                )
            )
        return chunks

    def describe(self) -> dict[str, Any]:
        return {
            "module": "ssm_retrieval",
            "local_docs": len(self._corpus),
            "bm25_docs": len(self._bm25),
            "matryoshka_dims": self._settings.matryoshka_dim_list,
        }
