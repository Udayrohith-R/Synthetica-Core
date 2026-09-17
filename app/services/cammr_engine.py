"""
Module 4: CAMMR Re-ranker & Information Entropy Engine.

Consensus-Aware Maximal Marginal Relevance over selected claims,
plus subgraph information entropy H(R_q) for fast-path / arbitration routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from uuid import uuid4

import numpy as np

from app.models.schemas import (
    AtomicClaim,
    Claim,
    ClaimRelation,
    ClaimRelationType,
    PipelinePath,
    RankedDocument,
    SearchResult,
)
from app.utils.entropy import (
    contradiction_entropy,
    normalized_entropy,
    softmax,
    subgraph_entropy,
    tokenize,
    token_distribution,
)

# Fixed publisher-diversity penalty applied when a domain is already selected.
DOMAIN_PENALTY: float = 0.15


@dataclass(frozen=True)
class CAMMRWeights:
    """Blend for consensus-aware MMR scoring."""

    relevance: float = 0.40
    consensus: float = 0.35
    diversity: float = 0.25

    def normalized(self) -> CAMMRWeights:
        total = self.relevance + self.consensus + self.diversity
        if total <= 0:
            raise ValueError("CAMMR weights must sum to a positive value")
        return CAMMRWeights(
            relevance=self.relevance / total,
            consensus=self.consensus / total,
            diversity=self.diversity / total,
        )


@dataclass(frozen=True)
class CAMMRResult:
    """Ranked docs plus entropy gate decision."""

    ranked: list[RankedDocument]
    selected_claims: list[Claim]
    subgraph_entropy: float
    path: PipelinePath


class CAMMREngine:
    """
    Consensus-Aware Maximal Marginal Relevance (CAMMR).

    Iteratively selects items that maximize:
      λ · (relevance · consensus) − (1−λ) · redundancy
    then computes H(R_q) over the selected claim subgraph.
    """

    def __init__(
        self,
        *,
        weights: CAMMRWeights | None = None,
        lam: float = 0.7,
        entropy_threshold: float = 0.55,
    ) -> None:
        self.weights = (weights or CAMMRWeights()).normalized()
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must be in [0, 1]")
        self.lam = lam
        self.entropy_threshold = entropy_threshold

    def compute_cammr_reranking(
        self,
        query_embedding: np.ndarray,
        candidate_claims: List[Dict[str, Any]],
        top_k: int = 10,
        lambda_param: float = 0.35,
        mu_param: float = 0.30,
        phi_param: float = 0.20,
    ) -> List[AtomicClaim]:
        """
        Instance entry-point for AtomicClaim CAMMR re-ranking.

        Delegates to the module-level pure-numpy implementation.
        """
        return compute_cammr_reranking(
            query_embedding,
            candidate_claims,
            top_k=top_k,
            lambda_param=lambda_param,
            mu_param=mu_param,
            phi_param=phi_param,
        )

    def rerank(
        self,
        query: str,
        documents: list[SearchResult],
        claims: list[Claim] | None = None,
        relations: list[ClaimRelation] | None = None,
        *,
        top_k: int | None = None,
        force_arbitration: bool = False,
        entropy_threshold: float | None = None,
    ) -> CAMMRResult:
        claims = claims or []
        relations = relations or []
        threshold = (
            self.entropy_threshold if entropy_threshold is None else entropy_threshold
        )

        if not documents:
            return CAMMRResult(
                ranked=[],
                selected_claims=[],
                subgraph_entropy=0.0,
                path=PipelinePath.FAST_PATH,
            )

        relevance = self._relevance_scores(query, documents)
        consensus = self._consensus_scores(documents, claims, relations)
        # Candidate utility before MMR diversity penalty
        utility = (
            self.weights.relevance * relevance + self.weights.consensus * consensus
        )

        k = top_k or len(documents)
        selected_idx = self._mmr_select(
            documents,
            utility,
            k=k,
        )

        ranked: list[RankedDocument] = []
        for rank_pos, i in enumerate(selected_idx, start=1):
            ranked.append(
                RankedDocument(
                    document=documents[i],
                    cammr_score=float(utility[i]),
                    entropy=None,
                    claim_coverage=None,
                    consensus_score=float(consensus[i]),
                    rank=rank_pos,
                )
            )

        selected_docs = {documents[i].id for i in selected_idx}
        selected_claims = [
            c for c in claims if c.document_id in selected_docs or c.document_id is None
        ]
        # Prefer claims tied to selected docs; if none matched, fall back to all
        if not selected_claims:
            selected_claims = claims

        h_rq = self.compute_subgraph_entropy(selected_claims, relations)
        for item in ranked:
            item.entropy = h_rq

        if force_arbitration or h_rq > threshold:
            path = PipelinePath.DEEP_ARBITRATION
        else:
            path = PipelinePath.FAST_PATH

        return CAMMRResult(
            ranked=ranked,
            selected_claims=selected_claims,
            subgraph_entropy=h_rq,
            path=path,
        )

    def compute_subgraph_entropy(
        self,
        claims: list[Claim],
        relations: list[ClaimRelation] | None = None,
    ) -> float:
        """H(R_q): mixture of claim content entropy and contradiction binary entropy."""
        relations = relations or []
        content_h = subgraph_entropy([c.text for c in claims])

        corr = sum(
            r.confidence
            for r in relations
            if r.relation == ClaimRelationType.CORROBORATES
        )
        contra = sum(
            r.confidence
            for r in relations
            if r.relation == ClaimRelationType.CONTRADICTS
        )
        rel_h = contradiction_entropy(corr, contra)

        # If no explicit relations, estimate disagreement from object diversity
        if corr == 0 and contra == 0 and claims:
            objects = [(c.object or c.text).lower() for c in claims]
            uniq = len(set(objects))
            if uniq <= 1:
                rel_h = 0.0
            else:
                counts = np.array(
                    [objects.count(o) for o in set(objects)], dtype=np.float64
                )
                probs = counts / counts.sum()
                rel_h = normalized_entropy(probs)

        return float(0.5 * content_h + 0.5 * rel_h)

    def _mmr_select(
        self,
        documents: list[SearchResult],
        utility: np.ndarray,
        *,
        k: int,
    ) -> list[int]:
        """Greedy Consensus-Aware Maximal Marginal Relevance selection."""
        n = len(documents)
        k = min(k, n)
        selected: list[int] = []
        remaining = set(range(n))

        # Precompute token sets for redundancy
        token_sets = [
            set(tokenize(f"{d.title or ''} {d.text or ''}")) for d in documents
        ]

        while remaining and len(selected) < k:
            best_i = None
            best_score = -np.inf
            for i in remaining:
                if not selected:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        self._jaccard(token_sets[i], token_sets[j]) for j in selected
                    )
                score = self.lam * float(utility[i]) - (1.0 - self.lam) * redundancy
                # Fold residual diversity weight
                score += self.weights.diversity * (1.0 - redundancy)
                if score > best_score:
                    best_score = score
                    best_i = i
            assert best_i is not None
            selected.append(best_i)
            remaining.remove(best_i)
        return selected

    def _relevance_scores(
        self,
        query: str,
        documents: list[SearchResult],
    ) -> np.ndarray:
        q_tokens = set(tokenize(query))
        scores = np.zeros(len(documents), dtype=np.float64)
        provider = np.array(
            [float(d.score) if d.score is not None else np.nan for d in documents],
            dtype=np.float64,
        )

        for i, doc in enumerate(documents):
            text = f"{doc.title or ''} {doc.text or ''}"
            d_tokens = set(tokenize(text))
            lexical = (
                len(q_tokens & d_tokens) / len(q_tokens) if q_tokens and d_tokens else 0.0
            )
            scores[i] = lexical

        if np.any(~np.isnan(provider)):
            filled = np.nan_to_num(provider, nan=0.0)
            pmin, pmax = float(np.min(filled)), float(np.max(filled))
            if pmax > pmin:
                filled = (filled - pmin) / (pmax - pmin)
            scores = 0.5 * scores + 0.5 * filled
        return scores

    def _consensus_scores(
        self,
        documents: list[SearchResult],
        claims: list[Claim],
        relations: list[ClaimRelation],
    ) -> np.ndarray:
        """
        Per-document consensus: corroboration mass minus contradiction mass
        among claims attached to that document, scaled to [0, 1].
        """
        n = len(documents)
        if not claims:
            return np.full(n, 0.5, dtype=np.float64)

        claims_by_doc: dict[str, list[Claim]] = {}
        for c in claims:
            if c.document_id:
                claims_by_doc.setdefault(c.document_id, []).append(c)

        claim_ids = {c.id for c in claims}
        corr_by_claim: dict = {c.id: 0.0 for c in claims}
        contra_by_claim: dict = {c.id: 0.0 for c in claims}
        for r in relations:
            if r.source_claim_id not in claim_ids or r.target_claim_id not in claim_ids:
                continue
            if r.relation == ClaimRelationType.CORROBORATES:
                corr_by_claim[r.source_claim_id] += r.confidence
                corr_by_claim[r.target_claim_id] += r.confidence
            elif r.relation == ClaimRelationType.CONTRADICTS:
                contra_by_claim[r.source_claim_id] += r.confidence
                contra_by_claim[r.target_claim_id] += r.confidence

        scores = np.zeros(n, dtype=np.float64)
        for i, doc in enumerate(documents):
            doc_claims = claims_by_doc.get(doc.id, [])
            if not doc_claims:
                # Soft fallback: token coverage against global claims
                haystack = f"{doc.title or ''} {doc.text or ''}".lower()
                hits = 0
                for claim in claims:
                    toks = tokenize(claim.text)
                    if toks and sum(1 for t in toks if t in haystack) / len(toks) >= 0.5:
                        hits += 1
                scores[i] = hits / len(claims) if claims else 0.5
                continue

            corr = sum(corr_by_claim[c.id] for c in doc_claims)
            contra = sum(contra_by_claim[c.id] for c in doc_claims)
            total = corr + contra
            if total <= 0:
                scores[i] = float(np.mean([c.confidence for c in doc_claims]))
            else:
                scores[i] = corr / total
        return scores

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Pure-numpy CAMMR over AtomicClaim candidates
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with zero-vector guard."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _as_embedding(value: Any, dim: int) -> np.ndarray:
    """Coerce an optional embedding to a fixed-dim float64 vector."""
    if value is None:
        return np.zeros(dim, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros(dim, dtype=np.float64)
    if arr.size != dim:
        # Pad / truncate so pairwise cosine stays well-defined.
        out = np.zeros(dim, dtype=np.float64)
        n = min(dim, arr.size)
        out[:n] = arr[:n]
        return out
    return arr


def _dict_to_atomic_claim(raw: Dict[str, Any]) -> AtomicClaim:
    """Build an ``AtomicClaim`` from a loose candidate dict."""
    statement = str(
        raw.get("statement") or raw.get("text") or raw.get("claim") or ""
    ).strip()
    if not statement:
        raise ValueError("candidate claim missing statement/text")

    confidence = raw.get("confidence_score", raw.get("confidence", 0.5))
    try:
        confidence_f = float(confidence)
    except (TypeError, ValueError):
        confidence_f = 0.5
    confidence_f = max(0.0, min(1.0, confidence_f))

    consensus = raw.get("consensus_score", 0.0)
    try:
        consensus_f = float(consensus or 0.0)
    except (TypeError, ValueError):
        consensus_f = 0.0
    consensus_f = max(0.0, min(1.0, consensus_f))

    entities = raw.get("entities") or []
    if not isinstance(entities, list):
        entities = []
    entities = [str(e).strip() for e in entities if str(e).strip()]

    domain = str(raw.get("source_domain") or raw.get("domain") or "unknown").strip()
    citation = str(
        raw.get("citation_anchor")
        or raw.get("url")
        or "https://example.com/unknown"
    ).strip()

    claim_id = raw.get("claim_id") or str(uuid4())
    embedding = raw.get("embedding")
    if embedding is not None and not isinstance(embedding, list):
        embedding = list(np.asarray(embedding, dtype=np.float64).reshape(-1))

    return AtomicClaim(
        claim_id=claim_id,
        statement=statement,
        confidence_score=confidence_f,
        entities=entities,
        source_domain=domain or "unknown",
        citation_anchor=citation,
        embedding=embedding,
        consensus_score=consensus_f,
    )


def compute_cammr_reranking(
    query_embedding: np.ndarray,
    candidate_claims: List[Dict[str, Any]],
    top_k: int = 10,
    lambda_param: float = 0.35,
    mu_param: float = 0.30,
    phi_param: float = 0.20,
) -> List[AtomicClaim]:
    """
    Consensus-Aware Maximal Marginal Relevance (CAMMR) re-ranking.

    Greedily builds a size-``top_k`` subset ``S`` by repeatedly selecting the
    candidate that maximizes::

        CAMMR(c | S) =
            λ · sim(q, c)          # query relevance
          + μ · consensus(c)      # cross-domain corroboration
          − φ · max_{s∈S} sim(c, s)   # redundancy vs already-selected
          − δ · 𝟙[domain(c) ∈ Dom(S)] # publisher-domain penalty (δ = 0.15)

    where ``λ + μ + φ + δ = 1`` under the default hyperparameters
    (0.35 + 0.30 + 0.20 + 0.15).

    Parameters
    ----------
    query_embedding:
        Dense query vector used for cosine relevance.
    candidate_claims:
        Loose claim dicts (``statement``, ``embedding``, ``source_domain``,
        ``consensus_score``, …) convertible to ``AtomicClaim``.
    top_k:
        Number of claims to return.
    lambda_param:
        Weight on query–claim cosine relevance.
    mu_param:
        Weight elevating high ``consensus_score`` (cross-domain corroboration).
    phi_param:
        Weight penalizing embedding similarity to the selected set.

    Returns
    -------
    Ranked list of up to ``top_k`` ``AtomicClaim`` objects (highest CAMMR first).
    """
    if not candidate_claims:
        return []

    q = np.asarray(query_embedding, dtype=np.float64).reshape(-1)
    if q.size == 0:
        raise ValueError("query_embedding must be a non-empty vector")

    dim = int(q.size)
    claims: List[AtomicClaim] = [_dict_to_atomic_claim(raw) for raw in candidate_claims]
    embeddings = [_as_embedding(c.embedding, dim) for c in claims]

    # Pre-compute query relevance once: Rel(c) = cos(q, e_c) ∈ [-1, 1] → [0, 1]
    relevance = np.array(
        [(_cosine_similarity(q, emb) + 1.0) * 0.5 for emb in embeddings],
        dtype=np.float64,
    )
    # Elevate claims with high cross-domain corroboration mass.
    consensus = np.array(
        [float(c.consensus_score) for c in claims],
        dtype=np.float64,
    )
    domains = [
        (c.source_domain or "").strip().lower() or "unknown" for c in claims
    ]

    k = max(0, min(int(top_k), len(claims)))
    if k == 0:
        return []

    selected: List[int] = []
    remaining = set(range(len(claims)))
    selected_domains: set[str] = set()

    # ------------------------------------------------------------------
    # Greedy CAMMR scoring loop
    # ------------------------------------------------------------------
    # At iteration t we choose
    #   c* = argmax_{c ∉ S} [ λ Rel(c) + μ Cons(c) − φ Red(c;S) − δ Dom(c;S) ]
    # where
    #   Rel(c)      = (cos(q, e_c) + 1) / 2
    #   Cons(c)     = consensus_score(c) ∈ [0, 1]
    #   Red(c; S)   = max_{s ∈ S} cos(e_c, e_s)          (0 when S = ∅)
    #   Dom(c; S)   = 1 if source_domain(c) already in S, else 0
    #   δ           = DOMAIN_PENALTY = 0.15
    # ------------------------------------------------------------------
    while remaining and len(selected) < k:
        best_i: int | None = None
        best_score = -np.inf

        for i in remaining:
            # Relevance term — pull toward the query embedding.
            rel_term = lambda_param * float(relevance[i])

            # Consensus term — reward cross-domain corroboration mass.
            cons_term = mu_param * float(consensus[i])

            # Redundancy term — penalize near-duplicates of already-selected claims.
            if not selected:
                redundancy = 0.0
            else:
                redundancy = max(
                    _cosine_similarity(embeddings[i], embeddings[j])
                    for j in selected
                )
                # Cosine may be negative; clamp so the penalty stays non-negative.
                redundancy = max(0.0, float(redundancy))
            red_term = phi_param * redundancy

            # Domain penalty — discourage stacking multiple claims from one publisher.
            dom_term = (
                DOMAIN_PENALTY if domains[i] in selected_domains else 0.0
            )

            score = rel_term + cons_term - red_term - dom_term

            if score > best_score:
                best_score = score
                best_i = i

        assert best_i is not None
        selected.append(best_i)
        remaining.remove(best_i)
        selected_domains.add(domains[best_i])

    return [claims[i] for i in selected]
