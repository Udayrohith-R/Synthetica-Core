"""Information entropy math helpers for CAMMR and subgraph H(R_q)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import List

import numpy as np

from app.models.schemas import AtomicClaim


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer."""
    return [
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if token
    ]


def token_distribution(text: str) -> dict[str, float]:
    """Return a probability distribution over tokens in `text`."""
    tokens = tokenize(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = float(sum(counts.values()))
    return {token: count / total for token, count in counts.items()}


def shannon_entropy(
    probabilities: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    base: float = 2.0,
) -> float:
    """Shannon entropy H(X) = -Σ p(x) log_b p(x)."""
    if isinstance(probabilities, Mapping):
        probs = np.asarray(list(probabilities.values()), dtype=np.float64)
    else:
        probs = np.asarray(list(probabilities), dtype=np.float64)

    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0

    log_probs = np.log(probs) / np.log(base)
    return float(-np.sum(probs * log_probs))


def normalized_entropy(
    probabilities: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    base: float = 2.0,
) -> float:
    """Entropy normalized by log_b(|support|) into [0, 1]."""
    if isinstance(probabilities, Mapping):
        n = sum(1 for p in probabilities.values() if p > 0)
    else:
        arr = np.asarray(list(probabilities), dtype=np.float64)
        n = int(np.count_nonzero(arr > 0))

    if n <= 1:
        return 0.0

    h = shannon_entropy(probabilities, base=base)
    return float(h / (np.log(n) / np.log(base)))


def softmax(values: Iterable[float], *, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return x
    z = (x - np.max(x)) / temperature
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def kl_divergence(
    p: Mapping[str, float] | Sequence[float],
    q: Mapping[str, float] | Sequence[float],
    *,
    base: float = 2.0,
    epsilon: float = 1e-12,
) -> float:
    """KL(P || Q) with Laplace smoothing via epsilon."""
    if isinstance(p, Mapping) and isinstance(q, Mapping):
        keys = set(p) | set(q)
        p_arr = np.asarray([p.get(k, 0.0) for k in keys], dtype=np.float64)
        q_arr = np.asarray([q.get(k, 0.0) for k in keys], dtype=np.float64)
    else:
        p_arr = np.asarray(list(p), dtype=np.float64)
        q_arr = np.asarray(list(q), dtype=np.float64)

    p_arr = np.clip(p_arr, epsilon, None)
    q_arr = np.clip(q_arr, epsilon, None)
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    return float(np.sum(p_arr * (np.log(p_arr / q_arr) / np.log(base))))


def subgraph_entropy(
    claim_texts: Sequence[str],
    *,
    relation_weights: Sequence[float] | None = None,
) -> float:
    """
    Subgraph information entropy H(R_q) over the selected claim set.

    Models the result subgraph as a mixture of claim token distributions,
    optionally reweighted by corroboration/contradiction edge strengths.
    Returns normalized entropy in [0, 1]:
      - low  → clean consensus
      - high → factual contradiction / disagreement
    """
    if not claim_texts:
        return 0.0

    weights = (
        np.asarray(list(relation_weights), dtype=np.float64)
        if relation_weights is not None
        else np.ones(len(claim_texts), dtype=np.float64)
    )
    if weights.size != len(claim_texts):
        raise ValueError("relation_weights must match claim_texts length")
    weights = np.clip(weights, 1e-12, None)
    weights = weights / weights.sum()

    mixture: Counter[str] = Counter()
    for text, w in zip(claim_texts, weights, strict=True):
        dist = token_distribution(text)
        for token, p in dist.items():
            mixture[token] += float(w) * p

    total = sum(mixture.values())
    if total <= 0:
        return 0.0
    probs = {t: c / total for t, c in mixture.items()}
    return normalized_entropy(probs)


def contradiction_entropy(
    corroboration_mass: float,
    contradiction_mass: float,
) -> float:
    """
    Binary entropy over corroborate vs contradict edge mass in the subgraph.

    High when the graph is split between supporting and opposing claims.
    """
    total = corroboration_mass + contradiction_mass
    if total <= 0:
        return 0.0
    p = corroboration_mass / total
    q = 1.0 - p
    probs = np.array([p, q], dtype=np.float64)
    return normalized_entropy(probs)


def calculate_subgraph_entropy(
    claims: List[AtomicClaim],
    contradiction_count: int,
) -> float:
    """
    Shannon Information Entropy H(R_q) for the selected claim subgraph.

    Combines two uncertainty signals into a single score in ``[0, 1]``:

    1. **Source entropy** — Shannon entropy of the empirical distribution over
       publishing domains (``source_domain``). High when evidence is spread
       across many publishers; low when a single domain dominates.
    2. **Contradiction-ratio entropy** — binary Shannon entropy of the
       contradiction ratio ``r = contradiction_count / n``. Peaks when the
       subgraph is evenly split between agreement and conflict.

    Formally::

        H_src   = H(p_domain) / log2(|support|)     # normalized to [0, 1]
        r       = clip(contradiction_count / n, 0, 1)
        H_contra = -r log2(r) - (1-r) log2(1-r)     # already in [0, 1]
        H(R_q)  = 0.5 · H_src + 0.5 · H_contra

    Returns 0.0 for an empty claim set (perfect certainty / no evidence).
    """
    if not claims:
        return 0.0

    n = len(claims)
    contradiction_count = max(0, int(contradiction_count))

    # --- Source / candidate-publisher distribution ---
    domain_counts = Counter(
        (c.source_domain or "").strip().lower() or "unknown" for c in claims
    )
    source_probs = np.asarray(
        [count / n for count in domain_counts.values()],
        dtype=np.float64,
    )
    h_src = normalized_entropy(source_probs, base=2.0)

    # --- Contradiction ratio as a Bernoulli uncertainty ---
    # r ∈ [0, 1]: fraction of the claim set implicated in contradictions.
    r = min(1.0, contradiction_count / float(n))
    if r <= 0.0 or r >= 1.0:
        h_contra = 0.0
    else:
        h_contra = float(-r * np.log2(r) - (1.0 - r) * np.log2(1.0 - r))

    return float(0.5 * h_src + 0.5 * h_contra)
