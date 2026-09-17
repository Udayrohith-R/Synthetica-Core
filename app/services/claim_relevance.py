"""
Query-conditioned claim filtering + numeric corroboration boosts.

Keeps only claims that can answer the research question (entity + metric + time),
and lifts consensus when independent sources assert the same normalized number.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass

from app.models.schemas import AtomicClaim
from app.utils.entropy import tokenize
from app.utils.metrics import (
    detect_metric_key,
    metric_compatible,
    money_signature,
    parse_money_values,
    query_metric_keys,
)

logger = logging.getLogger(__name__)

# Common English stopwords — not treated as focus entities
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "for",
        "to",
        "from",
        "with",
        "by",
        "at",
        "as",
        "is",
        "was",
        "were",
        "be",
        "been",
        "are",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "does",
        "did",
        "do",
        "its",
        "their",
        "our",
        "your",
        "this",
        "that",
        "these",
        "those",
        "into",
        "over",
        "under",
        "about",
        "after",
        "before",
        "between",
        "during",
        "q1",
        "q2",
        "q3",
        "q4",
        "fy",
        "yo",
        "yoy",
        "qoq",
        "vs",
        "versus",
        "data",
        "center",
        "server",
        "revenue",
        "sales",
        "report",
        "earnings",
        "billion",
        "million",
    }
)

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_QUARTER_RE = re.compile(
    r"\b(?:q([1-4])|([1-4])q)\s*(?:fy)?\s*(20\d{2})?\b|"
    r"\bfy\s*(20\d{2})\s*q([1-4])\b|"
    r"\b(?:fiscal\s+)?(?:second|2nd|first|1st|third|3rd|fourth|4th)\s+quarter\b",
    re.IGNORECASE,
)

# Well-known issuer aliases for financial demos
_ENTITY_ALIASES: dict[str, set[str]] = {
    "nvidia": {"nvidia", "nvda", "nvidia corporation", "nvidia's"},
    "amd": {"amd", "advanced micro devices"},
    "intel": {"intel", "intc"},
    "microsoft": {"microsoft", "msft"},
    "apple": {"apple", "aapl"},
    "google": {"google", "alphabet", "goog", "googl"},
    "amazon": {"amazon", "amzn"},
    "meta": {"meta", "facebook", "fb"},
}


@dataclass(frozen=True, slots=True)
class QueryFocus:
    """Parsed intent from a research query."""

    raw: str
    tokens: set[str]
    entities: set[str]  # canonical entity keys
    entity_surface: set[str]  # surface forms to match in claims
    years: set[int]
    quarters: set[int]
    metric_keys: set[str]
    wants_money: bool


@dataclass
class FilterResult:
    """Kept vs rejected claims after query conditioning."""

    kept: list[AtomicClaim]
    rejected: list[tuple[AtomicClaim, str]]  # claim, reason


def parse_query_focus(query: str) -> QueryFocus:
    """Extract entities, years, quarters, and metric intent from ``query``."""
    tokens = set(tokenize(query))
    lowered = query.lower()

    entities: set[str] = set()
    surface: set[str] = set()
    for canon, aliases in _ENTITY_ALIASES.items():
        if any(a in lowered for a in aliases):
            entities.add(canon)
            surface |= aliases

    # Capitalized proper nouns / tickers as soft entities
    for match in _TICKER_RE.finditer(query):
        tok = match.group(0)
        if tok.lower() in _STOP or len(tok) < 2:
            continue
        # Skip pure years
        if tok.isdigit():
            continue
        surface.add(tok.lower())

    # Multi-word company-ish tokens from query (non-stop content words)
    for tok in tokens:
        if tok in _STOP or tok.isdigit() or len(tok) < 3:
            continue
        surface.add(tok)

    years = {int(y) for y in _YEAR_RE.findall(query)}
    quarters: set[int] = set()
    q = lowered
    for m in re.finditer(r"\bq([1-4])\b", q):
        quarters.add(int(m.group(1)))
    if "second quarter" in q or "2nd quarter" in q:
        quarters.add(2)
    if "first quarter" in q or "1st quarter" in q:
        quarters.add(1)
    if "third quarter" in q or "3rd quarter" in q:
        quarters.add(3)
    if "fourth quarter" in q or "4th quarter" in q:
        quarters.add(4)

    metric_keys = query_metric_keys(query)
    wants_money = bool(
        parse_money_values(query)
        or any(
            w in lowered
            for w in ("revenue", "sales", "profit", "income", "guidance", "$", "billion")
        )
    )

    return QueryFocus(
        raw=query,
        tokens=tokens,
        entities=entities,
        entity_surface=surface,
        years=years,
        quarters=quarters,
        metric_keys=metric_keys,
        wants_money=wants_money,
    )


def score_claim_relevance(claim: AtomicClaim, focus: QueryFocus) -> float:
    """
    Relevance in [0, 1] of a claim to the query focus.

    Hard components (entity / metric) dominate; soft lexical overlap fills gaps.
    """
    statement = claim.statement or ""
    blob = f"{statement} {' '.join(claim.entities)}".lower()
    score = 0.0

    # --- Entity ---
    entity_hit = False
    if focus.entities:
        for canon in focus.entities:
            aliases = _ENTITY_ALIASES.get(canon, {canon})
            if any(a in blob for a in aliases):
                entity_hit = True
                break
        score += 0.45 if entity_hit else 0.0
    else:
        # Soft: any focus surface token in claim
        surf_hits = sum(1 for s in focus.entity_surface if s in blob)
        entity_hit = surf_hits > 0
        score += min(0.35, 0.1 * surf_hits)

    # --- Metric ---
    claim_metric = detect_metric_key(statement)
    if focus.metric_keys:
        if metric_compatible(focus.metric_keys, claim_metric):
            score += 0.30
        elif claim_metric is None and focus.wants_money and parse_money_values(statement):
            score += 0.10
        else:
            score += 0.0
    else:
        score += 0.10

    # --- Temporal ---
    claim_years = {int(y) for y in _YEAR_RE.findall(statement)}
    if focus.years:
        if claim_years & focus.years:
            score += 0.15
        elif not claim_years:
            score += 0.05  # undated but maybe OK
    else:
        score += 0.05

    if focus.quarters:
        claim_q = set()
        for m in re.finditer(r"\bq([1-4])\b", statement.lower()):
            claim_q.add(int(m.group(1)))
        if "second quarter" in statement.lower():
            claim_q.add(2)
        if claim_q & focus.quarters:
            score += 0.10

    # --- Lexical overlap soft boost ---
    c_tokens = set(tokenize(statement))
    if focus.tokens and c_tokens:
        overlap = len(focus.tokens & c_tokens) / max(len(focus.tokens), 1)
        score += 0.10 * min(1.0, overlap * 2)

    return float(max(0.0, min(1.0, score)))


def filter_claims_for_query(
    claims: list[AtomicClaim],
    query: str,
    *,
    min_score: float = 0.55,
    require_primary_entity: bool = True,
) -> FilterResult:
    """
    Drop off-topic claims (e.g. Intel/AMD quotes on an NVIDIA revenue query).
    """
    focus = parse_query_focus(query)
    kept: list[AtomicClaim] = []
    rejected: list[tuple[AtomicClaim, str]] = []

    for claim in claims:
        score = score_claim_relevance(claim, focus)
        blob = f"{claim.statement} {' '.join(claim.entities)}".lower()

        if require_primary_entity and focus.entities:
            entity_ok = False
            for canon in focus.entities:
                aliases = _ENTITY_ALIASES.get(canon, {canon})
                if any(a in blob for a in aliases):
                    entity_ok = True
                    break
            if not entity_ok:
                rejected.append(
                    (
                        claim,
                        f"missing primary entity ({', '.join(sorted(focus.entities))})",
                    )
                )
                continue

        claim_metric = detect_metric_key(claim.statement)
        if focus.metric_keys and not metric_compatible(focus.metric_keys, claim_metric):
            # Allow through only if score still high and no conflicting metric
            if claim_metric and claim_metric != "revenue_unspecified":
                rejected.append(
                    (
                        claim,
                        f"metric mismatch (query wants {sorted(focus.metric_keys)}, "
                        f"claim is {claim_metric})",
                    )
                )
                continue

        if score < min_score:
            rejected.append((claim, f"query relevance {score:.2f} < {min_score:.2f}"))
            continue

        kept.append(claim)

    logger.info(
        "claim_filter kept=%d rejected=%d query_entities=%s metrics=%s",
        len(kept),
        len(rejected),
        sorted(focus.entities),
        sorted(focus.metric_keys),
    )
    return FilterResult(kept=kept, rejected=rejected)


def boost_numeric_consensus(claims: list[AtomicClaim]) -> list[AtomicClaim]:
    """
    Raise ``consensus_score`` when multiple domains assert the same money signature
    under a compatible metric.
    """
    if len(claims) < 2:
        return claims

    # Group by (metric, money_signature)
    groups: dict[tuple[str, str], list[AtomicClaim]] = defaultdict(list)
    for claim in claims:
        metric = detect_metric_key(claim.statement) or "unknown"
        monies = parse_money_values(claim.statement)
        if not monies:
            continue
        # Use the largest money mention as the primary asserted figure
        primary = max(monies, key=lambda m: m.amount_usd)
        sig = money_signature(primary.amount_usd)
        groups[(metric, sig)].append(claim)

    domain_counts: dict[tuple[str, str], set[str]] = {}
    for key, members in groups.items():
        domain_counts[key] = {m.source_domain.lower() for m in members}

    for claim in claims:
        metric = detect_metric_key(claim.statement) or "unknown"
        monies = parse_money_values(claim.statement)
        if not monies:
            continue
        primary = max(monies, key=lambda m: m.amount_usd)
        key = (metric, money_signature(primary.amount_usd))
        n_domains = len(domain_counts.get(key, set()))
        if n_domains >= 2:
            # Strong cross-publisher numeric corroboration
            boost = min(0.45, 0.15 * (n_domains - 1) + 0.25)
            claim.consensus_score = float(
                max(claim.consensus_score, min(1.0, claim.confidence_score * 0.4 + boost))
            )
        elif n_domains == 1 and len(groups.get(key, [])) >= 2:
            claim.consensus_score = float(
                max(claim.consensus_score, min(1.0, claim.consensus_score + 0.1))
            )

    return claims


def select_query_window(text: str, query: str, *, max_chars: int = 4000) -> str:
    """
    Keep the most query-relevant window of a long document for faster extraction.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    focus = parse_query_focus(query)
    needles = sorted(focus.entity_surface | focus.tokens, key=len, reverse=True)
    lowered = text.lower()
    best_idx = 0
    best_hits = -1
    # Slide windows
    step = max(500, max_chars // 4)
    for start in range(0, max(1, len(text) - max_chars + 1), step):
        window = lowered[start : start + max_chars]
        hits = sum(1 for n in needles if n and n in window)
        if hits > best_hits:
            best_hits = hits
            best_idx = start
    return text[best_idx : best_idx + max_chars]
