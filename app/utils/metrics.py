"""Numeric + financial metric normalization for corroboration / conflicts."""

from __future__ import annotations

import re
from dataclasses import dataclass

# $10.32 billion | 10.32B | US$10.32 billion | 10,320 million
_MONEY_RE = re.compile(
    r"""
    (?:US\$|\$)?\s*
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<unit>billion|million|bn|mm|m|b)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PERCENT_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*%|(?P<num2>\d+(?:\.\d+)?)\s*percent",
    re.IGNORECASE,
)

# Canonical metric keys → lexical triggers (order matters: more specific first)
_METRIC_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    (
        "data_center_revenue",
        (
            "data center revenue",
            "datacenter revenue",
            "data-center revenue",
            "dc revenue",
        ),
    ),
    (
        "data_center_compute_revenue",
        (
            "data center compute revenue",
            "compute revenue",
        ),
    ),
    (
        "total_revenue",
        (
            "total revenue",
            "company revenue",
            "overall revenue",
            "net revenue",
            "revenue was",
            "in revenue",
            "staggering",
        ),
    ),
    (
        "operating_income",
        ("operating income", "operating profit"),
    ),
    (
        "net_income",
        ("net income", "pure profit", "net profit"),
    ),
    (
        "guidance",
        ("guidance", "outlook", "expects revenue"),
    ),
    (
        "yoy_growth",
        ("year over year", "year-over-year", "yoy", "yo y"),
    ),
    (
        "qoq_growth",
        ("quarter over quarter", "quarter-over-quarter", "qoq"),
    ),
]


@dataclass(frozen=True, slots=True)
class NormalizedMoney:
    """Absolute USD amount (float dollars) + original span."""

    amount_usd: float
    span: str


def parse_money_values(text: str) -> list[NormalizedMoney]:
    """Extract money mentions as absolute USD floats."""
    out: list[NormalizedMoney] = []
    if not text:
        return out
    for match in _MONEY_RE.finditer(text):
        raw_num = match.group("num").replace(",", "")
        try:
            value = float(raw_num)
        except ValueError:
            continue
        unit = (match.group("unit") or "").lower()
        # Bare integers like years (2024) — skip if no $ and no unit and looks like a year
        span = match.group(0).strip()
        if not unit and "$" not in span and "US$" not in span.upper():
            if 1900 <= value <= 2100 and value.is_integer():
                continue
            # Unqualified large plain numbers without unit are ambiguous — skip
            if value < 100:
                continue
        multiplier = 1.0
        if unit in {"billion", "bn", "b"}:
            multiplier = 1_000_000_000.0
        elif unit in {"million", "mm", "m"}:
            multiplier = 1_000_000.0
        amount = value * multiplier
        # Ignore tiny non-currency leftovers
        if amount <= 0:
            continue
        out.append(NormalizedMoney(amount_usd=amount, span=span))
    return out


def money_signature(amount_usd: float, *, tol: float = 0.02) -> str:
    """
    Bucket money into a stable signature for corroboration.

    ``tol`` is relative tolerance (default 2%) so $10.32B ≈ $10.3B.
    """
    if amount_usd <= 0:
        return "0"
    # Round to 3 significant figures in scientific-ish buckets
    # e.g. 1.032e10 → "1.03e10"
    from math import log10, floor

    exp = floor(log10(amount_usd))
    mant = amount_usd / (10**exp)
    # Snap mantissa onto a coarse grid (~2% buckets)
    step = max(tol, 0.01)
    snapped = round(mant / step) * step
    if snapped >= 10:
        snapped /= 10
        exp += 1
    return f"{snapped:.3f}e{exp}"


def detect_metric_key(text: str) -> str | None:
    """Map free text onto a coarse financial metric ontology key."""
    lowered = (text or "").lower()
    for key, triggers in _METRIC_PATTERNS:
        if any(t in lowered for t in triggers):
            return key
    # Fallback: any revenue mention
    if "revenue" in lowered:
        return "revenue_unspecified"
    return None


def query_metric_keys(query: str) -> set[str]:
    """Metric keys implied by the user query."""
    keys: set[str] = set()
    detected = detect_metric_key(query)
    if detected:
        keys.add(detected)
    q = query.lower()
    if "data center" in q or "datacenter" in q:
        keys.add("data_center_revenue")
    if "revenue" in q and not keys:
        keys.add("revenue_unspecified")
    return keys


def metric_compatible(query_keys: set[str], claim_key: str | None) -> bool:
    """True if claim metric can satisfy the query's metric intent."""
    if not query_keys:
        return True
    if claim_key is None:
        return False
    if claim_key in query_keys:
        return True
    # data_center_revenue query should not accept total_revenue as a match
    if "data_center_revenue" in query_keys:
        return claim_key in {
            "data_center_revenue",
            "data_center_compute_revenue",
        }
    if "revenue_unspecified" in query_keys:
        return claim_key.startswith("data_center") or claim_key in {
            "total_revenue",
            "revenue_unspecified",
        }
    return False


def parse_percents(text: str) -> list[float]:
    vals: list[float] = []
    for match in _PERCENT_RE.finditer(text or ""):
        raw = match.group("num") or match.group("num2")
        if raw is None:
            continue
        try:
            vals.append(float(raw))
        except ValueError:
            continue
    return vals
