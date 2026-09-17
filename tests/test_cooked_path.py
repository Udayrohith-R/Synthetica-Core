"""Unit tests for query filter, numeric normalization, and conflict-aware answers."""

from __future__ import annotations

from uuid import uuid4

from app.models.schemas import AtomicClaim
from app.services.claim_relevance import (
    boost_numeric_consensus,
    filter_claims_for_query,
    parse_query_focus,
)
from app.services.conflict_resolver import build_conflict_aware_answer
from app.utils.metrics import detect_metric_key, money_signature, parse_money_values


def _claim(statement: str, *, domain: str, entities: list[str], conf: float = 0.9) -> AtomicClaim:
    return AtomicClaim(
        claim_id=uuid4(),
        statement=statement,
        confidence_score=conf,
        entities=entities,
        source_domain=domain,
        citation_anchor=f"https://{domain}/x",
    )


def test_query_focus_nvidia_dc_revenue() -> None:
    focus = parse_query_focus("NVIDIA Q2 2024 data center server revenue")
    assert "nvidia" in focus.entities
    assert "data_center_revenue" in focus.metric_keys


def test_filter_drops_offtopic_intel() -> None:
    q = "NVIDIA Q2 2024 data center server revenue"
    claims = [
        _claim(
            "NVIDIA Corporation's Data Center revenue was $10.32 billion in Q2 2024.",
            domain="yahoo.com",
            entities=["NVIDIA"],
        ),
        _claim(
            "Nvidia reported $10.32 billion in data center revenue, up 171% year over year.",
            domain="cnbc.com",
            entities=["Nvidia"],
        ),
        _claim(
            "Intel CEO Pat Gelsinger said 'build AI into every platform we build'.",
            domain="theverge.com",
            entities=["Intel", "Pat Gelsinger"],
            conf=0.8,
        ),
        _claim(
            "On a staggering $13.5 billion in revenue.",
            domain="theverge.com",
            entities=["Nvidia"],
            conf=0.85,
        ),
    ]
    filt = filter_claims_for_query(claims, q)
    assert all("Intel" not in c.statement for c in filt.kept)
    assert any("Intel" in r[0].statement for r in filt.rejected)
    assert any("10.32" in c.statement for c in filt.kept)


def test_numeric_corroboration_and_grounded_answer() -> None:
    q = "NVIDIA Q2 2024 data center server revenue"
    a = _claim(
        "NVIDIA Corporation's Data Center revenue was $10.32 billion in Q2 2024.",
        domain="yahoo.com",
        entities=["NVIDIA"],
    )
    b = _claim(
        "Nvidia reported $10.32 billion in data center revenue, up 171% year over year.",
        domain="cnbc.com",
        entities=["Nvidia"],
    )
    intel = _claim(
        "Intel CEO Pat Gelsinger said 'build AI into every platform'.",
        domain="theverge.com",
        entities=["Intel"],
        conf=0.8,
    )
    total = _claim(
        "On a staggering $13.5 billion in revenue.",
        domain="theverge.com",
        entities=["Nvidia"],
        conf=0.85,
    )
    filt = filter_claims_for_query([a, b, intel, total], q)
    boost_numeric_consensus(filt.kept)
    assert max(c.consensus_score for c in filt.kept) >= 0.4

    ans = build_conflict_aware_answer(q, filt.kept, rejected_by_filter=filt.rejected)
    assert ans.answer is not None
    assert "10.32" in ans.answer
    assert any("Intel" in (x.claim.statement + x.why_rejected) for x in ans.conflicts)
    assert detect_metric_key(a.statement) == "data_center_revenue"
    sig = money_signature(parse_money_values(a.statement)[0].amount_usd)
    assert sig == money_signature(parse_money_values(b.statement)[0].amount_usd)
