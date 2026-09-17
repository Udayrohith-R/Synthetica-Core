"""
Conflict-aware answer assembly.

Builds a structured object Exa-style RAG never returns:
supported consensus fact + rejected rivals with explicit why_rejected reasons.
"""

from __future__ import annotations

from collections import defaultdict

from app.models.schemas import (
    AtomicClaim,
    ConflictAwareAnswer,
    ConflictEvidence,
    RejectedClaim,
)
from app.services.claim_relevance import QueryFocus, parse_query_focus, score_claim_relevance
from app.utils.metrics import (
    detect_metric_key,
    metric_compatible,
    money_signature,
    parse_money_values,
)


def build_conflict_aware_answer(
    query: str,
    ranked_claims: list[AtomicClaim],
    *,
    rejected_by_filter: list[tuple[AtomicClaim, str]] | None = None,
) -> ConflictAwareAnswer:
    """
    Derive the primary answered fact, supporting citations, and rejected conflicts.
    """
    focus = parse_query_focus(query)
    rejected_by_filter = rejected_by_filter or []

    if not ranked_claims:
        return ConflictAwareAnswer(
            answer=None,
            confidence=0.0,
            metric_key=next(iter(focus.metric_keys), None),
            support=[],
            conflicts=[
                RejectedClaim(claim=c, why_rejected=reason)
                for c, reason in rejected_by_filter[:8]
            ],
            act_safe=False,
        )

    # Prefer money-bearing, metric-compatible claims as answer candidates
    answer_claim = _pick_answer_claim(ranked_claims, focus)
    metric = detect_metric_key(answer_claim.statement) if answer_claim else None
    monies = parse_money_values(answer_claim.statement) if answer_claim else []
    answer_sig = (
        money_signature(max(monies, key=lambda m: m.amount_usd).amount_usd)
        if monies
        else None
    )

    support: list[ConflictEvidence] = []
    conflicts: list[RejectedClaim] = []

    # Support = same metric + same money signature (or same statement cluster)
    for claim in ranked_claims:
        c_metric = detect_metric_key(claim.statement)
        c_monies = parse_money_values(claim.statement)
        c_sig = (
            money_signature(max(c_monies, key=lambda m: m.amount_usd).amount_usd)
            if c_monies
            else None
        )

        if answer_claim and claim.claim_id == answer_claim.claim_id:
            support.append(
                ConflictEvidence(
                    claim=claim,
                    role="support",
                    note="primary consensus claim",
                )
            )
            continue

        if (
            answer_sig
            and c_sig == answer_sig
            and metric_compatible({metric} if metric else set(), c_metric)
        ):
            support.append(
                ConflictEvidence(
                    claim=claim,
                    role="support",
                    note=f"corroborates normalized amount ({c_sig})",
                )
            )
            continue

        # Metric mismatch vs query / answer
        if focus.metric_keys and c_metric and not metric_compatible(focus.metric_keys, c_metric):
            conflicts.append(
                RejectedClaim(
                    claim=claim,
                    why_rejected=(
                        f"metric mismatch: claim asserts {c_metric}, "
                        f"query asks for {sorted(focus.metric_keys)}"
                    ),
                )
            )
            continue

        if answer_sig and c_sig and c_sig != answer_sig and c_metric == metric:
            conflicts.append(
                RejectedClaim(
                    claim=claim,
                    why_rejected=(
                        f"conflicting figure for {metric or 'metric'}: "
                        f"{c_sig} vs consensus {answer_sig}"
                    ),
                )
            )
            continue

        if answer_claim and score_claim_relevance(claim, focus) < 0.55:
            conflicts.append(
                RejectedClaim(
                    claim=claim,
                    why_rejected="low query relevance vs selected consensus",
                )
            )

    # Attach filter rejects (Intel/AMD etc.) with their reasons
    seen_ids = {c.claim.claim_id for c in conflicts} | {e.claim.claim_id for e in support}
    for claim, reason in rejected_by_filter:
        if claim.claim_id in seen_ids:
            continue
        conflicts.append(RejectedClaim(claim=claim, why_rejected=reason))
        seen_ids.add(claim.claim_id)

    # Cap conflict list for agent payloads
    conflicts = conflicts[:12]

    confidence = 0.0
    if answer_claim:
        n_support_domains = len({e.claim.source_domain.lower() for e in support})
        confidence = min(
            1.0,
            0.55 * answer_claim.confidence_score
            + 0.35 * answer_claim.consensus_score
            + 0.10 * min(1.0, n_support_domains / 3.0),
        )

    act_safe = bool(answer_claim) and confidence >= 0.72 and len(support) >= 1

    answer_text = None
    if answer_claim:
        answer_text = answer_claim.statement
        if monies:
            primary = max(monies, key=lambda m: m.amount_usd)
            answer_text = (
                f"{answer_claim.statement} "
                f"[normalized≈${_fmt_usd(primary.amount_usd)}]"
            )

    return ConflictAwareAnswer(
        answer=answer_text,
        confidence=float(confidence),
        metric_key=metric or next(iter(focus.metric_keys), None),
        support=support,
        conflicts=conflicts,
        act_safe=act_safe,
    )


def _pick_answer_claim(claims: list[AtomicClaim], focus: QueryFocus) -> AtomicClaim:
    """Choose the best claim to ground the agent-facing answer."""
    scored: list[tuple[float, AtomicClaim]] = []
    for claim in claims:
        rel = score_claim_relevance(claim, focus)
        monies = parse_money_values(claim.statement)
        metric = detect_metric_key(claim.statement)
        money_bonus = 0.25 if (focus.wants_money and monies) else 0.0
        metric_bonus = 0.20 if metric_compatible(focus.metric_keys, metric) else 0.0
        # Prefer higher consensus among relevant money claims
        total = (
            0.40 * rel
            + 0.25 * claim.consensus_score
            + 0.15 * claim.confidence_score
            + money_bonus
            + metric_bonus
        )
        scored.append((total, claim))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _fmt_usd(amount: float) -> str:
    if amount >= 1_000_000_000:
        return f"{amount / 1_000_000_000:.2f}B"
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.2f}M"
    return f"{amount:,.0f}"


def group_claims_by_normalized_figure(
    claims: list[AtomicClaim],
) -> dict[tuple[str, str], list[AtomicClaim]]:
    """Debug/helper: (metric, money_sig) → claims."""
    groups: dict[tuple[str, str], list[AtomicClaim]] = defaultdict(list)
    for claim in claims:
        metric = detect_metric_key(claim.statement) or "unknown"
        monies = parse_money_values(claim.statement)
        if not monies:
            groups[(metric, "none")].append(claim)
            continue
        sig = money_signature(max(monies, key=lambda m: m.amount_usd).amount_usd)
        groups[(metric, sig)].append(claim)
    return groups
