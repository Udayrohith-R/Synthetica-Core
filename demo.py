#!/usr/bin/env python3
"""
Synthetica-Core vs standard Exa RAG demo.

1. Hit Exa directly, dump raw chunks, and show a naïve LLM collapsing on
   conflicting revenue figures.
2. Hit local ``POST /v1/research/query`` and print atomic claims, consensus
   vs contradictions, CAMMR-ranked anchors, and a latency comparison.

Usage
-----
    # terminal A
    uvicorn app.main:app --host 0.0.0.0 --port 8000

    # terminal B
    python demo.py
    python demo.py --query "NVIDIA Q2 FY2025 data center revenue"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from exa_py import Exa

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_QUERY = "NVIDIA Q2 2024 data center server revenue"
DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_TOP_N = 5
CHUNK_PREVIEW_CHARS = 420
MONEY_RE = re.compile(
    r"\$?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s?(?:billion|million|bn|m)?|"
    r"\$\s?\d+(?:\.\d+)?\s?(?:billion|million|bn|m)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Terminal formatting (no emojis — plain ANSI)
# ---------------------------------------------------------------------------

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    WHITE = "\033[97m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def style(text: str, *codes: str) -> str:
    if not _supports_color():
        return text
    return f"{''.join(codes)}{text}{C.RESET}"


def rule(title: str = "", char: str = "─", width: int = 78) -> None:
    if title:
        pad = max(0, width - len(title) - 4)
        left = pad // 2
        right = pad - left
        print(style(f"{char * left}  {title}  {char * right}", C.DIM))
    else:
        print(style(char * width, C.DIM))


def heading(text: str) -> None:
    print()
    rule()
    print(style(f"  {text}", C.BOLD, C.CYAN))
    rule()


def subheading(text: str) -> None:
    print(style(f"\n▸ {text}", C.BOLD, C.WHITE))


def kv(key: str, value: Any) -> None:
    print(f"  {style(key + ':', C.DIM)} {value}")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@dataclass
class ExaChunk:
    index: int
    title: str
    url: str
    text: str
    score: float | None

    @property
    def domain(self) -> str:
        host = urlparse(self.url).hostname or "unknown"
        return host[4:] if host.startswith("www.") else host


def _truncate(text: str, limit: int = CHUNK_PREVIEW_CHARS) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _extract_money(text: str) -> list[str]:
    return [m.group(0).strip() for m in MONEY_RE.finditer(text or "")]


def _load_env() -> None:
    load_dotenv()


# ---------------------------------------------------------------------------
# Step 1 — Exa direct (standard RAG baseline)
# ---------------------------------------------------------------------------

def fetch_exa_chunks(query: str, *, top_n: int = DEFAULT_TOP_N) -> tuple[list[ExaChunk], float]:
    api_key = os.getenv("EXA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set (load from .env or the environment)")

    client = Exa(api_key=api_key)
    t0 = time.perf_counter()
    # Current Exa API: search() + contents (not deprecated search_and_contents).
    response = client.search(
        query,
        type="auto",
        num_results=top_n,
        contents={"text": {"max_characters": 12_000}},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    chunks: list[ExaChunk] = []
    for i, item in enumerate(response.results, start=1):
        text = getattr(item, "text", None) or ""
        if not text:
            highlights = getattr(item, "highlights", None) or []
            text = "\n".join(str(h) for h in highlights)
        chunks.append(
            ExaChunk(
                index=i,
                title=getattr(item, "title", None) or "(untitled)",
                url=getattr(item, "url", None) or "",
                text=text,
                score=getattr(item, "score", None),
            )
        )
    return chunks, elapsed_ms


def print_exa_chunks(chunks: list[ExaChunk], elapsed_ms: float) -> None:
    heading("BASELINE — Standard Exa RAG (raw chunks)")
    kv("latency", f"{elapsed_ms:.1f} ms")
    kv("chunks", len(chunks))

    all_figures: list[str] = []
    for chunk in chunks:
        figures = _extract_money(chunk.text)
        all_figures.extend(figures)
        subheading(f"Chunk {chunk.index} · {chunk.domain}")
        kv("title", chunk.title)
        kv("url", chunk.url)
        if chunk.score is not None:
            kv("score", f"{chunk.score:.4f}")
        kv("preview", _truncate(chunk.text))
        if figures:
            unique_figs = list(dict.fromkeys(figures))[:6]
            kv("$-figures", style(", ".join(unique_figs), C.YELLOW))

    unique = list(dict.fromkeys(all_figures))
    subheading("Conflicting dollar figures across chunks")
    if len(unique) >= 2:
        print(
            style(
                "  Multiple distinct revenue figures appear in the retrieved context — "
                "a naïve RAG LLM will often blend or pick one arbitrarily.",
                C.YELLOW,
            )
        )
        for fig in unique[:12]:
            print(f"    · {style(fig, C.BOLD, C.YELLOW)}")
    elif unique:
        print(f"  Only one figure pattern spotted: {unique[0]}")
    else:
        print(style("  No clear $-figures regex-matched in previews.", C.DIM))


def naive_llm_answer(query: str, chunks: list[ExaChunk]) -> tuple[str, float]:
    """Stuff Exa chunks into a single prompt — the classic confused-RAG path."""
    api_key = (os.getenv("GROQ_API_KEY") or os.getenv("TOGETHER_API_KEY") or "").strip()
    base = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model = os.getenv("LLM_MODEL_FAST", "llama-3.3-70b-versatile")

    if not api_key and "localhost" not in base and "127.0.0.1" not in base:
        return (
            "[skipped] No GROQ_API_KEY / TOGETHER_API_KEY — cannot run naïve LLM baseline.",
            0.0,
        )

    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[Source {c.index}] {c.title} ({c.url})\n{_truncate(c.text, 900)}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a financial research assistant. Answer using ONLY the provided "
        "web snippets. Be concise. If sources disagree, still give a single "
        "best-guess number — do not refuse."
    )
    user = (
        f"Question: {query}\n\n"
        f"Retrieved snippets:\n{context}\n\n"
        "What was the Q2 server / data-center revenue figure? "
        "Reply in 2–4 sentences with a single definitive number."
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    t0 = time.perf_counter()
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{base}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content).strip(), elapsed_ms


def print_naive_llm(answer: str, elapsed_ms: float) -> None:
    subheading("Naïve LLM answer over raw Exa chunks (often blends conflicts)")
    kv("latency", f"{elapsed_ms:.1f} ms")
    print(style(f"\n  {answer}\n", C.RED))
    print(
        style(
            "  ↑ Standard RAG has no contradiction graph / CAMMR — conflicting "
            "publisher numbers collapse into one fluent-sounding guess.",
            C.DIM,
        )
    )


# ---------------------------------------------------------------------------
# Step 2 — Synthetica research API
# ---------------------------------------------------------------------------

def call_synthetica(
    query: str,
    *,
    api_base: str,
    required_credibility: float = 0.80,
) -> tuple[dict[str, Any], float]:
    url = f"{api_base.rstrip('/')}/v1/research/query"
    payload = {
        "agent_id": "demo-script",
        "query": query,
        "temporal_scope": {"start_year": 2024, "end_year": 2025},
        "required_credibility": required_credibility,
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(url, json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Synthetica API {resp.status_code} from {url}: {resp.text[:800]}"
            )
        return resp.json(), elapsed_ms


def _classify_claims(
    claims: list[dict[str, Any]],
    *,
    consensus_floor: float = 0.55,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corroborated = [c for c in claims if float(c.get("consensus_score") or 0) >= consensus_floor]
    flagged = [c for c in claims if float(c.get("consensus_score") or 0) < consensus_floor]
    return corroborated, flagged


def print_synthetica(result: dict[str, Any], elapsed_ms: float) -> None:
    heading("SYNTHETICA — /v1/research/query")
    kv("session_id", result.get("session_id"))
    kv("status", result.get("status"))
    kv(
        "factual_consensus_index",
        f"{float(result.get('factual_consensus_index') or 0):.3f}",
    )
    kv("entropy_H(R_q)", f"{float(result.get('entropy_score') or 0):.4f}")
    api_latency = result.get("latency_ms")
    kv(
        "end-to-end latency",
        style(
            f"{float(api_latency if api_latency is not None else elapsed_ms):.1f} ms",
            C.BOLD,
            C.GREEN,
        ),
    )
    steps = result.get("step_ms") or {}
    if steps:
        kv("step_ms", ", ".join(f"{k}={v}" for k, v in steps.items()))

    # --- Cooked path: conflict-aware grounded answer ---
    grounded = result.get("grounded_answer") or {}
    subheading("Grounded answer (conflict-aware)")
    if grounded.get("answer"):
        print(style(f"  → {grounded.get('answer')}", C.BOLD, C.GREEN))
        kv(
            "    confidence",
            f"{float(grounded.get('confidence') or 0):.3f}",
        )
        kv("    metric_key", grounded.get("metric_key"))
        kv(
            "    act_safe",
            style(
                str(bool(grounded.get("act_safe"))),
                C.GREEN if grounded.get("act_safe") else C.YELLOW,
            ),
        )
        support = list(grounded.get("support") or [])
        if support:
            print(style(f"  support ({len(support)})", C.CYAN))
            for item in support:
                claim = item.get("claim") or {}
                print(
                    style("    + ", C.GREEN)
                    + f"{claim.get('statement')} "
                    + style(f"[{claim.get('source_domain')}]", C.DIM)
                )
        conflicts = list(grounded.get("conflicts") or [])
        if conflicts:
            print(style(f"  rejected ({len(conflicts)})", C.RED))
            for item in conflicts[:8]:
                claim = item.get("claim") or {}
                print(
                    style("    ! ", C.RED)
                    + f"{claim.get('statement')}"
                )
                print(
                    style(
                        f"      why: {item.get('why_rejected')}",
                        C.DIM,
                    )
                )
    else:
        print(style("  (no grounded answer — requires clarification)", C.YELLOW))

    claims = list(result.get("claims") or [])
    corroborated, flagged = _classify_claims(claims)

    # --- Atomic claims ---
    subheading(f"Query-filtered Atomic Claims ({len(claims)})")
    if not claims:
        print(style("  (none returned — check extractor keys / credibility gate)", C.DIM))
    for i, claim in enumerate(claims, start=1):
        conf = float(claim.get("confidence_score") or 0)
        cons = float(claim.get("consensus_score") or 0)
        print(
            f"  {style(f'[{i}]', C.CYAN)} "
            f"{claim.get('statement')}"
        )
        print(
            f"      conf={conf:.2f}  consensus={cons:.2f}  "
            f"domain={claim.get('source_domain')}"
        )
        ents = claim.get("entities") or []
        if ents:
            print(f"      entities={', '.join(ents)}")

    # --- Corroborated vs contradictions ---
    subheading("Corroborated facts")
    if corroborated:
        for claim in corroborated:
            print(
                style("  + ", C.GREEN)
                + f"{claim.get('statement')} "
                + style(
                    f"(consensus={float(claim.get('consensus_score') or 0):.2f})",
                    C.DIM,
                )
            )
    else:
        print(style("  (none above consensus floor — graph may still be warming)", C.DIM))

    subheading("Flagged contradictions / low-consensus claims")
    if flagged:
        for claim in flagged:
            print(
                style("  ! ", C.RED)
                + f"{claim.get('statement')} "
                + style(
                    f"(consensus={float(claim.get('consensus_score') or 0):.2f})",
                    C.DIM,
                )
            )
    else:
        print(style("  (none — selected slate is internally consistent)", C.DIM))

    # --- CAMMR ranked output with anchors ---
    subheading("Final CAMMR re-ranked consensus (with source anchors)")
    if not claims:
        print(style("  (empty slate)", C.DIM))
    for rank, claim in enumerate(claims, start=1):
        print(
            f"  {style(f'#{rank}', C.BOLD, C.MAGENTA)}  "
            f"{claim.get('statement')}"
        )
        print(
            f"      source   : {style(str(claim.get('source_domain')), C.CYAN)}"
        )
        print(
            f"      anchor   : {style(str(claim.get('citation_anchor')), C.GREEN)}"
        )
        print(
            f"      consensus: {float(claim.get('consensus_score') or 0):.3f}  "
            f"confidence: {float(claim.get('confidence_score') or 0):.3f}"
        )


def print_latency_comparison(
    *,
    exa_ms: float,
    naive_llm_ms: float,
    synthetica_ms: float,
) -> None:
    heading("LATENCY COMPARISON")
    baseline_total = exa_ms + naive_llm_ms
    rows = [
        ("Exa search (direct)", exa_ms),
        ("Naïve LLM over Exa chunks", naive_llm_ms),
        ("Baseline total (Exa + LLM)", baseline_total),
        ("Synthetica /v1/research/query", synthetica_ms),
    ]
    width = max(len(name) for name, _ in rows)
    for name, ms in rows:
        bar_len = max(1, int(ms / 50)) if ms > 0 else 0
        bar = "█" * min(bar_len, 40)
        color = C.GREEN if "Synthetica" in name else C.DIM
        print(
            f"  {name:<{width}}  "
            f"{style(f'{ms:8.1f} ms', C.BOLD, color)}  "
            f"{style(bar, color)}"
        )

    if synthetica_ms > 0 and baseline_total > 0:
        delta = baseline_total - synthetica_ms
        note = (
            f"Synthetica faster by {delta:.1f} ms"
            if delta > 0
            else f"Synthetica slower by {-delta:.1f} ms (full graph+CAMMR path)"
        )
        print(style(f"\n  {note}", C.BOLD))
        print(
            style(
                "  Note: Synthetica latency includes Exa + 70B extraction + Neo4j + CAMMR "
                "inside one request — compare apples-to-apples with baseline total.",
                C.DIM,
            )
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: Exa RAG confusion vs Synthetica CAMMR consensus",
    )
    parser.add_argument(
        "--query",
        default=os.getenv("DEMO_QUERY", DEFAULT_QUERY),
        help=f'Conflicting-data research query (default: "{DEFAULT_QUERY}")',
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("DEMO_API_BASE", DEFAULT_API_BASE),
        help=f"Synthetica base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Exa result count for the baseline path",
    )
    parser.add_argument(
        "--credibility",
        type=float,
        default=0.80,
        help="required_credibility for /v1/research/query",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the naïve LLM baseline (Exa chunks only)",
    )
    return parser.parse_args()


def main() -> int:
    _load_env()
    args = parse_args()

    print(style("\nSYNTHETICA-CORE DEMO", C.BOLD, C.CYAN))
    print(style("Exa RAG baseline  vs  CAMMR + entropy consensus\n", C.DIM))
    kv("query", style(args.query, C.BOLD))
    kv("api", args.api_base)

    # --- Baseline path ---
    try:
        chunks, exa_ms = fetch_exa_chunks(args.query, top_n=args.top_n)
    except Exception as exc:
        print(style(f"\nExa baseline failed: {exc}", C.RED), file=sys.stderr)
        return 1

    print_exa_chunks(chunks, exa_ms)

    naive_ms = 0.0
    if args.skip_llm:
        print(style("\n  [skipped naïve LLM baseline]", C.DIM))
    else:
        try:
            answer, naive_ms = naive_llm_answer(args.query, chunks)
            print_naive_llm(answer, naive_ms)
        except Exception as exc:
            print(style(f"\n  Naïve LLM baseline failed: {exc}", C.RED))
            naive_ms = 0.0

    # --- Synthetica path ---
    try:
        result, synth_ms = call_synthetica(
            args.query,
            api_base=args.api_base,
            required_credibility=args.credibility,
        )
    except Exception as exc:
        print(
            style(
                f"\nSynthetica API call failed: {exc}\n"
                "Is the server running?  uvicorn app.main:app --port 8000",
                C.RED,
            ),
            file=sys.stderr,
        )
        return 1

    print_synthetica(result, synth_ms)
    print_latency_comparison(
        exa_ms=exa_ms,
        naive_llm_ms=naive_ms,
        synthetica_ms=synth_ms,
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
