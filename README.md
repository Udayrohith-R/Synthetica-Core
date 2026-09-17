# Synthetica-Core

High-throughput search middleware for agent swarms. Synthetica turns noisy web retrieval into citation-anchored atomic claims, materializes them in a Neo4j knowledge graph, then re-ranks with **CAMMR** (Consensus-Aware Maximal Marginal Relevance) and routes on subgraph entropy `H(R_q)`.

## Architecture

```
Agent Swarm
    │
    ▼
[1] SSM + BM25 retrieval          → filtered snippets
    │
    ▼
[2] Llama-3.1-8B fact extractor   → AtomicClaim JSON
    │
    ▼
[3] Neo4j 4D property graph       → CORROBORATES / CONTRADICTS
    │
    ▼
[4] CAMMR + entropy gate H(R_q)
    ├─ H ≤ threshold  → fast-path citation JSON
    └─ H > threshold  → [5] deep arbitration → grounded synthesis
```

1. **Agent state-space retrieval** — AST / stack / state tensors + BM25 lexical index + Matryoshka projection; Exa for remote hits.
2. **Zero-latency fact extractor** — Llama-3.1-8B (cascade to 70B) via httpx; concurrent `asyncio.gather` over snippets; JSON repair for malformed LLM output.
3. **4D property graph** — `Document` / `Claim` / `Entity` / `Source` with `MENTIONS_ENTITY`, `CORROBORATES`, `CONTRADICTS`.
4. **CAMMR + entropy** — consensus-aware MMR; low `H(R_q)` → fast path; high → arbitration.
5. **Deep arbitration** — larger-model pass to resolve factual conflicts into grounded synthesis.

## Requirements

- Python **3.11+**
- Neo4j 5.x (optional for local graph writes; API still runs without it)
- API keys: **Exa** + **Groq** or **Together** (OpenAI-compatible LLM endpoint; vLLM works too)

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
# fill EXA_API_KEY, GROQ_API_KEY (or TOGETHER_API_KEY), NEO4J_*

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### Demo (Exa RAG vs Synthetica)

With the API running:

```powershell
python demo.py
python demo.py --query "NVIDIA Q2 FY2025 data center revenue"
```

## Configuration

Copy [`.env.example`](.env.example) → `.env`:

| Variable | Purpose |
|----------|---------|
| `EXA_API_KEY` | Neural web search |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Knowledge graph |
| `GROQ_API_KEY` or `TOGETHER_API_KEY` | Claim extraction + arbitration |
| `LLM_BASE_URL` | OpenAI-compatible base (`https://api.groq.com/openai/v1` or local vLLM) |
| `LLM_MODEL_FAST` | Default 8B cascade tier (`llama-3.1-8b-instant`) |
| `LLM_MODEL_DENSE` | Dense / legal-financial tier (`llama-3.3-70b-versatile`) |
| `ENTROPY_THRESHOLD` | `H(R_q)` gate (default `0.55`) |
| `CAMMR_LAMBDA` | Relevance vs diversity blend |

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness |
| `GET` | `/v1/architecture` | Live module map |
| `POST` | `/v1/research/query` | Primary research pipeline (`ResearchQueryRequest` → `SynthesisResponse`) |
| `POST` | `/v1/query` | Legacy full pipeline (`QueryRequest` → `QueryResponse`) |

### Research query example

```bash
curl -X POST http://localhost:8000/v1/research/query \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "cursor-1",
    "query": "NVIDIA Q2 2024 data center server revenue",
    "required_credibility": 0.80,
    "temporal_scope": {"start_year": 2023, "end_year": 2025}
  }'
```

Response shape (`SynthesisResponse`):

- `status`: `COMPLETED` or `REQUIRES_CLARIFICATION`
- `factual_consensus_index`, `entropy_score`
- `claims`: query-filtered `AtomicClaim` list (statement, confidence, entities, citation URL, consensus)
- `grounded_answer`: conflict-aware object — `answer`, `support[]`, `conflicts[]` with `why_rejected`, `act_safe`
- `latency_ms` / `step_ms`: end-to-end and per-stage timings

Cooked path defaults: extract top 3 of 4 Exa docs, query-window truncation, claim cache, drop off-topic entities/metrics, normalize money for corroboration.

## Project layout

```
app/
  main.py                 # FastAPI entrypoint
  config.py               # Pydantic Settings
  models/schemas.py       # AtomicClaim, ResearchQueryRequest, SynthesisResponse, …
  services/
    ssm_retrieval.py      # Module 1 — BM25 + Matryoshka hybrid retrieval
    extractor.py          # Module 2 — async 8B AtomicClaim extraction
    graph_service.py      # Module 3 — Neo4j async driver + Cypher
    cammr_engine.py       # Module 4 — CAMMR + H(R_q)
    arbitration.py        # Module 5 — deep conflict resolution
    pipeline.py           # End-to-end orchestrator
  utils/
    bm25.py
    entropy.py
demo.py                   # Side-by-side Exa vs Synthetica demo
```

## Development

```powershell
pip install -e ".[dev]"
ruff check app
pytest
```

Python package metadata lives in [`pyproject.toml`](pyproject.toml); runtime pins in [`requirements.txt`](requirements.txt).
