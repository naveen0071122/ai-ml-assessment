# Enterprise-Oriented Implementation Notes

This document collects the production/enterprise-readiness decisions
already made in this submission, and the gap between "assessment
prototype" and "enterprise production system" that a real rollout would
need to close. Written as a standalone reference for the interview
discussion, cross-referenced to the specific files where each concern is
actually implemented (not just described).

## 1. Containerization & deployment

- Each task has its own `Dockerfile` (`task1_multimodal_retrieval/Dockerfile`,
  `task2_mcp_agent/Dockerfile`), independently buildable — a real
  microservice boundary, not a monolith split after the fact.
- Root `docker-compose.yml` orchestrates both for local/CI use, with the
  web app (`task1-web`, long-running) and the batch job (`task2-agent`,
  one-shot) deliberately given different lifecycle semantics (`up` vs.
  `run --rm`) — conflating a web service and a batch job in one `up`
  invocation is exactly the kind of thing a real production compose file
  avoids.
- `DEPLOYMENT.md` covers getting a public URL (Hugging Face Spaces,
  Docker-native, or Render.com as backup) — both build directly from the
  same `Dockerfile` used for local dev, so "what's deployed" and "what
  you tested locally" are the same artifact, not a separate deploy-only
  config that could drift.

## 2. Secrets management

- No credential (DB path, API keys) ever appears in agent/LLM-facing
  code — see `task2_mcp_agent/README.md`'s "Connector flow" section for
  the specific mechanism (MCP server process boundary).
- `.env.example` documents every environment variable across both tasks
  with no real values; `.gitignore` excludes `.env` itself.
- `docker-compose.yml` reads secrets via `env_file: .env` at container
  runtime — never baked into an image layer, so an image can't leak a
  credential even if pushed to a public registry.
- Deployment guide explicitly calls out using the hosting platform's
  secret store (Hugging Face Space secrets, not a committed file) for
  any real API key used in a live deployment.

## 3. Read/write safety and data integrity (Task 2)

- Defense-in-depth read-only enforcement (DB-connection level, MCP-tool
  level, row cap) — see `task2_mcp_agent/mcp_server.py` and its README's
  "How read-only is enforced" section. This is the specific control that
  matters most in an enterprise context: an LLM-driven agent must not be
  able to mutate production data through a prompt-injection or planning
  error, and the enforcement here doesn't depend on the LLM behaving.

## 4. Observability

- Both tasks log structurally (`ingestion.py`, `mcp_server.py`,
  `agent.py`) — query/question, tool called, success/failure, timing —
  using Python's standard `logging` module rather than bare `print`
  (where it matters for the request path; a few `print` statements
  remain in CLI-only demo scripts where a human is watching stdout
  directly, not a log aggregator).
- Task 1's Streamlit UI surfaces per-stage latency (query understanding
  / candidate retrieval / VLM rerank) directly in the product UI, not
  just in a log file — an enterprise-minded choice: the person using the
  tool can see performance characteristics without needing log access.
- **Gap, honestly flagged:** neither task exports metrics to a real
  observability backend (Prometheus/Datadog/CloudWatch). Structured logs
  exist; a production rollout would add a metrics client and dashboards.
  Noted rather than faked.

## 5. Testing & CI-readiness

- 26 automated tests across both tasks (`task1_multimodal_retrieval/tests/`,
  `task2_mcp_agent/tests/`), runnable via plain `pytest` with no external
  service dependencies (offline-mock/rule-based fallbacks mean tests
  don't need API keys or a live LLM) — this is what makes them suitable
  for a CI pipeline as-is, not just local development.
- A real CI setup (not included, but the natural next step) would run
  `pytest` on every push, then build and push the Docker images on merge
  to main — the Dockerfiles are already CI-friendly (no interactive
  prompts, deterministic base image, pinned Python version).

## 6. Scalability — honestly scoped, not oversold

This submission intentionally does **not** pretend to be
horizontally-scalable production infrastructure; it's an assessment
prototype with the specific enterprise controls above already in place,
and the following gaps clearly named rather than hidden:

- **SQLite (Task 2)** is single-writer and file-based — fine for this
  demo's read-only agent workload, wrong for multi-tenant production
  write concurrency. The read-only connection + defense-in-depth
  guardrails would carry over unchanged to a Postgres-backed version
  (the MCP tool interface doesn't change); only `database.py`'s
  connection setup would need to swap.
- **TF-IDF (Task 1)** is an in-memory index rebuilt at process start —
  fine for a 12-document demo corpus, would need a real vector database
  (e.g. pgvector, Qdrant) and an incremental-indexing story for a corpus
  that changes over time in production.
- **The local Qwen-VL path** ties inference to whatever container it
  runs in — a real deployment would separate model serving (e.g. behind
  its own scaled inference endpoint) from the retrieval application, so
  the two scale independently.

## 7. Supporting documentation map

| Document | Covers |
|---|---|
| `README.md` (root) | Full setup, both tasks, environment variables, troubleshooting |
| `task1_multimodal_retrieval/README.md` | Task 1 design decisions, judgment calls, evaluation, real bugs found via local-model testing |
| `task2_mcp_agent/README.md` | Task 2 connector flow, guardrails, defense-in-depth reasoning |
| `task2_mcp_agent/writeup_mcp_as_connector.md` | Required MCP conceptual explanation |
| `task1_multimodal_retrieval/MANUAL_TESTING_GUIDE.md` | Exact queries/expected results for manual QA |
| `DEPLOYMENT.md` | Getting a live public URL, container verification |
| `ENTERPRISE_NOTES.md` (this file) | Enterprise-readiness decisions and honestly-scoped gaps |
| `reflection.md` | Required written reflection (Task 3) |
