Markdown
# Task 2 — MCP Database Connector + Agentic Retrieval

## What this is

A dummy SQLite company DB (`employees`, `projects`, `issues`, with FK relationships) exposed to an LLM **only** through three MCP tools (`list_tables`, `describe_schema`, `run_query`), plus a small agent that plans → acts → observes over those tools to answer natural-language questions, including one that needs a JOIN and one that needs error recovery / clarification. A Streamlit UI (`app.py`) wraps the same agent for a browser-accessible, deployable demo.

## Setup

```bash
cd task2_mcp_agent
pip install -r requirements.txt
python database.py          # (re)builds data/company.db from scratch

LLM Backend Options
For the real LLM-driven agent loop instead of the offline rule-based fallback, you can choose one of the following backends:

A. NVIDIA Nemotron (Recommended Cloud Option):
Get an API key from build.nvidia.com and export it:

export NVIDIA_API_KEY="nvapi-your-key-here"
export LLM_BACKEND="nvidia"             # Optional -- auto-detected if NVIDIA_API_KEY is present
export NVIDIA_MODEL="nvidia/nemotron-3.5-lightning-30b-a3b" # Optional default

B. Claude (Hosted Anthropic API):

export ANTHROPIC_API_KEY="sk-ant-your-key-here"

Note on Anthropic Keys: If using an identity-linked key (personal Console login), Anthropic requires a workspace ID header. You can set:

export ANTHROPIC_WORKSPACE_ID="wrkspc_..."

agent.py automatically attaches anthropic-workspace-id header if present.

C. Ollama (Free, Local, No API key):
Install Ollama, pull a tool-calling model, and point the agent at it:

ollama pull qwen2.5:7b     # One-time download
ollama serve                # Start local server
export LLM_BACKEND=ollama   # Optional -- auto-detected if local server is reachable

D. Offline Rule-Based Fallback:
Default when no API keys are set and no local Ollama instance is reachable. Reproduces exact MCP tool sequences for the three canned demo questions without incurring API costs.

Backend resolution priority order:
LLM_BACKEND env var (nvidia / claude / ollama / offline) → NVIDIA_API_KEY present → ANTHROPIC_API_KEY present → local Ollama server reachable → offline fallback.

Run
CLI Demo

python demo.py

Runs over a real MCP stdio subprocess (a genuine separate process using standard I/O):

"Fetch employee details where department = 'AI'" — Simple lookup.

"Which AI-team members have open issues on Project X?" — 3-table JOIN with automatic SQL error recovery.

"Show me the issues for the AI team" — Ambiguous query where the agent asks for clarification.

Guardrail proof: An attempted DELETE statement is rejected by run_query before touching the DB.

Or ask a custom question directly:

python agent.py "Which projects does Karthik Iyer lead?"

Browser UI (Streamlit)
Bash
streamlit run app.py

Opens a browser demo: pick demo questions from the dropdown or type custom questions. Shows the live tool-call trace and final answer.

Deployment on Streamlit Cloud
When deploying app.py to Streamlit Cloud, add your API keys under App Settings → Secrets:

Ini, TOML
NVIDIA_API_KEY = "nvapi-your-key-here"
LLM_BACKEND = "nvidia"
NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
app.py automatically syncs these secrets to os.environ so agent.py can access them securely without committing secrets to GitHub.

Connector Flow

agent.py (subprocess A)                 mcp_server.py (subprocess B)
 ─────────────────────────               ──────────────────────────────
 1. Spawns mcp_server.py over
    stdio, session.initialize()   ───▶
 2. list_tools()                  ───▶   Returns tool schemas
                                         (list_tables / describe_schema /
                                          run_query) — no data yet
 3. LLM (or offline planner)
    decides: call list_tables()   ───▶   SELECT name FROM sqlite_master...
                                  ◀───   ["employees","projects","issues"]
 4. LLM decides: call
    describe_schema("issues")     ───▶   PRAGMA table_info / foreign_key_list
                                  ◀───   {columns, foreign_keys}
 5. LLM writes SQL, calls
    run_query(sql)                ───▶   Auto-strip trailing ';'
                                         -> _enforce_read_only(sql)
                                         -> Open DB in mode=ro
                                         -> Execute, LIMIT-capped
                                  ◀───   {row_count, rows} OR {error}
 6. If {error}: LLM reads
    message, corrects SQL,
    retries call.
 7. LLM forms final response ───▶   Answers in plain English.

 Credentials & DB connection details never leave mcp_server.py.
DB_PATH is read from mcp_server.py's environment. The agent process only receives JSON tool results over stdio.

How Read-Only is Enforced (Defense in Depth)
DB-Connection Level: mcp_server.py opens SQLite with file:...?mode=ro (native read-only mode). The OS/DB driver physically rejects write operations.

MCP-Tool Level: run_query validates statements with regex before execution: must start with SELECT/WITH, must be a single statement, and must not contain forbidden keywords (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, etc.).

Auto-Strip Semicolons: Trailing semicolons (e.g., SELECT * FROM table;) are automatically stripped before single-statement checks, preventing false-positive syntax errors.

Row Cap: Every query is capped at MAX_ROWS = 200 server-side (LIMIT is appended or tightened), preventing large result sets from consuming context space.

Agent Turn Cap: MAX_TURNS is capped at 8 to allow sufficient turns for multi-step schema exploration while preventing runaway tool-calling loops.

Schema Exposure: On-Demand vs Full Schema
Schema is loaded on-demand (list_tables() → describe_schema(table)). Instead of dumping the full database schema into the initial prompt, the agent inspects only the tables relevant to the query. This saves tokens and avoids confusing the LLM with irrelevant table schemas.

Known Limitations & Production Improvements
Rule-based offline planner: Only handles the 3 canned questions verbatim. Active LLM backends (NVIDIA, Claude, Ollama) unlock arbitrary natural language questions.

Statement Timeout: A production implementation would add a hard query timeout on SQLite execution to prevent runaway complex JOINs on large databases.

Query Caching: In production, describe_schema results would be cached client-side to eliminate extra round-trips for static schemas.

Bugs Found & Resolved
Trailing Semicolon Errors: LLMs frequently output SELECT ...;. run_query and agent.py now strip trailing semicolons prior to single-statement validation.

SDK inputSchema Attribute Discrepancy: Resolved differences across mcp SDK versions using getattr(t, "inputSchema", None) or getattr(t, "input_schema", None).

Anthropic Identity-Linked Key Requirement: Added ANTHROPIC_WORKSPACE_ID support to handle identity-linked Anthropic API key authentication header requirements.

Max Turns Exhaustion: Increased agent loop MAX_TURNS from 5 to 8 to support complex multi-table schema inspection, query execution, and error recovery in a single session.
