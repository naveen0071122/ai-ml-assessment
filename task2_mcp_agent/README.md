# Task 2 — MCP Database Connector + Agentic Retrieval

## What this is

A dummy SQLite company DB (`employees`, `projects`, `issues`, with FK
relationships) exposed to an LLM **only** through three MCP tools
(`list_tables`, `describe_schema`, `run_query`), plus a small agent that
plans → acts → observes over those tools to answer natural-language
questions, including one that needs a JOIN and one that needs error
recovery / clarification. A Streamlit UI (`app.py`) wraps the same agent
for a browser-accessible, deployable demo.

## Setup

```bash
cd task2_mcp_agent
pip install -r requirements.txt
python database.py          # (re)builds data/company.db from scratch
```

Optional — for the real LLM-driven agent loop instead of the offline
rule-based fallback, two choices:

**A. Claude (hosted API, needs a key):**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

If your key is an **identity-linked key** (created against a personal
Console login rather than scoped to one workspace), the Anthropic API
requires a workspace ID on every request, or you'll get:

400 - anthropic-workspace-id is required when authenticating with an
identity-linked API key; send the id of the workspace this request acts in.

Confirmed hitting this with a real key during development. Two fixes,
either works:
- **Simplest:** when creating the key at
  [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys),
  set its **Linked account / Workspace** to one specific workspace
  instead of leaving it identity-linked — this avoids needing a
  workspace ID at all.
- **Or:** keep the identity-linked key and also set:
```bash
  export ANTHROPIC_WORKSPACE_ID=wrkspc_...   # find it under Settings -> Workspaces
```
  `agent.py`'s `ask_llm()` reads this and sends it as the
  `anthropic-workspace-id` header automatically if present; if unset,
  behavior is unchanged from before.

Separately, a valid key still needs **API credits** in
[Plans & Billing](https://console.anthropic.com/settings/billing) — a
zero-balance account fails with `"Your credit balance is too low..."`
regardless of the workspace-id fix above. Confirmed this is a distinct
failure mode from the auth issue during development; both are billing/
account-level, not code issues.

**B. Ollama (free, local, no API key)** — install
[Ollama](https://ollama.com), pull a tool-calling-capable model, and
point the agent at it:
```bash
ollama pull qwen2.5:7b     # one-time download, ~4.7GB
ollama serve                # usually auto-starts after install
export LLM_BACKEND=ollama   # optional -- auto-detected if a local server is reachable
```

Backend selection order: `LLM_BACKEND` env var (`claude` / `ollama` /
`offline`) if set, else `ANTHROPIC_API_KEY` present → Claude, else a
local Ollama server reachable → Ollama, else → offline rule-based
fallback. `agent.py`'s docstring has the full explanation.

## Run

### CLI demo

```bash
python demo.py
```

This runs, over a **real MCP stdio subprocess** (not a direct function
call):
1. `"Fetch employee details where department = 'AI'"`
2. `"Which AI-team members have open issues on Project X?"` — a 3-table
   JOIN, with a deliberately typo'd column name on the first attempt so
   you can see the agent read the SQLite error and retry with a corrected
   query.
3. `"Show me the issues for the AI team"` — deliberately ambiguous (more
   than one AI project exists), so the agent asks a clarifying question
   instead of guessing.
4. A guardrail proof: an attempted `DELETE` is rejected by `run_query`
   before it ever reaches the database.

Or ask a one-off question directly:

```bash
python agent.py "Which projects does Karthik Iyer lead?"
```

Without any LLM backend configured, `agent.py` falls back to a small
rule-based planner that reproduces the exact MCP call sequence for the
three demo questions above, so the connector and guardrails can be
graded end-to-end without API credentials or a local model. This is
logged explicitly as `OFFLINE MODE` — it exists for reproducibility, not
as the intended production path. With Claude or Ollama configured, the
LLM genuinely chooses which tools to call, in what order, and how to
react to a failed query — including arbitrary custom questions, not just
the three canned ones.

### Browser UI

```bash
streamlit run app.py
```

Opens a browser demo: pick one of the three required questions from a
dropdown (works in every mode, including offline), or choose "Custom
question..." to type your own (needs Claude or Ollama configured — the
offline planner only recognizes the three canned questions verbatim).
Shows the live tool-call trace and the final answer. The database is
built automatically on first load if it doesn't exist yet.

## Connector flow

```
 agent.py (subprocess A)                mcp_server.py (subprocess B)
 ─────────────────────────              ──────────────────────────────
 1. spawns mcp_server.py over
    stdio, session.initialize()  ───▶
 2. list_tools()                 ───▶   returns tool schemas
                                          (list_tables / describe_schema /
                                           run_query) — no data yet
 3. LLM (or offline planner)
    decides: call list_tables()  ───▶   SELECT name FROM sqlite_master...
                                  ◀───   ["employees","projects","issues"]
 4. LLM decides: call
    describe_schema("issues")    ───▶   PRAGMA table_info / foreign_key_list
                                  ◀───   {columns, foreign_keys}
 5. LLM writes SQL, calls
    run_query(sql)                ───▶  _enforce_read_only(sql)
                                          -> open DB in mode=ro
                                          -> execute, LIMIT-capped
                                  ◀───   {row_count, rows} OR {error}
 6. If {error}: LLM reads the
    message, regenerates SQL,
    goes back to step 5.
 7. LLM has enough data ->
    answers in plain English.
```

**Credentials never leave `mcp_server.py`.**

`DB_PATH` is read from that
process's own environment (`os.environ`); the agent process only ever
sees JSON tool results over stdio. There is no connection string, no
file path, and no SQL the agent didn't itself generate anywhere in the
message history sent to the LLM's context except the query text it wrote
— it never sees `sqlite3.connect(...)` or the DB file location.

## How read-only is enforced (defence in depth)

1. **DB-connection level:** `mcp_server.py` opens SQLite with
   `file:...?mode=ro` — a native read-only handle. Even a bug in the SQL
   allow-list below can't produce a write, because the OS-level file
   handle physically rejects it.
2. **MCP-tool level:** `run_query` regex-checks the statement before
   executing: must start with `SELECT`/`WITH`, must be a single
   statement, and must not contain `INSERT/UPDATE/DELETE/DROP/ALTER/
   CREATE/ATTACH/PRAGMA/VACUUM/...`. This blocks write attempts even if
   the connection mode were ever changed by mistake.
3. **Row cap:** every query is capped at `MAX_ROWS = 200` server-side
   (`LIMIT` is appended, or a looser existing `LIMIT` is tightened), so a
   careless "get everything" query can't blow up token cost or memory.

I chose to enforce this at the **MCP tool boundary**, not in the system
prompt, because a prompt instruction ("please only SELECT") is advice the
model can be talked out of by a cleverly-worded question or a bug in the
planner logic — it constrains behavior, not capability. The read-only
connection + regex check constrain what is *physically possible* to
execute, regardless of what the LLM asks for. The prompt still tells the
model it's read-only too (defence in depth: if the tool ever rejects a
call, the model should understand why and not keep retrying the same
kind of statement).

## How much schema does the LLM see?

**On-demand, not full-schema-up-front.** The agent calls `list_tables()`
then `describe_schema()` only for the tables it decides are relevant to
the question. For this 3-table demo the token cost difference is
negligible, but the pattern matters at real scale: a production DB can
have hundreds of tables, and dumping the full schema into every prompt
(a) wastes tokens on tables irrelevant to 95% of questions and (b)
increases the chance the model gets confused by an irrelevant
similarly-named column. The trade-off is one extra round-trip (schema
lookup before the real query) per unfamiliar table — acceptable latency
for an analytics agent, not acceptable for a sub-100ms path. If this were
powering a low-latency product surface instead of an internal Q&A tool,
I'd cache `describe_schema` results in the agent process (schema changes
rarely) rather than re-fetching every call.

## Guardrail / error-recovery evidence

See `demo.py` output (or `app.py`'s live trace): attempt 1 of the JOIN
query uses a misspelled `employes` table and gets back `{"error":
"SQLite error: no such table: employes", ...}`; the agent parses that,
corrects the identifier, and succeeds on attempt 2. A separate `DELETE`
call to `run_query` is rejected with `{"error": "Only SELECT (or WITH
... SELECT) statements are permitted."}` before touching the database.

## MCP as a connector layer — see `writeup_mcp_as_connector.md`

## Known limitations
- The offline planner only handles the three demo questions verbatim
  (string-matching), not arbitrary NL — it's a grading fallback, not a
  general capability. Claude or Ollama unlocks the general case.
- `run_query` allows arbitrary read SQL (any SELECT), not a fixed set of
  parameterised queries — appropriate for an internal analyst-agent tool,
  but I would tighten this to an allow-list of query *templates* before
  exposing it to an external/untrusted caller.
- No query cost/complexity limit beyond row count (e.g. a full cross
  join with no `LIMIT` on a much bigger DB could still be slow before it
  gets capped at the output stage). A production version would add a
  statement timeout on the SQLite connection.
- **A public deployment of `app.py` should never carry a real API key as
  a baked-in secret.** If deployed with `ANTHROPIC_API_KEY` set (e.g. via
  a hosting platform's secret store), anyone who finds the URL can spend
  your API credits. For a public demo, either leave it in offline mode,
  or add the key only temporarily for a live walkthrough and remove it
  immediately after — see root `DEPLOYMENT.md`.

## Bugs found via real testing (and fixed)
- **`inputSchema` vs. `input_schema`:** `agent.py` originally read the
  MCP tool schema as `t.inputSchema` (camelCase), which matched the
  `mcp` SDK version used in development but crashed with
  `AttributeError` on a different `mcp` package version (2.1.1) installed
  by a real user, where the field is named `t.input_schema` (snake_case).
  Now reads `getattr(t, "inputSchema", None) or getattr(t, "input_schema",
  None)` so it works across SDK versions. Confirmed working against both
  naming conventions.
- **Identity-linked API key needs a workspace ID:** see the `Setup`
  section above — `ask_llm()` now sends `anthropic-workspace-id` when
  `ANTHROPIC_WORKSPACE_ID` is set, fixing a 400 error a real identity-linked
  key hit immediately on the first live Claude call. Confirmed the fix
  resolves the auth error (a separate, unrelated billing-credits error
  can still occur afterward — that's an account state, not a code bug).