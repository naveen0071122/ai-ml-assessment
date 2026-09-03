# MCP as a connector / abstraction layer between an LLM and a database

## The naive alternatives, and why each is worse

**1. Give the LLM a raw connection string.**
The model (or, more realistically, the application code assembling the
model's context) now holds a credential capable of arbitrary access —
reads, writes, schema changes, everything the DB user account can do.
That credential can leak through a prompt-injected document, a logged
transcript, or a careless tool implementation. There is also no seam at
which to enforce "read-only" or "200 rows max": those become conventions
the calling code has to remember to apply consistently everywhere the
connection string is used, rather than a property of one audited choke
point.

**2. Let the LLM generate and execute arbitrary SQL directly against the
DB (e.g. via a single `execute_sql` tool with no restrictions).**
Better than (1) — at least there's one tool boundary — but that boundary
does no work. Nothing stops a `DROP TABLE`, an unbounded `SELECT *` on a
100M-row table, or a query that joins in a way that's technically legal
SQL but operationally catastrophic. The model's job is to reason about
*what data answers the question*, not to be trusted as the last line of
defense against destructive or runaway queries.

## What MCP actually buys you

MCP turns "the LLM talks to a database" into "the LLM calls named,
typed, described tools, and a separate process decides what those tools
are allowed to do." Concretely, in this project:

- **Capability boundary, not just an API boundary.** `list_tables`,
  `describe_schema`, and `run_query` are the *entire* surface area the
  model can act through — there is no path from the LLM's output to the
  filesystem, the DB driver, or the credential. The MCP server enforces
  read-only and row limits in code the model never sees and cannot
  influence at the character level (contrast with a system-prompt rule,
  which is advisory).
- **Process isolation.** The server runs as its own subprocess with its
  own environment (`DB_PATH`, and in a real deployment, DB credentials).
  The agent process — the one whose context window a prompt-injection
  attack or a buggy tool result could most plausibly corrupt — never has
  those credentials in its address space at all, so there's nothing to
  exfiltrate even if the agent's reasoning is compromised.
- **A uniform, swappable interface.** The same `ClientSession` code in
  `agent.py` would work against a Postgres-backed MCP server, a
  read-replica, or a completely different data source, without the
  agent's tool-calling logic changing at all. This is the same value
  proposition as an ODBC driver or a REST API in front of a DB, adapted
  to the shape an LLM tool-use loop expects (named tools + JSON
  schemas), and it composes with the ecosystem of other MCP servers
  (files, ticketing systems, etc.) through one protocol instead of one
  bespoke integration per data source.
- **An auditable choke point.** Because every DB interaction is one of
  three named tool calls, logging/observability is trivial and complete
  — you can log every `run_query` call and its outcome without having to
  intercept raw SQL scattered through prompt strings.

## Where the analogy ends

MCP doesn't magically make the underlying access safe — it only makes
*where* to put the safety logic obvious and *hard to bypass by mistake*.
The actual guardrails (read-only connection, statement allow-list, row
cap) are ordinary application code living in `mcp_server.py`; MCP's
contribution is forcing all LLM-originated DB access through that one
piece of code instead of leaving the choice of "did we remember to check
this" to whichever tool implementation an engineer wrote most recently.
