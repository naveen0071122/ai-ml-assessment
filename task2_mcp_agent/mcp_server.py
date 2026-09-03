"""
mcp_server.py
Exposes company.db to an LLM via the Model Context Protocol (MCP).

Design decisions (see README for full discussion):
- Credentials / DB path live ONLY here, in this process's environment
  (DB_PATH env var, defaulting to data/company.db). The agent process never
  sees a connection string — it only sees tool names and JSON results.
- Defence in depth for "read-only":
    1. SQLite connection is opened in URI mode with `mode=ro`, so the OS/DB
       driver physically refuses writes regardless of what SQL text arrives.
    2. `run_query` additionally rejects any statement that is not a single
       SELECT (regex + sqlite3 statement check) before it ever reaches the
       read-only connection. This is belt-and-suspenders: even if someone
       swaps the connection mode, the tool-level check still blocks
       INSERT/UPDATE/DELETE/DROP/ATTACH/PRAGMA-write etc.
    3. A hard row limit (MAX_ROWS) is enforced server-side by appending
       `LIMIT` to any query that doesn't already have one tighter than the
       cap, so a runaway/careless query can't blow up context or cost.
- Schema is exposed on-demand (list_tables / describe_schema) rather than
  dumped in full up front, so the agent only pays token cost for the tables
  it actually needs (see README "How much does the LLM see?").

Run standalone for a quick manual check:
    python mcp_server.py --selftest

Run as an MCP stdio server (what the agent actually launches):
    python mcp_server.py
"""
import os
import re
import sqlite3
import sys
import json

from mcp.server.mcpserver import MCPServer

DB_PATH = os.environ.get(
    "DB_PATH", os.path.join(os.path.dirname(__file__), "data", "company.db")
)
MAX_ROWS = 200

mcp = MCPServer("company-db")


def _ro_connect() -> sqlite3.Connection:
    """Open the DB in SQLite's native read-only URI mode (guardrail #1)."""
    uri = f"file:{os.path.abspath(DB_PATH)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


_WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)


def _enforce_read_only(sql: str) -> None:
    """Guardrail #2: tool-level statement allow-list, independent of the
    connection mode above."""
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        raise ValueError("Only a single statement is allowed per call.")
    if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE):
        raise ValueError("Only SELECT (or WITH ... SELECT) statements are permitted.")
    if _WRITE_KEYWORDS.search(stripped):
        raise ValueError(
            "Query contains a disallowed keyword. This connector is read-only."
        )


def _apply_row_limit(sql: str, max_rows: int) -> str:
    """Guardrail #3: cap result size server-side."""
    if re.search(r"\bLIMIT\s+\d+\s*$", sql, re.IGNORECASE):
        # Respect an existing tighter limit, but still cap at MAX_ROWS.
        m = re.search(r"\bLIMIT\s+(\d+)\s*$", sql, re.IGNORECASE)
        existing = int(m.group(1))
        if existing > max_rows:
            sql = re.sub(r"\bLIMIT\s+\d+\s*$", f"LIMIT {max_rows}", sql, flags=re.IGNORECASE)
        return sql
    return f"{sql.rstrip()} LIMIT {max_rows}"


@mcp.tool()
def list_tables() -> list[str]:
    """List all table names available in the database."""
    conn = _ro_connect()
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return [r["name"] for r in rows]
    finally:
        conn.close()


@mcp.tool()
def describe_schema(table_name: str) -> dict:
    """Return column names/types and foreign keys for a single table.

    Call list_tables() first to discover valid table names.
    """
    conn = _ro_connect()
    try:
        cols = conn.execute(f"PRAGMA table_info({_safe_ident(table_name)})").fetchall()
        if not cols:
            raise ValueError(f"Unknown table: {table_name}")
        fks = conn.execute(f"PRAGMA foreign_key_list({_safe_ident(table_name)})").fetchall()
        return {
            "table": table_name,
            "columns": [
                {"name": c["name"], "type": c["type"], "pk": bool(c["pk"])} for c in cols
            ],
            "foreign_keys": [
                {"column": f["from"], "references_table": f["table"], "references_column": f["to"]}
                for f in fks
            ],
        }
    finally:
        conn.close()


@mcp.tool()
def run_query(sql: str) -> dict:
    """Execute a READ-ONLY SQL query (SELECT only) and return up to
    MAX_ROWS rows as JSON-serialisable records.

    Rejects any non-SELECT statement. On a SQL error, returns a dict with
    an "error" key containing the sqlite3 error message so the caller (the
    agent) can inspect it and retry with a corrected query — the agent
    does not get a raw traceback/crash.
    """
    try:
        _enforce_read_only(sql)
    except ValueError as e:
        return {"error": str(e)}

    limited_sql = _apply_row_limit(sql, MAX_ROWS)
    conn = _ro_connect()
    try:
        cur = conn.execute(limited_sql)
        rows = [dict(r) for r in cur.fetchall()]
        return {"row_count": len(rows), "rows": rows, "sql_executed": limited_sql}
    except sqlite3.Error as e:
        return {"error": f"SQLite error: {e}", "sql_attempted": limited_sql}
    finally:
        conn.close()


def _safe_ident(name: str) -> str:
    """Whitelist table/column identifiers used inside f-strings (PRAGMA
    doesn't support bind params) to avoid identifier-injection."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Invalid identifier: {name}")
    return name


def _selftest():
    print("Tables:", list_tables())
    print("Schema(issues):", json.dumps(describe_schema("issues"), indent=2))
    print("Query result:", json.dumps(
        run_query("SELECT name FROM employees WHERE department='AI'"), indent=2))
    print("Blocked write:", json.dumps(run_query("DELETE FROM employees"), indent=2))
    print("Bad SQL (for error-recovery demo):", json.dumps(
        run_query("SELECT name FROM emplyees"), indent=2))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        mcp.run()  # defaults to stdio transport, as launched by agent.py
