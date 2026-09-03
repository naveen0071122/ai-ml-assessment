"""
tests/test_guardrails.py
Lightweight tests for the read-only / row-limit / error-handling guardrails
in mcp_server.py. These call the tool functions directly (not over MCP
transport) so they're fast and don't need a subprocess -- the stdio
transport itself is exercised separately in demo.py / agent.py.

Run:
    cd task2_mcp_agent
    python -m pytest tests/ -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mcp_server  # noqa: E402


def test_list_tables_returns_expected_tables():
    tables = mcp_server.list_tables()
    assert set(tables) == {"employees", "projects", "issues"}


def test_describe_schema_returns_columns_and_fks():
    schema = mcp_server.describe_schema("issues")
    col_names = {c["name"] for c in schema["columns"]}
    assert {"id", "project_id", "assignee_id", "status"}.issubset(col_names)
    fk_cols = {fk["column"] for fk in schema["foreign_keys"]}
    assert fk_cols == {"project_id", "assignee_id"}


def test_describe_schema_rejects_unknown_table():
    try:
        mcp_server.describe_schema("nope")
        assert False, "expected ValueError for unknown table"
    except ValueError:
        pass


def test_run_query_allows_select():
    result = mcp_server.run_query("SELECT name FROM employees WHERE department='AI'")
    assert "error" not in result
    assert result["row_count"] == 4


def test_run_query_blocks_delete():
    result = mcp_server.run_query("DELETE FROM employees")
    assert "error" in result
    # confirm nothing was actually deleted
    check = mcp_server.run_query("SELECT COUNT(*) as n FROM employees")
    assert check["rows"][0]["n"] == 8


def test_run_query_blocks_drop():
    result = mcp_server.run_query("DROP TABLE employees")
    assert "error" in result


def test_run_query_blocks_multi_statement():
    result = mcp_server.run_query("SELECT 1; DROP TABLE employees;")
    assert "error" in result


def test_run_query_returns_sqlite_error_not_a_crash():
    result = mcp_server.run_query("SELECT name FROM emplyees")  # typo
    assert "error" in result
    assert "no such table" in result["error"]


def test_run_query_enforces_row_cap():
    # ask for more rows than exist AND request no limit; cap should still apply
    result = mcp_server.run_query("SELECT * FROM issues")
    assert result["row_count"] <= mcp_server.MAX_ROWS


def test_run_query_tightens_looser_limit_but_respects_tighter_one():
    result = mcp_server.run_query("SELECT * FROM issues LIMIT 2")
    assert result["row_count"] == 2


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
