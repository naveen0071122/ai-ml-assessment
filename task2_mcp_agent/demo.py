"""
demo.py -- end-to-end demonstration for Task 2.

Runs, over the real MCP stdio transport (a genuine subprocess, not a direct
function call):
  1. A simple lookup: "Fetch employee details where department = 'AI'"
  2. A multi-table JOIN + follow-up: "Which AI-team members have open
     issues on Project X?" (includes one deliberate typo -> error recovery)
  3. A guardrail proof: an attempted DELETE is rejected by run_query.

Run:
    python demo.py
"""
import asyncio
import json
from agent import MCPAgent


async def main():
    async with MCPAgent() as agent:
        print("=" * 70)
        print("DEMO 1: simple lookup")
        print("=" * 70)
        print(await agent.ask("Fetch employee details where department = 'AI'"))

        print("\n" + "=" * 70)
        print("DEMO 2: multi-step JOIN + error recovery")
        print("=" * 70)
        print(await agent.ask("Which AI-team members have open issues on Project X?"))

        print("\n" + "=" * 70)
        print("DEMO 3: ambiguous question -> agent asks for clarification")
        print("=" * 70)
        print(await agent.ask("Show me the issues for the AI team"))

        print("\n" + "=" * 70)
        print("DEMO 4: guardrail proof -- write attempt is rejected")
        print("=" * 70)
        result = await agent.call_tool(
            "run_query", {"sql": "DELETE FROM employees WHERE id = 1"}
        )
        print(f"  run_query('DELETE FROM employees...') -> {json.dumps(result)}")
        assert "error" in result, "guardrail failed to block a write!"
        print("  Guardrail confirmed: write statement was rejected before touching the DB.")


if __name__ == "__main__":
    asyncio.run(main())
