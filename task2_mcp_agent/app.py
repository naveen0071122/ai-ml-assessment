"""
app.py -- Streamlit UI for Task 2 (MCP Database Connector + Agentic
Retrieval). Task 2 is fundamentally a CLI/agent (see agent.py, demo.py) --
this file exists purely to give it a browser-accessible, deployable demo
with a public URL, matching Task 1's Streamlit deliverable.

Run locally:
    streamlit run app.py

The underlying MCPAgent (agent.py) is unchanged -- this is a thin UI
layer over it. Same three backends, same auto-detection: LLM_BACKEND env
var, else ANTHROPIC_API_KEY present -> Claude, else a local Ollama server
reachable -> Ollama, else the offline rule-based planner (no keys/cost).
"""
import asyncio
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import MCPAgent, _resolve_backend  # noqa: E402
import database  # noqa: E402

st.set_page_config(page_title="MCP Database Agent", layout="wide")


@st.cache_resource(show_spinner="Building the company database (first run only)...")
def ensure_database():
    database.main()
    return True


ensure_database()

st.title("MCP Database Connector + Agentic Retrieval")
st.caption(
    "A dummy company SQLite DB (employees / projects / issues), exposed to an "
    "LLM only through three MCP tools (list_tables, describe_schema, run_query) "
    "over a real MCP stdio connection -- no credentials ever reach the agent."
)

backend = _resolve_backend()
if backend == "claude":
    st.success("Running with **Claude** (`ANTHROPIC_API_KEY` set) — the real plan → act → observe loop.", icon="🤖")
elif backend == "ollama":
    st.success("Running with a **local Ollama model** — the real plan → act → observe loop.", icon="🖥️")
else:
    st.info(
        "Running with the **offline rule-based planner** (no `ANTHROPIC_API_KEY` set, "
        "no local Ollama server reachable). It reproduces the exact MCP call sequence "
        "for the questions below without any LLM -- good for verifying the connector "
        "and guardrails, not a general-purpose planner. See README for how to enable "
        "Claude or Ollama instead.",
        icon="ℹ️",
    )

st.subheader("Ask a question")
example_questions = [
    "Fetch employee details where department = 'AI'",
    "Which AI-team members have open issues on Project X?",
    "Show me the issues for the AI team",
]
question = st.selectbox(
    "Pick one of the required demo questions, or choose 'Custom question...' below",
    example_questions + ["Custom question..."],
)
if question == "Custom question...":
    question = st.text_input(
        "Type your own question",
        placeholder="e.g. Which projects does Karthik Iyer lead?",
    )
    if backend != "claude" and backend != "ollama":
        st.warning(
            "The offline planner only recognizes the three demo questions above "
            "verbatim. A custom question needs Claude or Ollama configured.",
            icon="⚠️",
        )

run = st.button("Ask the agent", type="primary")

if run and question:
    log_placeholder = st.empty()  # a single slot that gets overwritten in
                                    # place, instead of st.container() which
                                    # stacks a new code block on every call
    log_lines = []

    class _StreamToStreamlit:
        def write(self, text):
            if text.strip():
                log_lines.append(text)
                log_placeholder.code("\n".join(log_lines), language=None)

        def flush(self):
            pass

    async def _run():
        async with MCPAgent() as agent:
            return await agent.ask(question)

    st.subheader("Agent trace (tool calls)")
    old_stdout = sys.stdout
    sys.stdout = _StreamToStreamlit()
    try:
        answer = asyncio.run(_run())
    finally:
        sys.stdout = old_stdout

    st.subheader("Answer")
    st.write(answer)

st.divider()
st.caption(
    "See README.md for the full write-up: MCP connector flow, defense-in-depth "
    "read-only guardrails, and the required demos (JOIN query, error recovery, "
    "clarifying question) -- or run `python demo.py` for all four in one go."
)