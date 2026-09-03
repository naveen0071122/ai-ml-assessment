# Task 2 -- MCP Database Connector + Agentic Retrieval, containerized.
#
# This is a CLI/agent service, not a web app -- there is no browser UI to
# expose a URL for (see Task 1's Dockerfile for the deployed web demo).
# Containerizing it still matters for enterprise reproducibility: the
# SQLite DB, MCP server, and agent all run inside one pinned environment,
# with no host Python/dependency drift between dev machines and CI.
#
# Runs in OFFLINE rule-based planner mode by default (no LLM key needed) --
# set ANTHROPIC_API_KEY (Claude) as a container env var at deploy time for
# the real plan-act-observe loop instead -- see task2_mcp_agent/README.md.
#
# Build:
#   docker build -t task2-mcp-agent .
# Run the full demo (builds the DB fresh, runs all 4 required scenarios):
#   docker run --rm task2-mcp-agent
# Run a one-off question instead:
#   docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY task2-mcp-agent \
#     python agent.py "Which projects does Karthik Iyer lead?"

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the SQLite DB at image-build time so `docker run` is instant and
# self-contained (no separate init step required by the caller).
RUN python database.py

# Default command demonstrates the full required workflow end-to-end;
# override with `docker run ... python agent.py "<question>"` for
# one-off use, as documented above.
CMD ["python", "demo.py"]
