# Task 2 -- MCP Database Connector + Agentic Retrieval, containerized.
#
# This is a CLI/agent service, not a web app -- there is no browser UI to
# expose a URL for.
# Containerizing it ensures enterprise reproducibility: the SQLite DB, 
# MCP server, and agent all run inside one pinned environment.
#
# Build:
#   docker build -t task2-mcp-agent .
# Run the full demo:
#   docker run --rm task2-mcp-agent
# Run a one-off question:
#   docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY task2-mcp-agent \
#     python agent.py "Which projects does Karthik Iyer lead?"

FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (real-time logging)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the SQLite DB at image-build time for instant self-contained execution
RUN python database.py

# Default command runs full demo workflow
CMD ["python", "demo.py"]