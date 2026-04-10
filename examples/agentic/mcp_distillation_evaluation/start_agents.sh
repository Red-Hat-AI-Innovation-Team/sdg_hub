#!/usr/bin/env bash
# Start LangGraph agents for MCP evaluation task generation.
#
# Each agent is a thin ReAct wrapper (create_react_agent) that connects to
# one MCP server via langchain_mcp_adapters. Agents are only needed for
# task generation (Section 2 of the notebook) — evaluation uses MCPAgentBlock
# directly and does NOT require these agents.
#
# Prerequisites:
#   1. MCP servers running (bash start_servers.sh)
#   2. langgraph + langchain_mcp_adapters installed
#   3. OPENAI_API_KEY set (agents use GPT-5.2 as the exploration model)
#
# Usage:
#   bash start_agents.sh          # start all agents
#   bash start_agents.sh --check  # check which agents are running
#   bash start_agents.sh --stop   # stop all agents

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Agent definitions ────────────────────────────────────────────────
# Format: DISPLAY_NAME|AGENT_PORT|MCP_SERVER_NAME|MCP_SERVER_PORT
AGENTS=(
    "Weather Data|2024|weather-data|8001"
    "Medical Calculator|2025|medical-calculator|8002"
    "Wikipedia|2026|wikipedia|8003"
    "Car Price Evaluator|2027|car-price-evaluator|8004"
    "Reddit|2028|reddit|8005"
    "DEX Paprika|2029|dex-paprika|8006"
)

check_port() {
    (echo > /dev/tcp/localhost/$1) 2>/dev/null
}

# ── --check ───────────────────────────────────────────────────────────
if [[ "${1:-}" == "--check" ]]; then
    echo "LangGraph Agent Status:"
    echo "─────────────────────────────────────────"
    for entry in "${AGENTS[@]}"; do
        IFS='|' read -r display_name port mcp_name mcp_port <<< "$entry"
        if check_port "$port"; then
            echo "  ✓ $display_name (port $port) → MCP localhost:$mcp_port"
        else
            echo "  ✗ $display_name (port $port)"
        fi
    done
    exit 0
fi

# ── --stop ────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping LangGraph agents..."
    for entry in "${AGENTS[@]}"; do
        IFS='|' read -r display_name port mcp_name mcp_port <<< "$entry"
        pids=$(lsof -ti :"$port" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill 2>/dev/null || true
            echo "  ✗ $display_name (port $port) — stopped"
        fi
    done
    exit 0
fi

# ── Create agent server script ───────────────────────────────────────
AGENT_SCRIPT="$SCRIPT_DIR/_langgraph_agent.py"
cat > "$AGENT_SCRIPT" << 'PYEOF'
"""Generic LangGraph agent — connects to one MCP server via env vars.

Reads MCP_SERVER_NAME and MCP_SERVER_URL from environment.
Uses GPT-5.2 as the exploration/distillation model.
Sets handle_tool_error=True to gracefully handle MCP tool errors.
"""
import asyncio
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", "mcp-server")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8001/mcp")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-5.2")

def _get_model(state, config):
    """Return the LLM, using configurable.model if provided at runtime."""
    model_name = config.get("configurable", {}).get("model", DEFAULT_MODEL)
    api_base = config.get("configurable", {}).get("api_base", None)
    api_key = config.get("configurable", {}).get("api_key", None)
    kwargs = {"model": model_name}
    if api_base:
        kwargs["base_url"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)

def _create_graph():
    client = MultiServerMCPClient({
        MCP_SERVER_NAME: {"transport": "streamable_http", "url": MCP_SERVER_URL},
    })
    tools = asyncio.run(client.get_tools())
    for tool in tools:
        tool.handle_tool_error = True
    print(f"[{MCP_SERVER_NAME}] Loaded {len(tools)} tools from {MCP_SERVER_URL}")
    return create_react_agent(_get_model, tools)

graph = _create_graph()
PYEOF

# ── Create per-server langgraph configs ──────────────────────────────
CONFIGS_DIR="$SCRIPT_DIR/_agent_configs"
mkdir -p "$CONFIGS_DIR"

for entry in "${AGENTS[@]}"; do
    IFS='|' read -r display_name port mcp_name mcp_port <<< "$entry"
    config_file="$CONFIGS_DIR/$mcp_name.json"
    cat > "$config_file" << JSONEOF
{
  "dependencies": ["."],
  "graphs": {"agent": "$AGENT_SCRIPT:graph"},
  "env": {
    "MCP_SERVER_NAME": "$mcp_name",
    "MCP_SERVER_URL": "http://localhost:$mcp_port/mcp"
  }
}
JSONEOF
done

# ── Find langgraph CLI ───────────────────────────────────────────────
LANGGRAPH_CLI="${LANGGRAPH_CLI:-}"
# Check common locations (command -v is safe under set -e)
for candidate in \
    "$(command -v langgraph 2>/dev/null || true)" \
    "$SCRIPT_DIR/../../../.venv/bin/langgraph" \
    "$(find /workspace -maxdepth 5 -name langgraph -type f 2>/dev/null | head -1 || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        LANGGRAPH_CLI="$candidate"
        break
    fi
done

if [ -z "$LANGGRAPH_CLI" ]; then
    echo "ERROR: langgraph CLI not found. Install it:"
    echo "  pip install langgraph-cli"
    echo ""
    echo "Or if using a separate venv, set LANGGRAPH_CLI env var:"
    echo "  LANGGRAPH_CLI=/path/to/langgraph bash start_agents.sh"
    exit 1
fi

echo "Using langgraph CLI: $LANGGRAPH_CLI"

# ── Start agents ─────────────────────────────────────────────────────
echo ""
echo "Starting LangGraph agents..."
echo "─────────────────────────────────────────"

for entry in "${AGENTS[@]}"; do
    IFS='|' read -r display_name port mcp_name mcp_port <<< "$entry"

    if check_port "$port"; then
        echo "  ✓ $display_name (port $port) — already running"
        continue
    fi

    config_file="$CONFIGS_DIR/$mcp_name.json"
    log_file="/tmp/langgraph_${mcp_name}.log"

    echo "  Starting $display_name on port $port (MCP: localhost:$mcp_port)..."
    (cd "$SCRIPT_DIR" && "$LANGGRAPH_CLI" dev \
        --config "$config_file" \
        --port "$port" \
        --no-browser \
        > "$log_file" 2>&1) &

    sleep 3
done

echo ""
echo "Waiting for agents to start (15s)..."
sleep 15

echo ""
echo "Agent Status:"
echo "─────────────────────────────────────────"
for entry in "${AGENTS[@]}"; do
    IFS='|' read -r display_name port mcp_name mcp_port <<< "$entry"
    if check_port "$port"; then
        echo "  ✓ $display_name → http://localhost:$port"
    else
        echo "  ✗ $display_name — FAILED (check /tmp/langgraph_${mcp_name}.log)"
    fi
done
echo ""
echo "To stop: bash start_agents.sh --stop"
