#!/usr/bin/env bash
# Start MCP servers for the evaluation benchmark example.
#
# Each server runs via supergateway (stdio → streamable HTTP) on a unique port.
# These servers come from the mcp-bench repository (https://github.com/Accenture/mcp-bench).
#
# Prerequisites:
#   1. Clone mcp-bench: git clone https://github.com/Accenture/mcp-bench.git ../mcp-bench
#   2. Install Node.js (for supergateway + Node-based servers)
#   3. Install Python server deps (script handles this)
#
# Usage:
#   bash start_servers.sh          # install deps + start all servers
#   bash start_servers.sh --check  # check which servers are running
#   bash start_servers.sh --stop   # stop all servers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_SERVERS_DIR="${MCP_BENCH_DIR:-$SCRIPT_DIR/../mcp-bench}/mcp_servers"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -d "$MCP_SERVERS_DIR" ]]; then
    echo "ERROR: mcp-bench not found at $MCP_SERVERS_DIR"
    echo "Clone it: git clone https://github.com/Accenture/mcp-bench.git $(dirname $MCP_SERVERS_DIR)"
    exit 1
fi

# ── Server definitions ────────────────────────────────────────────────
# Format: NAME|PORT|CWD|STDIO_CMD|INSTALL_CMD
# All servers are data-dependent (models genuinely need the tools)
SERVERS=(
    "weather-data|8001|weather_mcp|$VENV_PYTHON server.py|$VENV_PYTHON -m pip install -r requirements.txt -q"
    "medical-calculator|8002|medcalc|$VENV_PYTHON medcalc/__main__.py|$VENV_PYTHON -m pip install -e . -q"
    "wikipedia|8003|wikipedia-mcp|$VENV_PYTHON -m wikipedia_mcp|$VENV_PYTHON -m pip install -r requirements.txt -q"
    "car-price|8004|car-price-mcp-main|$VENV_PYTHON server.py|$VENV_PYTHON -m pip install -r requirements.txt -q"
    "reddit|8005|mcp-reddit|$VENV_PYTHON -m mcp_reddit.reddit_fetcher|$VENV_PYTHON -m pip install -e . -q"
    "dex-paprika|8006|dexpaprika-mcp|node src/index.js|npm install -q"
)

declare -A SERVER_NAMES=(
    ["weather-data"]="Weather Data"
    ["medical-calculator"]="Medical Calculator"
    ["wikipedia"]="Wikipedia"
    ["car-price"]="Car Price Evaluator"
    ["reddit"]="Reddit"
    ["dex-paprika"]="DEX Paprika"
)

check_port() {
    local port=$1
    if (echo > /dev/tcp/localhost/$port) 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# ── --check ───────────────────────────────────────────────────────────
if [[ "${1:-}" == "--check" ]]; then
    echo "MCP Server Status:"
    echo "─────────────────────────────────────────"
    for entry in "${SERVERS[@]}"; do
        IFS='|' read -r name port cwd cmd install <<< "$entry"
        bench_name="${SERVER_NAMES[$name]:-$name}"
        if check_port "$port"; then
            echo "  ✓ $bench_name (port $port)"
        else
            echo "  ✗ $bench_name (port $port)"
        fi
    done
    exit 0
fi

# ── --stop ────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping MCP servers..."
    for entry in "${SERVERS[@]}"; do
        IFS='|' read -r name port cwd cmd install <<< "$entry"
        pid=$(lsof -ti :"$port" 2>/dev/null | head -1)
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null
            echo "  ✗ ${SERVER_NAMES[$name]:-$name} (port $port) — stopped"
        fi
    done
    exit 0
fi

# ── Install + Start ──────────────────────────────────────────────────
echo "Installing dependencies..."
for entry in "${SERVERS[@]}"; do
    IFS='|' read -r name port cwd cmd install <<< "$entry"
    server_dir="$MCP_SERVERS_DIR/$cwd"
    [[ ! -d "$server_dir" ]] && continue
    echo "  ${SERVER_NAMES[$name]:-$name}..."
    (cd "$server_dir" && eval "$install") 2>&1 | tail -1 || true
done

echo ""
echo "Starting servers..."
PIDS=()
for entry in "${SERVERS[@]}"; do
    IFS='|' read -r name port cwd cmd install <<< "$entry"
    server_dir="$MCP_SERVERS_DIR/$cwd"
    [[ ! -d "$server_dir" ]] && continue

    if check_port "$port"; then
        echo "  ✓ ${SERVER_NAMES[$name]:-$name} (port $port) — already running"
        continue
    fi

    echo "  Starting ${SERVER_NAMES[$name]:-$name} on port $port..."
    (cd "$server_dir" && npx -y supergateway \
        --stdio "$cmd" --port "$port" \
        --outputTransport streamableHttp --stateful \
        > "/tmp/mcp_${name}.log" 2>&1) &
    PIDS+=($!)
    sleep 2
done

echo ""
echo "Waiting for servers to start..."
sleep 8

echo ""
echo "Server Status:"
echo "─────────────────────────────────────────"
for entry in "${SERVERS[@]}"; do
    IFS='|' read -r name port cwd cmd install <<< "$entry"
    bench_name="${SERVER_NAMES[$name]:-$name}"
    if check_port "$port"; then
        echo "  ✓ $bench_name → http://localhost:$port/mcp"
    else
        echo "  ✗ $bench_name — FAILED (check /tmp/mcp_${name}.log)"
    fi
done
echo ""
echo "To stop: bash start_servers.sh --stop"
