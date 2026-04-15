#!/usr/bin/env bash
# Start MCP servers for the evaluation benchmark example.
#
# Each Python server runs natively via FastMCP with streamable-http transport.
# No supergateway or Node.js bridge needed — servers serve HTTP directly.
#
# The one Node.js server (DEX Paprika) uses supergateway as a fallback.
#
# Prerequisites:
#   1. Clone mcp-bench: git clone https://github.com/Accenture/mcp-bench.git ../mcp-bench
#   2. Install Python server deps (script handles this)
#   3. For DEX Paprika only: Node.js + npm
#
# Usage:
#   bash start_servers.sh          # install deps + start all servers
#   bash start_servers.sh --check  # check which servers are running
#   bash start_servers.sh --stop   # stop all servers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_SERVERS_DIR="${MCP_BENCH_DIR:-$SCRIPT_DIR/../../mcp-bench}/mcp_servers"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -d "$MCP_SERVERS_DIR" ]]; then
    echo "ERROR: mcp-bench not found at $MCP_SERVERS_DIR"
    echo "Clone it: git clone https://github.com/Accenture/mcp-bench.git $(dirname "$MCP_SERVERS_DIR")"
    exit 1
fi

# ── Server definitions ────────────────────────────────────────────────
# Format: NAME|PORT|CWD|PYTHON_LAUNCH_CMD|INSTALL_CMD|DISPLAY_NAME
#
# PYTHON_LAUNCH_CMD: a Python snippet that imports the server's FastMCP
# instance and runs it with streamable-http transport on the given port.
# The literal PORT is replaced with the actual port number at launch time.
SERVERS=(
    "weather-data|8001|weather_mcp|from server import mcp; mcp.settings.host='0.0.0.0'; mcp.settings.port=PORT; mcp.run(transport='streamable-http')|$VENV_PYTHON -m pip install -r requirements.txt -q|Weather Data"
    "medical-calculator|8002|medcalc|from medcalc.__main__ import mcp; mcp.settings.host='0.0.0.0'; mcp.settings.port=PORT; mcp.run(transport='streamable-http')|$VENV_PYTHON -m pip install -e . -q|Medical Calculator"
    "wikipedia|8003|wikipedia-mcp|from wikipedia_mcp.server import create_server; s=create_server(); s.settings.host='0.0.0.0'; s.settings.port=PORT; s.run(transport='streamable-http')|$VENV_PYTHON -m pip install -r requirements.txt -q|Wikipedia"
    "car-price|8004|car-price-mcp-main|from server import mcp; mcp.settings.host='0.0.0.0'; mcp.settings.port=PORT; mcp.run(transport='streamable-http')|$VENV_PYTHON -m pip install -r requirements.txt -q|Car Price Evaluator"
    "reddit|8005|mcp-reddit|from mcp_reddit.reddit_fetcher import mcp; mcp.settings.host='0.0.0.0'; mcp.settings.port=PORT; mcp.run(transport='streamable-http')|$VENV_PYTHON -m pip install -e . -q|Reddit"
    "dex-paprika|8006|dexpaprika-mcp|NODE|npm install -q|DEX Paprika"
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
        IFS='|' read -r name port cwd cmd install display_name <<< "$entry"
        if check_port "$port"; then
            echo "  ✓ ${display_name:-$name} (port $port)"
        else
            echo "  ✗ ${display_name:-$name} (port $port)"
        fi
    done
    exit 0
fi

# ── --stop ────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping MCP servers..."
    for entry in "${SERVERS[@]}"; do
        IFS='|' read -r name port cwd cmd install display_name <<< "$entry"
        pids=$(lsof -ti :"$port" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill 2>/dev/null || true
            echo "  ✗ ${display_name:-$name} (port $port) — stopped"
        fi
    done
    exit 0
fi

# ── Install + Start ──────────────────────────────────────────────────
echo "Installing dependencies..."
for entry in "${SERVERS[@]}"; do
    IFS='|' read -r name port cwd cmd install display_name <<< "$entry"
    server_dir="$MCP_SERVERS_DIR/$cwd"
    [[ ! -d "$server_dir" ]] && continue
    echo "  ${display_name:-$name}..."
    (cd "$server_dir" && eval "$install") 2>&1 | tail -1 || true
done

echo ""
echo "Starting servers..."
for entry in "${SERVERS[@]}"; do
    IFS='|' read -r name port cwd cmd install display_name <<< "$entry"
    server_dir="$MCP_SERVERS_DIR/$cwd"
    [[ ! -d "$server_dir" ]] && continue

    if check_port "$port"; then
        echo "  ✓ ${display_name:-$name} (port $port) — already running"
        continue
    fi

    if [[ "$cmd" == "NODE" ]]; then
        # DEX Paprika: Node.js server — use supergateway as fallback
        echo "  Starting ${display_name:-$name} on port $port (via supergateway)..."
        (cd "$server_dir" && npx -y supergateway \
            --stdio "node src/index.js" --port "$port" \
            --outputTransport streamableHttp \
            > "/tmp/mcp_${name}.log" 2>&1) &
    else
        # Python servers: run natively via FastMCP
        launch_cmd="${cmd//PORT/$port}"
        echo "  Starting ${display_name:-$name} on port $port..."
        (cd "$server_dir" && $VENV_PYTHON -c "$launch_cmd" \
            > "/tmp/mcp_${name}.log" 2>&1) &
    fi

    sleep 2
done

echo ""
echo "Waiting for servers to start..."
sleep 5

echo ""
echo "Server Status:"
echo "─────────────────────────────────────────"
for entry in "${SERVERS[@]}"; do
    IFS='|' read -r name port cwd cmd install display_name <<< "$entry"
    if check_port "$port"; then
        echo "  ✓ ${display_name:-$name} → http://localhost:$port/mcp"
    else
        echo "  ✗ ${display_name:-$name} — FAILED (check /tmp/mcp_${name}.log)"
    fi
done
echo ""
echo "To stop: bash start_servers.sh --stop"
