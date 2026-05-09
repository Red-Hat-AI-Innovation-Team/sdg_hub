#!/bin/bash
set -e

export HOME=/home/default

mkdir -p /home/default/.claude
cat > /home/default/.claude/mcp.json << 'MCPEOF'
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest", "--headless"]
    }
  }
}
MCPEOF

multica config set server_url "${MULTICA_SERVER_URL:-http://multica-backend:8080}"
multica config set app_url "${MULTICA_APP_URL:-http://localhost:3000}"

if [ -n "${MULTICA_TOKEN}" ] && [ "${MULTICA_TOKEN}" != "placeholder-generate-from-ui" ]; then
    multica login --token "${MULTICA_TOKEN}"
else
    echo "WARNING: MULTICA_TOKEN not set. Daemon will not connect to server."
    echo "Generate a token from the Multica UI: Settings → Runtimes → Add Runtime"
    echo "Then update the secret: oc -n sdg-hub-agents patch secret multica-secrets --type merge -p '{\"stringData\":{\"MULTICA_TOKEN\":\"mdt_YOUR_TOKEN\"}}'"
    sleep infinity
fi

exec multica daemon start --foreground
