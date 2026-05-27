#!/usr/bin/env bash
# Detect sdg_hub environment: library, installer, config.
#
# Usage: sdg_detect.sh
# Output: key=value pairs, one per line:
#   library=installed|missing
#   installer=uv|pip|none
#   config=found|missing
#
# Config path can be overridden via SDG_HUB_CONFIG env var.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG_PATH="${SDG_HUB_CONFIG:-.sdg-hub/config.json}"

# Check library
if $PYTHON -c "import sdg_hub" 2>/dev/null; then
    echo "library=installed"
else
    echo "library=missing"
fi

# Check installer
if command -v uv > /dev/null 2>&1; then
    echo "installer=uv"
elif command -v pip > /dev/null 2>&1; then
    echo "installer=pip"
else
    echo "installer=none"
fi

# Check config
if [ -f "$CONFIG_PATH" ]; then
    echo "config=found"
else
    echo "config=missing"
fi
