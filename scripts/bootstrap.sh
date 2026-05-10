#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Bootstrap script for SDG Hub — installs the package, clones the repo, and
# verifies that blocks and flows are discoverable.

set -euo pipefail

echo "==> Bootstrapping SDG Hub..."

# --- 1. Install sdg-hub --------------------------------------------------- #
if command -v uv &>/dev/null; then
  echo "==> Installing sdg-hub via uv..."
  uv pip install sdg-hub
else
  echo "==> uv not found, falling back to pip..."
  pip install sdg-hub
fi

# --- 2. Clone the repository for flows and skills ------------------------- #
REPO_URL="https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git"
TARGET_DIR="sdg_hub"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "==> Repository already cloned at ./$TARGET_DIR — pulling latest..."
  git -C "$TARGET_DIR" pull --ff-only || true
else
  echo "==> Cloning SDG Hub repository..."
  git clone "$REPO_URL" "$TARGET_DIR"
fi

# --- 3. Verify installation ----------------------------------------------- #
echo "==> Verifying installation..."
python3 -c "
from sdg_hub import FlowRegistry, BlockRegistry
FlowRegistry.discover_flows()
BlockRegistry.discover_blocks()
flows = FlowRegistry.list_flows()
blocks = BlockRegistry.list_blocks()
print(f'  Flows available:  {len(flows)}')
print(f'  Blocks available: {len(blocks)}')
"

echo ""
echo "==> SDG Hub is ready!"
echo "    cd $TARGET_DIR && claude"
echo ""
echo "    Then describe what data you need — Claude has the"
echo "    synthetic-data-generation skill loaded automatically."
