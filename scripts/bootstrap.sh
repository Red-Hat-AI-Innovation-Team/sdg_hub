#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Bootstrap script for SDG Hub — installs the package and clones the repo.

set -euo pipefail

TARGET_DIR="${SDG_HUB_DIR:-$HOME/sdg_hub}"

python3 -c 'import sys; assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version}"'

if command -v uv &>/dev/null; then
  echo "Installing sdg-hub with uv..."
  uv pip install sdg-hub
else
  echo "Installing sdg-hub with pip..."
  pip install sdg-hub
fi

if [ -d "$TARGET_DIR/.git" ]; then
  echo "Repository already exists at $TARGET_DIR, pulling latest..."
  git -C "$TARGET_DIR" pull --ff-only || echo "Warning: pull failed — using existing checkout"
else
  echo "Cloning sdg_hub repository..."
  git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git "$TARGET_DIR"
fi

echo ""
echo "Verifying installation..."
python3 -c "
from sdg_hub import FlowRegistry, BlockRegistry
FlowRegistry.discover_flows()
BlockRegistry.discover_blocks()
flows = FlowRegistry.list_flows()
blocks = BlockRegistry.list_blocks()
print(f'SDG Hub installed successfully: {len(flows)} flows, {len(blocks)} blocks available')
"

echo ""
echo "Setup complete! Repository cloned to: $TARGET_DIR"
echo "To start generating data, run:"
echo "  cd $TARGET_DIR && claude"
