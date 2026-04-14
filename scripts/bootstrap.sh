#!/usr/bin/env bash
# bootstrap.sh -- Bootstrap sdg_hub + Claude Code skill for synthetic data generation.
# Usage: curl -fsSL https://raw.githubusercontent.com/Red-Hat-AI-Innovation-Team/sdg_hub/main/scripts/bootstrap.sh | claude --dangerously-skip-permissions
#
# This script is designed to be piped into Claude Code. It:
#   1. Checks prerequisites (Python 3.10+)
#   2. Installs sdg_hub via uv or pip
#   3. Downloads the Claude Code skill for synthetic data generation
#   4. Sets up a workspace directory
#   5. Verifies the installation
#
# Safe to run multiple times (idempotent).

set -euo pipefail

REPO_URL="https://raw.githubusercontent.com/Red-Hat-AI-Innovation-Team/sdg_hub/main"
SKILL_DIR=".claude/skills/synthetic-data-generation"
SKILL_FILES=(
    "SKILL.md"
    "references/block_reference.md"
    "references/flow_patterns.md"
    "references/model_configs.md"
    "references/pre_built_flows.md"
    "references/yaml_schema.md"
)

info() { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
fail() { echo "ERROR: $*" >&2; exit 1; }

# -------------------------------------------------------------------
# 1. Check Python version
# -------------------------------------------------------------------
info "Checking Python version..."

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    fail "Python not found. Please install Python 3.10+ first."
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    fail "Python 3.10+ required, found $PY_VERSION"
fi
info "Found Python $PY_VERSION"

# -------------------------------------------------------------------
# 2. Install sdg_hub
# -------------------------------------------------------------------
info "Installing sdg_hub..."

if command -v uv >/dev/null 2>&1; then
    info "Using uv for installation"
    uv pip install sdg-hub 2>/dev/null || uv pip install sdg-hub
else
    info "Using pip for installation (install uv for faster installs: https://docs.astral.sh/uv/)"
    "$PYTHON" -m pip install --quiet sdg-hub 2>/dev/null || "$PYTHON" -m pip install sdg-hub
fi

# -------------------------------------------------------------------
# 3. Download Claude Code skill
# -------------------------------------------------------------------
info "Setting up Claude Code skill for synthetic data generation..."

mkdir -p "$SKILL_DIR/references"

for file in "${SKILL_FILES[@]}"; do
    target="$SKILL_DIR/$file"
    url="$REPO_URL/.claude/skills/synthetic-data-generation/$file"
    if curl -fsSL "$url" -o "$target" 2>/dev/null; then
        info "  Downloaded $file"
    else
        warn "  Could not download $file (non-critical, skill may still work)"
    fi
done

# -------------------------------------------------------------------
# 4. Set up workspace
# -------------------------------------------------------------------
info "Setting up workspace..."

if [ ! -f "CLAUDE.md" ]; then
    cat > CLAUDE.md << 'CLAUDEMD'
# SDG Hub Workspace

This workspace is configured for synthetic data generation with sdg_hub.

## Quick Start

Describe what synthetic data you need and Claude will help you:

- **QA pairs:** "Generate question-answer pairs from the documents in ./data/"
- **Red-teaming:** "Create adversarial prompts to test my chatbot"
- **RAG evaluation:** "Build an evaluation dataset for my RAG system"
- **Text analysis:** "Extract structured insights from my text corpus"
- **MCP distillation:** "Create tool-use training data from my MCP server"

## Available Skills

The `synthetic-data-generation` skill is installed at `.claude/skills/synthetic-data-generation/`.
It teaches Claude how to use sdg_hub's blocks, flows, and connectors.
CLAUDEMD
    info "Created CLAUDE.md"
else
    info "CLAUDE.md already exists, skipping"
fi

# Create a data directory if it doesn't exist
mkdir -p data
info "Created data/ directory for input files"

# -------------------------------------------------------------------
# 5. Verify installation
# -------------------------------------------------------------------
info "Verifying installation..."

if "$PYTHON" -c "from sdg_hub import FlowRegistry; flows = FlowRegistry.list_flows(); print(f'  Found {len(flows)} pre-built flows')" 2>/dev/null; then
    info "sdg_hub is installed and working"
else
    warn "sdg_hub installed but could not list flows (this may be fine for a fresh install)"
fi

if [ -f "$SKILL_DIR/SKILL.md" ]; then
    info "Claude Code skill is installed"
else
    warn "Claude Code skill files may be incomplete"
fi

# -------------------------------------------------------------------
# Done
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo "  sdg_hub is ready!"
echo "============================================"
echo ""
echo "Next step -- start a Claude session:"
echo ""
echo "  claude"
echo ""
echo "Then describe what data you need:"
echo ""
echo "  > Generate QA pairs from the documents in ./data/"
echo "  > Create a red-team evaluation dataset for my chatbot"
echo "  > Build a RAG evaluation dataset using my knowledge base"
echo ""
echo "Place your input documents (CSV, JSON, TXT, PDF) in the data/ directory."
echo ""
