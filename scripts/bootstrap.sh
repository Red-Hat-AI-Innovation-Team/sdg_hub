#!/usr/bin/env bash
# bootstrap.sh -- Bootstrap sdg_hub + Claude Code skill for synthetic data generation.
# Usage: curl -fsSL https://raw.githubusercontent.com/Red-Hat-AI-Innovation-Team/sdg_hub/main/scripts/bootstrap.sh | bash
#
# This script:
#   1. Checks prerequisites (Python 3.10+)
#   2. Installs sdg_hub via uv or pip
#   3. Downloads the Claude Code skill for synthetic data generation
#   4. Sets up a workspace with a data/ directory
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
    fail "Python not found. Please install Python 3.10+ first: https://www.python.org/downloads/"
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    fail "Python 3.10+ required, found $PY_VERSION. Update at https://www.python.org/downloads/"
fi
info "Found Python $PY_VERSION"

# -------------------------------------------------------------------
# 2. Install sdg_hub
# -------------------------------------------------------------------
info "Installing sdg_hub..."

if command -v uv >/dev/null 2>&1; then
    uv pip install sdg-hub 2>/dev/null || uv pip install sdg-hub
else
    "$PYTHON" -m pip install --quiet sdg-hub 2>/dev/null || "$PYTHON" -m pip install sdg-hub
fi

# -------------------------------------------------------------------
# 3. Download Claude Code skill
# -------------------------------------------------------------------
info "Downloading Claude Code skill..."

mkdir -p "$SKILL_DIR/references"

for file in "${SKILL_FILES[@]}"; do
    target="$SKILL_DIR/$file"
    url="$REPO_URL/.claude/skills/synthetic-data-generation/$file"
    if curl -fsSL "$url" -o "$target" 2>/dev/null; then
        info "  Downloaded $file"
    else
        warn "  Could not download $file (non-critical)"
    fi
done

# -------------------------------------------------------------------
# 4. Set up workspace
# -------------------------------------------------------------------
mkdir -p data

if [ ! -f "CLAUDE.md" ]; then
    cat > CLAUDE.md << 'CLAUDEMD'
# SDG Hub Workspace

This workspace uses sdg_hub for synthetic data generation.
Describe what data you need and Claude will select and run the right pipeline.

Put your input files (CSV, JSON, TXT, PDF) in the `data/` directory.
CLAUDEMD
fi

# -------------------------------------------------------------------
# Done
# -------------------------------------------------------------------
echo ""
echo "  sdg_hub is ready!"
echo ""
echo "  Next: run 'claude' and tell it what data you need."
echo ""
