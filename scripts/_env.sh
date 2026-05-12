#!/usr/bin/env bash
# Shared environment setup: resolve the project's virtual-env Python.
# Sourced by other scripts in this directory — not run directly.

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_SCRIPTS_DIR/.." && pwd)"

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -x "$_REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$_REPO_ROOT/.venv/bin/python"
else
    echo "ERROR: No virtual environment found at $_REPO_ROOT/.venv" >&2
    echo "Run: cd $_REPO_ROOT && uv sync --extra dev" >&2
    exit 1
fi
