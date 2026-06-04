#!/usr/bin/env bash
# Shared environment setup: resolve the project's virtual-env Python.
# Sourced by other scripts in this directory — not run directly.
#
# Exports: $PYTHON — path to the Python interpreter.
# Resolution order: $VIRTUAL_ENV/bin/python → .venv/bin/python → python3 → python.

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_SCRIPTS_DIR/.." && pwd)"

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -x "$_REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$_REPO_ROOT/.venv/bin/python"
elif command -v python3 > /dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi
