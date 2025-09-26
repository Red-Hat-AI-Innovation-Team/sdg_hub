#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# Script to run ruff formatting and linting
set -euo pipefail

# Default to 'fix' if no argument provided
ACTION=${1:-fix}

echo "Running ruff with action: $ACTION"

if [[ "$ACTION" == "fix" ]]; then
    echo "Formatting and fixing code with ruff..."
    ruff format src/ tests/
    ruff check --fix src/ tests/
elif [[ "$ACTION" == "check" ]]; then
    echo "Checking code with ruff..."
    ruff format --check src/ tests/
    ruff check src/ tests/
else
    echo "Running ruff with custom arguments: $*"
    ruff "$@"
fi