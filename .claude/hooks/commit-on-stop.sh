#!/bin/bash
# Stop hook — auto-commit tracked changes as a checkpoint

if git diff --quiet HEAD 2>/dev/null; then
  exit 0
fi

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
git add -A
git commit -m "session checkpoint: $TIMESTAMP" --no-verify 2>/dev/null || true
