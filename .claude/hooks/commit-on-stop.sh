#!/bin/bash
# Stop hook — auto-commit tracked changes as a checkpoint

if git diff --quiet HEAD 2>/dev/null; then
  exit 0
fi

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
git add -u
git commit -m "chore: session checkpoint $TIMESTAMP" 2>&1 || echo "WARNING: checkpoint commit failed" >&2
