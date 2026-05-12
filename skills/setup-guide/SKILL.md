---
name: setup-guide
description: "Use when the user wants to set up synthetic data generation for the first time, or when sdg_hub is not yet installed/configured in the current environment."
---

# sdg_hub Setup Guide

You are helping the user set up synthetic data generation for the first time.

## Detection

First, detect the environment by running:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh"
```

## If Nothing is Installed

1. Explain what sdg_hub does: "sdg_hub is a framework for synthetic data generation — it uses composable blocks and YAML-defined flows to build LLM training datasets from seed data."
2. Ask permission: "I can install it for you. This will add the `sdg_hub` Python package to your environment. Want me to proceed?"
3. If yes: install using the detected installer (`uv pip install sdg_hub` or `pip install sdg_hub`)
4. Proceed to configuration

## Configuration

Invoke the `/sdg-setup` command to walk through configuration:
- LLM provider and model selection
- API endpoint and key
- Generation parameters (temperature, concurrency)
- Checkpoint directory

## After Setup

Once configured, hand off to the `data-generation` skill if the user had an original generation request. Otherwise, tell the user:
- "You're all set! You can now use `/sdg-generate` to run data generation."
- Mention `/sdg-flows` to browse available pre-built flows.
