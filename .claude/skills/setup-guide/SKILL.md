---
name: setup-guide
description: "Use when the user wants to set up synthetic data generation for the first time, or when sdg_hub is not yet installed/configured in the current environment."
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_flows.sh:*)"]
---

# sdg_hub Setup Guide

You are helping the user set up synthetic data generation.

## Step 1: Detect Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh"
```

## Step 2: Install if Needed

If `library=missing`:
- Explain: "sdg_hub is a framework for synthetic data generation — it uses composable blocks and YAML-defined flows to build LLM training datasets from seed data."
- Ask permission: "I can install it for you. Want me to proceed?"
- If yes and `installer=uv`: run `uv pip install sdg_hub`
- If yes and `installer=pip`: run `pip install sdg_hub`
- If `installer=none`: tell the user they need Python and pip/uv installed first

## Step 3: Collect Configuration

Ask these questions **one at a time**:

1. **Model**: "Which LLM model do you want to use for generation?" — e.g., `openai/gpt-4o-mini`, `meta-llama/Llama-3.3-70B-Instruct`, `anthropic/claude-sonnet-4-20250514`
2. **API endpoint**: "What's your model endpoint URL?" — e.g., `http://localhost:8000/v1` for vLLM, or leave empty for cloud provider defaults
3. **API key**: "What's your API key?" — required for cloud providers

## Step 4: Generation Settings

Ask:
1. **Temperature**: "What temperature for generation?" (default: 0.7)
2. **Max concurrency**: "How many parallel LLM requests?" (default: 5) — higher is faster but may hit rate limits
3. **Checkpoint directory**: "Where should generation checkpoints be saved?" (default: `./checkpoints`) — allows resuming interrupted runs

## Step 5: Save Config

Write the config to `.sdg-hub/config.json`:

```json
{
  "model": "<model>",
  "api_key": "<api_key>",
  "api_base": "<endpoint>",
  "temperature": 0.7,
  "max_concurrency": 5,
  "checkpoint_dir": "./checkpoints"
}
```

Add `.sdg-hub/` to `.gitignore` if not already present.

## Step 6: Verify

List available flows to confirm the installation works:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_flows.sh" list
```

Report success and remind the user they can now use the `data-generation` skill to run generation, or the `flow-browser` skill to browse available flows.

## Updating Config

If this skill is invoked again and a config already exists, ask: "You already have a configuration. Do you want to update it or start fresh?"
