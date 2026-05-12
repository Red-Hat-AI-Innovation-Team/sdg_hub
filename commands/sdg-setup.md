---
description: "Guided first-run configuration for synthetic data generation"
argument-hint: ""
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh:*)"]
---

# sdg-hub Setup

You are helping the user configure synthetic data generation for their workflows.

## Step 1: Detect Environment

Run the detection script to understand the current state:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh"
```

## Step 2: Install if Needed

If `library=missing`:
- Ask the user: "sdg_hub isn't installed. I can install it for you — want me to proceed?"
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

Report success and remind the user they can now use:
- `/sdg-generate` to run data generation
- `/sdg-flows` to browse available flows

## Updating Config

If the user runs `/sdg-setup` again and a config already exists, ask: "You already have a configuration. Do you want to update it or start fresh?"
