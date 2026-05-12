# Get Started

Get up and running with SDG Hub in minutes using [Claude Code](https://claude.ai/claude-code).

## Prerequisites

- **Python 3.10+**
- **Claude Code** installed ([instructions](https://docs.anthropic.com/en/docs/claude-code/overview))

!!! tip "Prefer manual installation?"
    See the full [Installation](installation.md) guide for pip/uv setup without Claude Code.

## Step 1: Bootstrap SDG Hub

Download the bootstrap script and review it before running:

```bash
curl -fsSL https://raw.githubusercontent.com/Red-Hat-AI-Innovation-Team/sdg_hub/main/scripts/bootstrap.sh -o bootstrap.sh
less bootstrap.sh   # review the script
bash bootstrap.sh
```

### What this does

1. Installs the `sdg-hub` Python package (via `uv` or `pip`)
2. Clones the SDG Hub repository, which includes the `synthetic-data-generation` Claude Code skill
3. Verifies the installation by discovering available blocks and flows

## Step 2: Start generating data

Navigate to the cloned repository and launch Claude Code:

```bash
cd ~/sdg_hub && claude
```

Claude Code automatically discovers the `synthetic-data-generation` skill bundled in the repository. Describe what data you need, and it will select the right blocks and flows.

### Example prompts

| What you need | Example prompt |
|---|---|
| Question-answer pairs | "Generate 50 QA pairs from my PDF at `./data/manual.pdf`" |
| Instruction tuning data | "Create instruction-response pairs for customer support using `./data/tickets.csv`" |
| Red-team evaluation set | "Build a red-team dataset to test my chatbot's safety guardrails" |
| RAG evaluation | "Generate evaluation questions from the documents in `./data/knowledge_base/`" |
| Multi-step pipeline | "Create a flow that extracts text, generates questions, then filters by difficulty" |

## What's next

- [Quick Start](quickstart.md) -- build a pipeline step by step in Python
- [Core Concepts](concepts.md) -- understand blocks, flows, and registries
- [Built-in Flows](flows/built-in-flows.md) -- browse ready-to-use pipelines
- [Custom Flows](flows/custom-flows.md) -- author your own YAML flow
