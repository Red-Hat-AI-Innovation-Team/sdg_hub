# Get Started

Get up and running with SDG Hub in two steps. By the end, you will have a fully bootstrapped environment ready to generate synthetic data from your documents.

## Prerequisites

- **Python 3.10+** installed
- **Claude Code** installed ([install guide](https://docs.anthropic.com/en/docs/claude-code/overview))
- An LLM API key (OpenAI, Anthropic, or any [supported provider](https://docs.litellm.ai/docs/providers))

## Step 1: Bootstrap SDG Hub

Run this single command to install SDG Hub, its dependencies, and all required Claude Code skills:

```bash
curl -fsSL https://raw.githubusercontent.com/Red-Hat-AI-Innovation-Team/sdg_hub/main/scripts/bootstrap.sh | claude --dangerously-skip-permissions
```

**What this does:**

1. Installs `sdg-hub` and its Python dependencies via `uv` (or `pip` as fallback)
2. Clones the SDG Hub repository so Claude has access to flow definitions and prompt templates
3. Installs the `synthetic-data-generation` skill into your Claude Code environment
4. Verifies the installation by discovering all available blocks and flows

??? note "Prefer a manual install?"

    If you'd rather install manually, see the [Installation](installation.md) guide. Then add the skill yourself:

    ```bash
    git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
    cd sdg_hub
    uv pip install .
    ```

    The Claude Code skill at `.claude/skills/synthetic-data-generation/` is included in the repo and will be picked up automatically when you run Claude from the project directory.

## Step 2: Generate Data

Launch a Claude Code session with SDG Hub fully initialized:

```bash
cd sdg_hub
claude
```

Claude already has the `synthetic-data-generation` skill loaded from the repository's `.claude/` directory. Just describe what you need in plain language:

> *"I have a folder of PDF documents in `./data/docs/`. Generate a question-answer dataset from them using the QA generation flow. Use `openai/gpt-4o-mini` as the model."*

Claude will:

- Discover the right pre-built flow for your task
- Load and validate your input data
- Configure the LLM provider
- Run a dry-run to catch errors early
- Execute the full pipeline and save results

### Example Prompts

Here are some things you can ask Claude to do once the session is running:

| What you want | Example prompt |
|---|---|
| Generate QA pairs | *"Generate question-answer pairs from the documents in `./data/`. Save as parquet."* |
| List available flows | *"What pre-built flows are available? Show me the ones tagged for QA generation."* |
| Build a custom pipeline | *"Create a YAML flow that takes a CSV of product descriptions, generates marketing copy, and filters out low-quality results."* |
| Use a local model | *"Run the text-analysis flow using my local vLLM server at `http://localhost:8000/v1` with `meta-llama/Llama-3.3-70B-Instruct`."* |
| Inspect a flow | *"Show me what blocks are in the `knowledge-base-qa` flow and what columns it needs."* |

## What's Next

- [Quick Start](quickstart.md) -- step-by-step walkthrough of the Python API
- [Core Concepts](concepts.md) -- understand blocks, flows, and registries
- [Built-in Flows](flows/built-in-flows.md) -- browse the full catalog of pre-built pipelines
- [Custom Flows](flows/custom-flows.md) -- author your own YAML flow definitions
