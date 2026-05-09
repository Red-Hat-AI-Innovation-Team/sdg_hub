---
title: Get Started
description: Start generating synthetic data in under 2 minutes with Claude Code and sdg_hub
---

# Get Started

Three commands. That's all it takes to go from zero to generating synthetic data.

## Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Claude Code](https://claude.ai/code) (`npm install -g @anthropic-ai/claude-code`)
- An LLM API key (OpenAI, Anthropic, or [any LiteLLM-supported provider](https://docs.litellm.ai/docs/providers))

## Step 1: Set your API key

Export the key for the LLM provider you want to use:

```bash
export OPENAI_API_KEY="sk-..."
```

<details>
<summary>Using a different provider?</summary>

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Together AI
export TOGETHER_API_KEY="..."

# Groq
export GROQ_API_KEY="..."

# Local model (vLLM / Ollama) -- no key needed
```

</details>

## Step 2: Bootstrap your workspace

```bash
curl -fsSL https://raw.githubusercontent.com/Red-Hat-AI-Innovation-Team/sdg_hub/main/scripts/bootstrap.sh | bash
```

This installs `sdg-hub`, downloads the Claude Code skill, and creates a `data/` directory for your input files.

## Step 3: Generate data

```bash
claude
```

Then tell Claude what you need:

```text
> Generate question-answer pairs from the documents in ./data/
```

Claude will pick the right pipeline, configure it, validate your data, run a dry-run, and produce results -- all from that single prompt. Drop your source files (CSV, JSON, TXT, PDF) into `data/` beforehand and you're set.

## Example prompts to try

| What you type | What happens |
|---------------|-------------|
| *"Generate QA pairs from the documents in ./data/"* | Runs a knowledge-infusion pipeline on your documents |
| *"Create a red-team evaluation dataset for my chatbot"* | Generates adversarial prompts across harm categories |
| *"Build a RAG evaluation dataset using my knowledge base"* | Produces question-context-answer triples for RAG eval |
| *"Extract structured insights from my text corpus"* | Pulls summaries, keywords, entities, and sentiment |

## Want more control?

If you prefer to write Python directly or author custom YAML pipelines, skip Claude Code and head to:

- [Installation](/docs/installation) -- Install sdg_hub and optional dependencies
- [Quick Start](/docs/quickstart) -- Hands-on tutorial using the Python API
- [Core Concepts](/docs/concepts) -- Blocks, flows, and registries explained
