---
title: Get Started
description: Start generating synthetic data in under 2 minutes with Claude Code and sdg_hub
---

# Get Started

Generate synthetic data in minutes using Claude Code and sdg_hub. This guide walks you through a two-command setup that installs everything you need and launches an interactive AI-assisted data generation session.

## Prerequisites

Before you begin, make sure you have:

- **Python 3.10+** installed ([python.org](https://www.python.org/downloads/))
- **An LLM API key** from any supported provider (OpenAI, Anthropic, Google, Azure, Together, Groq, or a local model via vLLM/Ollama)
- **Claude Code** installed ([claude.ai/code](https://claude.ai/code)):

```bash
npm install -g @anthropic-ai/claude-code
```

## Step 1: Bootstrap your environment

Run this single command to install sdg_hub, download the Claude Code skill, and set up your workspace:

```bash
curl -fsSL https://raw.githubusercontent.com/Red-Hat-AI-Innovation-Team/sdg_hub/main/scripts/bootstrap.sh \
  | claude --dangerously-skip-permissions
```

This command:

1. Checks that Python 3.10+ is available
2. Installs `sdg-hub` via `uv` (or `pip` as fallback)
3. Downloads the synthetic data generation skill for Claude Code
4. Creates a workspace with a `data/` directory for your input files
5. Verifies the installation

## Step 2: Start generating data

Place your input documents in the `data/` directory, then launch Claude:

```bash
claude
```

Describe what you need in plain language. Claude will select the right pipeline, configure it, validate your data, and run generation:

```
> Generate question-answer pairs from the documents in ./data/
```

```
> Create a red-team evaluation dataset for my chatbot
```

```
> Build a RAG evaluation dataset using my knowledge base
```

```
> Extract structured insights from the text files in ./data/
```

## What happens next

When you describe your data generation task, Claude will:

1. **Scan your workspace** for input data files (CSV, JSON, Parquet, TXT, PDF)
2. **Recommend a pipeline** from 13+ pre-built flows or help you create a custom one
3. **Configure the LLM** by checking your environment for API keys or asking which provider to use
4. **Validate your data** against the pipeline's requirements
5. **Run a dry run** with a small sample to verify everything works
6. **Generate your dataset** with checkpointing for large runs
7. **Save the results** in your preferred format (Parquet, CSV, JSONL)

## Available pipelines

sdg_hub includes pre-built flows for common data generation tasks:

| Category | Pipelines |
|----------|-----------|
| **Knowledge Infusion** | QA pair generation from documents (English, Spanish, Japanese) with multiple summarization strategies |
| **Text Analysis** | Structured insights extraction (summaries, keywords, entities, sentiment) |
| **Red Teaming** | Adversarial prompt generation across harm categories |
| **Evaluation** | RAG evaluation datasets, agent tool-use benchmarks |
| **Agentic** | MCP server distillation for tool-use training data |

## Supported LLM providers

sdg_hub uses LiteLLM and supports 100+ model providers out of the box. Set your API key as an environment variable and it will be detected automatically:

```bash
# Use any of these (or many more)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export TOGETHER_API_KEY="..."
export GROQ_API_KEY="..."
```

Local models via vLLM or Ollama are also supported. See the [installation guide](/docs/installation) for details.

## Manual setup

If you prefer to set things up manually without Claude Code, follow these guides:

- [Installation](/docs/installation) -- Install sdg_hub and optional dependencies
- [Quick Start](/docs/quickstart) -- Step-by-step tutorial using the Python API
- [Core Concepts](/docs/concepts) -- Understanding blocks, flows, and registries
