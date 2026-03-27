# InstructLab Knowledge Q&A Generation — Example

Automatically generate InstructLab-compatible `qna.yaml` training data from documents.

## Quick Start

```bash
# From repo root
uv pip install .[dev]

# Set your API key
export OPENAI_API_KEY="sk-..."

# Run the notebook
jupyter notebook examples/knowledge_tuning/instructlab_qna/demo.ipynb
```

## What It Does

Takes document chunks + taxonomy metadata → produces submission-ready InstructLab files:

```
input:   2 document chunks (biology, culinary arts)
output:  2 qna.yaml files with 5+ Q&A pairs each
```

## Files

| File | Description |
|------|-------------|
| `demo.ipynb` | Step-by-step tutorial notebook |
| `output/` | Generated `qna.yaml` and `attribution.txt` files (created by notebook) |

## Prerequisites

1. **Python 3.10+** with `uv`
2. **An API key** for any OpenAI-compatible LLM (GPT, Granite, Llama via vLLM, etc.)

## Configuration

Set `SDG_MODEL` to use a different model:

```bash
export SDG_MODEL="ibm/granite-3.3-8b-instruct"
export OPENAI_API_KEY="your-granite-key"
export OPENAI_API_BASE="https://your-endpoint/v1"
```
