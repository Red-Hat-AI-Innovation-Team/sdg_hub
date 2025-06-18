# Quick Start Guide

This guide will help you get started with SDG Hub in just a few minutes.

## Prerequisites

- Python 3.8+
- SDG Hub installed (see [Installation](installation.md))

## Basic Usage

### Generate with an Existing Flow

You can invoke any built-in flow using `run_flow`:

```python
from sdg_hub.flow_runner import run_flow

run_flow(
    ds_path="path/to/dataset.json",
    save_path="path/to/output.json",
    endpoint="https://api.openai.com/v1",
    flow_path="path/to/flow.yaml",
    checkpoint_dir="path/to/checkpoints",
    batch_size=8,
    num_workers=32,
    save_freq=2,
)
```

### Available Built-in Flows

SDG Hub comes with several pre-built flows you can use immediately:

#### 🔎 Knowledge Flows

| Flow Name | Description |
|-----------|-------------|
| `flows/generation/knowledge/synth_knowledge.yaml` | Produces document-grounded questions and answers for factual memorization |
| `flows/generation/knowledge/synth_knowledge1.5.yaml` | Improved version that builds intermediate representations for better recall |

#### 🧠 Skills Flows

| Flow Name | Description |
|-----------|-------------|
| `flows/generation/skills/synth_skills.yaml` | Freeform skills QA generation (eg: "Create a new github issue to add type hints") |
| `flows/generation/skills/synth_grounded_skills.yaml` | Domain-specific skill generation (eg: "From the given conversation create a table for feature requests") |
| `flows/generation/skills/improve_responses.yaml` | Uses planning and critique-based refinement to improve generated answers |

## Your First Flow

Let's create a simple flow that generates questions and answers from a document:

### 1. Prepare Your Data

Create a JSON file with your input data:

```json
[
    {
        "document": "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre Museum."
    }
]
```

### 2. Use a Built-in Flow

```python
from sdg_hub.flow_runner import run_flow

# Generate knowledge-based Q&A pairs
run_flow(
    ds_path="my_documents.json",
    save_path="generated_qa.json",
    endpoint="https://api.openai.com/v1",
    flow_path="flows/generation/knowledge/synth_knowledge.yaml",
    checkpoint_dir="./checkpoints",
    batch_size=4,
    num_workers=8,
    save_freq=1,
)
```

### 3. View Results

The generated Q&A pairs will be saved to `generated_qa.json`:

```json
[
    {
        "document": "The capital of France is Paris...",
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris."
    }
]
```

## Monitoring Your Flows

SDG Hub provides configurable logging to help you monitor execution:

```bash
# Enable verbose logging to see dataset metrics
SDG_HUB_LOG_LEVEL=verbose python your_script.py
```

This will show rich tables with dataset information as blocks execute. See the [Configuration Guide](configuration.md#logging-configuration) for all logging options.

## Next Steps

- Configure [Logging and Environment Variables](configuration.md) for better monitoring
- Explore the [Architecture](architecture.md) to understand how SDG Hub works
- Learn about [Blocks](blocks.md) to create custom processing units
- Check out [Examples](examples.md) for more complex use cases
- Read about [Prompts](prompts.md) to customize LLM behavior

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ds_path` | Path to input dataset | Required |
| `save_path` | Path to save output | Required |
| `endpoint` | LLM API endpoint | Required |
| `flow_path` | Path to flow YAML file | Required |
| `batch_size` | Number of items to process in batch | 8 |
| `num_workers` | Number of parallel workers | 32 |
| `checkpoint_dir` | Directory for checkpoints | None |
| `save_freq` | Save frequency (every N batches) | 2 |