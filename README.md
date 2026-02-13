# `sdg_hub`: Synthetic Data Generation Toolkit

[![Build](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/pypi.yml/badge.svg?branch=main)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/pypi.yml)
[![Release](https://img.shields.io/github/v/release/Red-Hat-AI-Innovation-Team/sdg_hub)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/releases)
[![License](https://img.shields.io/github/license/Red-Hat-AI-Innovation-Team/sdg_hub)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/main/LICENSE)
[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub/graph/badge.svg?token=SP75BCXWO2)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Red-Hat-AI-Innovation-Team/sdg_hub)

<p align="center">
  <img src="docs/assets/sdg-hub-cover.png" alt="SDG Hub Cover" width="400">
</p>

A modular Python framework for building synthetic data generation pipelines using composable blocks and flows. Transform datasets through **building-block composition** - mix and match LLM-powered and traditional processing blocks to create sophisticated data generation workflows.

**📖 Full documentation available at: [DeepWiki](https://deepwiki.com/Red-Hat-AI-Innovation-Team/sdg_hub)**

## ✨ Key Features

**🔧 Modular Composability** - Mix and match blocks like Lego pieces. Build simple transformations or complex multi-stage pipelines with YAML-configured flows.

**⚡ Async Performance** - High-throughput LLM processing with built-in error handling.

**🛡️ Built-in Validation** - Pydantic-based type safety ensures your configurations and data are correct before execution.

**🔍 Auto-Discovery** - Automatic block and flow registration. No manual imports or complex setup.

**📊 Rich Monitoring** - Detailed logging with progress bars and execution summaries.

**📋 Dataset Schema Discovery** - Instantly discover required data formats. Get empty datasets with correct schema for easy validation and data preparation.

**🧩 Easily Extensible** - Create custom blocks with simple inheritance. Rich logging and monitoring built-in.


## 📦 Installation

Recommended: Install uv  — see https://docs.astral.sh/uv/getting-started/installation/

```bash
# Production
uv pip install sdg-hub

# Development
git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
cd sdg_hub
uv pip install .[dev]
# or: uv sync --extra dev
```

### Optional Dependencies
```bash
# For vLLM support
uv pip install sdg-hub[vllm]

# For examples
uv pip install sdg-hub[examples]
```

## 🚀 Quick Start

### Core Concepts

**Blocks** are composable units that transform datasets - think of them as data processing Lego pieces. Each block performs a specific task: LLM chat, text parsing, evaluation, or transformation.

**Flows** orchestrate multiple blocks into complete pipelines defined in YAML. Chain blocks together to create complex data generation workflows with validation and parameter management.

```python
# Simple concept: Blocks transform data, Flows chain blocks together
dataset → Block₁ → Block₂ → Block₃ → enriched_dataset
```

### Try it out!

#### Flow Discovery
```python
from sdg_hub import FlowRegistry, Flow

# Auto-discover all available flows (no setup needed!)
FlowRegistry.discover_flows()

# List available flows
flows = FlowRegistry.list_flows()
print(f"Available flows: {flows}")

# Search for specific types
qa_flows = FlowRegistry.search_flows(tag="question-generation")
print(f"QA flows: {qa_flows}")
```

Each flow has a **unique, human-readable ID** automatically generated from its name. These IDs provide a convenient shorthand for referencing flows:

```python
# Every flow gets a deterministic ID 
# Same flow name always generates the same ID
flow_id = "small-rock-799" 

# Use ID to reference the flow
flow_path = FlowRegistry.get_flow_path(flow_id)
flow = Flow.from_yaml(flow_path)
```

#### Discovering Models and Configuring them
```python
# Discover recommended models
default_model = flow.get_default_model()
recommendations = flow.get_model_recommendations()

# Configure model settings at runtime
# This assumes you have a hosted vLLM instance of meta-llama/Llama-3.3-70B-Instruct running at http://localhost:8000/v1
flow.set_model_config(
    model=f"hosted_vllm/{default_model}",
    api_base="http://localhost:8000/v1",
    api_key="your_key",
)
```
#### Discover dataset requirements and create your dataset
```python
# First, discover what data the flow needs
# Get an empty dataset with the exact schema needed
schema_dataset = flow.get_dataset_schema()  # Get empty dataset with correct schema
print(f"Required columns: {schema_dataset.column_names}")
print(f"Schema: {schema_dataset.features}")

# Option 1: Add data directly to the schema dataset
dataset = schema_dataset.add_item({
    'document': 'Your document text here...',
    'document_outline': '1. Topic A; 2. Topic B; 3. Topic C',
    'domain': 'Computer Science',
    'icl_document': 'Example document for in-context learning...',
    'icl_query_1': 'Example question 1?',
    'icl_response_1': 'Example answer 1',
    'icl_query_2': 'Example question 2?', 
    'icl_response_2': 'Example answer 2',
    'icl_query_3': 'Example question 3?',
    'icl_response_3': 'Example answer 3'
})

# Option 2: Create your own dataset and validate the schema
my_dataset = Dataset.from_dict(my_data_dict)
if my_dataset.features == schema_dataset.features:
    print("✅ Schema matches - ready to generate!")
    dataset = my_dataset
else:
    print("❌ Schema mismatch - check your columns")

# Option 3: Get raw requirements for detailed inspection
requirements = flow.get_dataset_requirements()
if requirements:
    print(f"Required: {requirements.required_columns}")
    print(f"Optional: {requirements.optional_columns}")
    print(f"Min samples: {requirements.min_samples}")
```

#### Dry Run and Generate
```python
# Quick Testing with Dry Run
dry_result = flow.dry_run(dataset, sample_size=1)
print(f"Dry run completed in {dry_result['execution_time_seconds']:.2f}s")
print(f"Output columns: {dry_result['final_dataset']['columns']}")

# Generate high-quality QA pairs
result = flow.generate(dataset)

# Access generated content
questions = result['question']
answers = result['response']
faithfulness_scores = result['faithfulness_judgment']
relevancy_scores = result['relevancy_score']
```


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

---

Built with ❤️ by the Red Hat AI Innovation Team
