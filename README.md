# `sdg_hub`: Synthetic Data Generation Toolkit

[![Build](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/pypi.yaml/badge.svg?branch=main)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/pypi.yaml)
[![Release](https://img.shields.io/github/v/release/Red-Hat-AI-Innovation-Team/sdg_hub)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/releases)
[![License](https://img.shields.io/github/license/Red-Hat-AI-Innovation-Team/sdg_hub)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/main/LICENSE)
[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub/graph/badge.svg?token=SP75BCXWO2)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub)



A modular Python framework for building synthetic data generation pipelines using composable blocks and flows. Transform datasets through **building-block composition** - mix and match LLM-powered and traditional processing blocks to create sophisticated data generation workflows.

## 🧱 Core Concepts

**Blocks** are composable units that transform datasets - think of them as data processing Lego pieces. Each block performs a specific task: LLM chat, text parsing, evaluation, or transformation.

**Flows** orchestrate multiple blocks into complete pipelines defined in YAML. Chain blocks together to create complex data generation workflows with validation and parameter management.

```python
# Simple concept: Blocks transform data, Flows chain blocks together
dataset → Block₁ → Block₂ → Block₃ → enriched_dataset
```

## ✨ Key Features

**🔧 Modular Composability** - Mix and match blocks like Lego pieces. Build simple transformations or complex multi-stage pipelines with YAML-configured flows.

**⚡ Async Performance** - High-throughput LLM processing with built-in error handling.

**🛡️ Built-in Validation** - Pydantic-based type safety ensures your configurations and data are correct before execution.

**🔍 Auto-Discovery** - Automatic block and flow registration. No manual imports or complex setup.

**📊 Rich Monitoring** - Detailed logging with progress bars and execution summaries.

**🧩 Easily Extensible** - Create custom blocks with simple inheritance. Rich logging and monitoring built-in.

## 📚 Documentation

For comprehensive documentation, including detailed API references, tutorials, and advanced usage examples, visit our **[documentation site](https://ai-innovation.team/)**.

## 📦 Installation


```bash
# Production
pip install sdg-hub

# Development
git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
cd sdg_hub
pip install .[dev]
# or with uv: uv sync --extra dev
```

### Optional Dependencies
```bash
# For vLLM support
pip install sdg-hub[vllm]
# or with uv: uv pip install sdg-hub[vllm]

# For examples
pip install sdg-hub[examples]
# or with uv: uv pip install sdg-hub[examples]
```

## 🚀 Quick Start

### Flow Discovery
```python
from sdg_hub.core.flow import FlowRegistry

# Auto-discover all available flows (no setup needed!)
FlowRegistry.discover_flows()

# List available flows
flows = FlowRegistry.list_flows()
print(f"Available flows: {flows}")

# Search for specific types
qa_flows = FlowRegistry.search_flows(tag="question-generation")
print(f"QA flows: {qa_flows}")
```

### Using Flows
```python
from sdg_hub.core.flow import FlowRegistry, Flow
from datasets import Dataset

# Load the flow by name
flow_name = "Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning"
flow_path = FlowRegistry.get_flow_path(flow_name)
flow = Flow.from_yaml(flow_path)

# Create your dataset with required columns
dataset = Dataset.from_dict({
    'document': ['Your document text here...'],
    'document_outline': ['1. Topic A; 2. Topic B; 3. Topic C'],
    'domain': ['Computer Science'],
    'icl_document': ['Example document for in-context learning...'],
    'icl_query_1': ['Example question 1?'],
    'icl_response_1': ['Example answer 1'],
    'icl_query_2': ['Example question 2?'], 
    'icl_response_2': ['Example answer 2'],
    'icl_query_3': ['Example question 3?'],
    'icl_response_3': ['Example answer 3']
})

# Generate high-quality QA pairs
result = flow.generate(dataset)

# Access generated content
questions = result['question']
answers = result['response']
faithfulness_scores = result['faithfulness_judgment']
relevancy_scores = result['relevancy_score']
```

### Quick Testing with Dry Run
```python
# Test the flow with a small sample first
dry_result = flow.dry_run(dataset, sample_size=1)
print(f"Dry run completed in {dry_result['execution_time_seconds']:.2f}s")
print(f"Output columns: {dry_result['final_dataset']['columns']}")
```


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

---

Built with ❤️ by the Red Hat AI Innovation Team
