# SDG Hub: Synthetic Data Generation Toolkit

[![Build](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/pypi.yaml/badge.svg?branch=main)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/pypi.yaml)
[![Release](https://img.shields.io/github/v/release/Red-Hat-AI-Innovation-Team/sdg_hub)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/releases)
[![License](https://img.shields.io/github/license/Red-Hat-AI-Innovation-Team/sdg_hub)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/main/LICENSE)
[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub/graph/badge.svg?token=SP75BCXWO2)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub)

<html>
    <h3 align="center">
      A modular, scalable, and efficient solution for creating synthetic data generation flows in a "low-code" manner.
    </h3>
    <h3 align="center">
      <a href="http://ai-innovation.team/sdg_hub">Documentation</a> |
      <a href="examples/">Examples</a> |
      <a href="https://www.youtube.com/watch?v=aGKCViWjAmA">Video Tutorial</a>
    </h3>
</html>

SDG Hub is designed to simplify data creation for LLMs, allowing users to chain computational units and build powerful flows for generating data and processing tasks. Define complex workflows using nothing but YAML configuration files.

**📖 Full documentation available at: [https://ai-innovation.team/sdg_hub](https://ai-innovation.team/sdg_hub)**

---

## ✨ Key Features

- **Low-Code Flow Creation**: Build sophisticated data generation pipelines using
  simple YAML configuration files without writing any code.

- **Modular Block System**: Compose workflows from reusable, self-contained
  blocks that handle LLM calls, data transformations, and filtering.

- **LLM-Agnostic**: Works with any language model through configurable
  prompt templates and generation parameters.

- **Prompt Engineering Friendly**: Tune LLM behavior by editing declarative YAML prompts.

## 🚀 Installation

### Stable Release (Recommended)

```bash
pip install sdg-hub
```

### Development Version

```bash
pip install git+https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
```

## 🏁 Quick Start

### Prerequisites

Before getting started, make sure you have:
- Python 3.8 or higher
- LLM Inference Endpoint exposed through OpenAI API

### Simple Example

Here's the simplest way to get started:

```python
from sdg_hub.flow_runner import run_flow

# Run a basic knowledge generation flow
run_flow(
    ds_path="my_data.jsonl",
    save_path="output.jsonl", 
    endpoint="http://0.0.0.0:8000/v1",
    flow_path="flows/generation/knowledge/synth_knowledge.yaml"
)
```

### Advanced Configuration
You can invoke any built-in flow using run_flow:
```python
from sdg_hub.flow_runner import run_flow

run_flow(
    ds_path="path/to/dataset.jsonl",
    save_path="path/to/output.jsonl",
    endpoint="http://0.0.0.0:8000/v1",
    flow_path="path/to/flow.yaml",
    checkpoint_dir="path/to/checkpoints",
    batch_size=8,
    num_workers=32,
    save_freq=2,
)
```

### 📂 Available Built-in Flows

You can start with any of these YAML flows out of the box:

#### 🔎 **Knowledge Flows**

| Flow | Description |
|------|-------------|
| `synth_knowledge.yaml` | Produces document-grounded questions and answers for factual memorization |
| `synth_knowledge1.5.yaml` | Improved version that builds intermediate representations for better recall |

#### 🧠 **Skills Flows**

| Flow | Description |
|------|-------------|
| `synth_skills.yaml` | Freeform skills QA generation (eg: "Create a new github issue to add type hints") |
| `synth_grounded_skills.yaml` | Domain-specific skill generation (eg: "From the given conversation create a table for feature requests") |
| `improve_responses.yaml` | Uses planning and critique-based refinement to improve generated answers |

All these can be found here: [flows](src/sdg_hub/flows)

## 📺 Video Tutorial

For a comprehensive walkthrough of sdg_hub:

[![SDG Hub Tutorial](https://img.youtube.com/vi/aGKCViWjAmA/0.jpg)](https://www.youtube.com/watch?v=aGKCViWjAmA)

## 🤝 Contributing

We welcome contributions from the community! Whether it's bug reports, feature requests, documentation improvements, or code contributions, please check out our [contribution guidelines](CONTRIBUTING.md).

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

Built with ❤️ by the Red Hat AI Innovation Team
