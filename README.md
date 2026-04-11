<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-banner-light.svg">
    <img alt="SDG Hub" src="docs/assets/logo-banner-light.svg" width="320">
  </picture>
</p>
<p align="center"><em>Composable blocks and flows for synthetic data generation</em></p>
<p align="center">
  <a href="https://pypi.org/project/sdg-hub/"><img src="https://img.shields.io/pypi/v/sdg-hub?style=flat-square" alt="PyPI"></a>
  <a href="https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/Red-Hat-AI-Innovation-Team/sdg_hub/test.yml?style=flat-square&label=tests" alt="Tests"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-brightgreen?style=flat-square" alt="Python 3.10+">
  <a href="https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Red-Hat-AI-Innovation-Team/sdg_hub?style=flat-square" alt="License"></a>
  <a href="https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub"><img src="https://img.shields.io/codecov/c/github/Red-Hat-AI-Innovation-Team/sdg_hub?style=flat-square" alt="Coverage"></a>
  <a href="https://deepwiki.com/Red-Hat-AI-Innovation-Team/sdg_hub"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>

---

<p align="center">
  <img src="docs/assets/demo.gif" alt="SDG Hub Demo" width="700">
</p>

SDG Hub is a Python framework for building synthetic data generation pipelines. Chain LLM, parsing, transform, filtering, and agent blocks into YAML-defined flows -- then generate training data at scale.

## Get Started

```bash
pip install sdg-hub
```

```python
from sdg_hub import FlowRegistry, Flow

# Discover and load a built-in flow
FlowRegistry.discover_flows()
flow = Flow.from_yaml(FlowRegistry.get_flow_path("MCP Server Distillation"))

# Configure and run
flow.set_model_config(model="openai/gpt-4o")
result = flow.generate(dataset)
```

See the [Quick Start](docs/quickstart.md) for a full walkthrough, or browse [all built-in flows](docs/flows/built-in-flows.md).

## Documentation

- [Installation](docs/installation.md) -- setup, optional dependencies, development install
- [Quick Start](docs/quickstart.md) -- end-to-end walkthrough from loading a flow to generating data
- [Core Concepts](docs/concepts.md) -- blocks, flows, registries, and dataset handling
- [Block Reference](docs/blocks/) -- LLM, parsing, transform, filtering, agent, and custom blocks
- [Flow Reference](docs/flows/) -- YAML schema, built-in flows, custom flows
- [API Reference](docs/reference/) -- auto-generated from source
- [Contributing](CONTRIBUTING.md) -- development setup and contribution guidelines

## License

Apache License 2.0 -- see [LICENSE](LICENSE).

---

Built by the [Red Hat AI Innovation Team](https://ai-innovation.team)
