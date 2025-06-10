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
      Important Links:
      <a href="docs/">Documentation</a>, 
      <a href="examples/">Examples</a>,
      <a href="https://www.youtube.com/watch?v=aGKCViWjAmA">Video Tutorial</a> &
      <a href="https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub">GitHub</a>.
    </h3>
</html>

SDG Hub is designed to simplify data creation for LLMs, allowing users to chain computational units and build powerful flows for generating data and processing tasks. Define complex workflows using nothing but YAML configuration files.

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

## 📚 Documentation

Explore the full documentation for detailed guides:

* **[Architecture Guide](docs/architecture.md)** - Core concepts and design principles
* **[Available Blocks](docs/blocks.md)** - Complete reference of all blocks
* **[Prompt Configuration](docs/prompts.md)** - How to configure LLM prompts

## 🏁 Quick Start



## 📺 Video Tutorial

For a comprehensive walkthrough of sdg_hub:

[![SDG Hub Tutorial](https://img.youtube.com/vi/aGKCViWjAmA/0.jpg)](https://www.youtube.com/watch?v=aGKCViWjAmA)

## 🤝 Contributing

We welcome contributions from the community! Whether it's bug reports, feature requests, documentation improvements, or code contributions, please check out our [contribution guidelines](CONTRIBUTING.md).

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

Built with ❤️ by the Red Hat AI Innovation Team
