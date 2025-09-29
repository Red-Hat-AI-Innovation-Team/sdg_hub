# SDG Hub - Synthetic Data Generation Toolkit

[![Build](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/pypi.yaml/badge.svg?branch=main)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/pypi.yaml)
[![Release](https://img.shields.io/github/v/release/Red-Hat-AI-Innovation-Team/sdg_hub)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/releases)
[![License](https://img.shields.io/github/license/Red-Hat-AI-Innovation-Team/sdg_hub)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/main/LICENSE)
[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub/graph/badge.svg?token=SP75BCXWO2)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub)

A modular Python framework for building synthetic data generation pipelines using composable blocks and flows. Transform datasets through **building-block composition** - mix and match LLM-powered and traditional processing blocks to create sophisticated data generation workflows.

## 🧱 Core Philosophy

**Blocks** are composable units that transform datasets - think of them as data processing Lego pieces. Each block performs a specific task: LLM chat, text parsing, evaluation, or transformation.

**Flows** orchestrate multiple blocks into complete pipelines defined in YAML. Chain blocks together to create complex data generation workflows with validation and parameter management.

```
# Simple concept: Blocks transform data, Flows chain blocks together
dataset → Block₁ → Block₂ → Block₃ → enriched_dataset
```

## ✨ Key Features

- **🔧 Modular Composability** - Mix and match blocks like Lego pieces. Build simple transformations or complex multi-stage pipelines with YAML-configured flows.

- **⚡ Async Performance** - High-throughput LLM processing with built-in error handling and concurrent execution.

- **🛡️ Built-in Validation** - Pydantic-based type safety ensures your configurations and data are correct before execution.

- **🔍 Auto-Discovery** - Automatic block and flow registration. No manual imports or complex setup required.

- **📊 Rich Monitoring** - Detailed logging with progress bars and execution summaries for visibility into your pipelines.

- **🧩 Easily Extensible** - Create custom blocks with simple inheritance. Rich logging and monitoring built-in.

## 🚀 Getting Started

Ready to start building synthetic data pipelines? Follow our step-by-step guides:

1. **[Installation](installation.md)** - Set up SDG Hub in your environment
2. **[Quick Start](quick-start.md)** - Build your first data generation pipeline in minutes
3. **[Core Concepts](concepts.md)** - Understand blocks, flows, and the composable architecture

## 📚 Documentation Sections

### Block System
Learn about the modular block architecture that powers SDG Hub:
- **[Block Overview](blocks/overview.md)** - Understanding the block system
- **[LLM Blocks](blocks/llm-blocks.md)** - Chat, prompt building, and text parsing
- **[Transform Blocks](blocks/transform-blocks.md)** - Data transformation and manipulation
- **[Filtering Blocks](blocks/filtering-blocks.md)** - Quality filtering and data validation
- **[Custom Blocks](blocks/custom-blocks.md)** - Building your own processing blocks

### Flow System
Master the orchestration system for building complete pipelines:
- **[Flow Overview](flows/overview.md)** - Understanding flow orchestration
- **[YAML Configuration](flows/yaml-configuration.md)** - Structure and parameters
- **[Flow Discovery](flows/discovery.md)** - Registry and auto-discovery system
- **[Custom Flows](flows/custom-flows.md)** - Building custom pipeline flows

### Advanced Topics
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Development](development.md)** - Contributing and development guidelines

## 🤝 Contributing

We welcome contributions! Please see our [development guide](development.md) for guidelines on how to contribute to this project.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/main/LICENSE) file for details.

---

Built with ❤️ by the Red Hat AI Innovation Team