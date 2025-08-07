# Contributing to SDG Hub

Welcome to SDG Hub development! This guide covers everything you need to know about contributing blocks, flows, and other improvements to the SDG Hub ecosystem.

For detailed documentation including examples and advanced patterns, see our comprehensive [Development Guide](docs/development.md).

## 🚀 Quick Start

### Development Setup

1. **Clone the Repository**
```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
cd sdg_hub
```

2. **Install Development Dependencies**
```bash
# Using uv (recommended)
uv sync --extra dev

# Or using pip
pip install .[dev]
```

## 🛠️ Development Tools

### Linting and Code Quality

SDG Hub uses a Makefile for linting:

- **CI changes** - `make actionlint`
- **Documentation changes** - `make md-lint` 
- **Code changes** - `make verify`

Individual linting tools:
```bash
tox -e lint        # Full pylint check
tox -e fastlint    # Quick pylint check
tox -e ruff        # Ruff formatting and fixes
tox -e mypy        # Type checking
```

### Testing

SDG Hub uses [tox](https://tox.wiki/) for test automation and [pytest](https://docs.pytest.org/) as a test framework:

```bash
# Run all tests
tox -e py3-unit

# Run with coverage
tox -e py3-unitcov

# Run specific tests
pytest tests/test_specific_file.py
pytest -k "test_pattern"
```

## 🧱 Contributing Blocks

Blocks are the core processing units of SDG Hub. To contribute a new block:

1. **Choose the appropriate category**: `llm`, `transform`, `filtering`, or `evaluation`
2. **Implement your block** following the [Custom Blocks Guide](docs/blocks/custom-blocks.md)
3. **Add comprehensive tests** in `tests/blocks/[category]/`
4. **Update documentation** in the relevant block category page

### Example Block Structure

```python
from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry

@BlockRegistry.register("MyNewBlock", "category", "Description")
class MyNewBlock(BaseBlock):
    """Comprehensive docstring with examples."""
    
    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        # Your implementation here
        pass
```

## 🌊 Contributing Flows

Flows orchestrate multiple blocks into complete pipelines. To contribute a new flow:

1. **Design your flow** with clear use case and objectives
2. **Create flow directory structure** under `src/sdg_hub/flows/[category]/`
3. **Implement the flow** with comprehensive YAML configuration
4. **Add tests** and documentation

### Flow Directory Structure

```
src/sdg_hub/flows/[category]/[use_case]/[variant]/
├── flow.yaml              # Main flow definition
├── prompt_template_1.yaml # Supporting templates
└── README.md             # Flow documentation
```

## 📋 Contribution Checklist

### For New Blocks
- [ ] Block placed in correct category directory
- [ ] Inherits from `BaseBlock` and implements `generate()`
- [ ] Registered with `@BlockRegistry.register()`
- [ ] Comprehensive docstring with examples
- [ ] Proper Pydantic field validation
- [ ] Comprehensive test suite
- [ ] Documentation updated
- [ ] All linting checks pass
- [ ] All tests pass

### For New Flows
- [ ] Flow directory structure follows conventions
- [ ] Complete metadata in `flow.yaml`
- [ ] Required input columns documented
- [ ] Supporting templates included
- [ ] Flow-specific README created
- [ ] Integration tests written
- [ ] Documentation updated

## 🔄 Development Workflow

### Git Workflow

**Branch Naming:**
- `feature/block-name-implementation` - New blocks
- `feature/flow-name-implementation` - New flows
- `fix/issue-description` - Bug fixes
- `docs/section-updates` - Documentation updates

**Commit Messages:**
Follow conventional commits:
```
feat(blocks): add TextSummarizerBlock for document summarization
fix(flows): correct parameter validation in QA generation flow
docs(blocks): update LLM block examples with new model config
```

**Pull Request Process:**
1. Create feature branch from `main`
2. Implement changes with tests and documentation
3. Run full verification: `make verify && tox -e py3-unit`
4. Create PR with clear description
5. Address review feedback
6. Squash and merge when approved

## 🤝 Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers get started
- Follow the project's coding standards
- Report issues responsibly

## 📚 Documentation

For comprehensive guides and examples:

- **[Development Guide](docs/development.md)** - Complete development documentation
- **[Custom Blocks](docs/blocks/custom-blocks.md)** - Building custom blocks
- **[Flow Configuration](docs/flows/yaml-configuration.md)** - YAML configuration guide
- **[Block System Overview](docs/blocks/overview.md)** - Understanding the block architecture
- **[Flow System Overview](docs/flows/overview.md)** - Understanding flow orchestration

## 🚀 Getting Help

- **GitHub Issues** - Report bugs, request features
- **GitHub Discussions** - Ask questions, share ideas
- **Documentation** - Check existing docs first
- **Code Examples** - Look at existing implementations

Thank you for contributing to SDG Hub! 🎉
