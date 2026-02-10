# Contributing to SDG Hub

Welcome to SDG Hub development! This guide covers everything you need to know about contributing blocks, flows, and other improvements to the SDG Hub ecosystem.

For detailed documentation including examples and advanced patterns, see our comprehensive [Development Guide](docs/development.md).

## Quick Start

### Development Setup

1. **Clone the Repository**

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
cd sdg_hub
```

2. **Install uv** (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

3. **Install Dependencies and Set Up Pre-commit Hooks**

```bash
# Install development dependencies
uv sync --extra dev

# Install pre-commit hooks (required for all contributors)
uv run pre-commit install                      # Runs ruff on commit
uv run pre-commit install --hook-type commit-msg  # Validates commit messages
```

## Development Tools

### Linting and Code Quality

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Pre-commit hooks run automatically, but you can also run manually:

```bash
# Run ruff linter with auto-fix
uv run ruff check --fix src/ tests/

# Run ruff formatter
uv run ruff format src/ tests/

# Check only (no fixes) - same as CI
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

**Optional development tools** (require additional dependencies):

```bash
make actionlint    # Lint GitHub Actions (requires: actionlint, shellcheck)
make md-lint       # Lint markdown files (requires: podman/docker)
```

### Testing

SDG Hub uses [pytest](https://docs.pytest.org/) for testing:

```bash
# Run all unit tests
uv run pytest tests/blocks tests/connectors tests/flow tests/utils -m "not (examples or slow)"

# Run with coverage
uv run pytest --cov=sdg_hub --cov-report=term tests/blocks tests/connectors tests/flow tests/utils

# Run specific tests
uv run pytest tests/test_specific_file.py
uv run pytest -k "test_pattern"

# Run integration tests (requires API keys)
uv run pytest tests/integration -v -s
```

## Contributing Blocks

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

## Contributing Flows

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

## Contribution Checklist

### For New Blocks

- [ ] Block placed in correct category directory
- [ ] Inherits from `BaseBlock` and implements `generate()`
- [ ] Registered with `@BlockRegistry.register()`
- [ ] Comprehensive docstring with examples
- [ ] Proper Pydantic field validation
- [ ] Comprehensive test suite
- [ ] Documentation updated
- [ ] All linting checks pass (`uv run ruff check`)
- [ ] All tests pass (`uv run pytest`)

### For New Flows

- [ ] Flow directory structure follows conventions
- [ ] Complete metadata in `flow.yaml`
- [ ] Required input columns documented
- [ ] Supporting templates included
- [ ] Flow-specific README created
- [ ] Integration tests written
- [ ] Documentation updated

## Development Workflow

### Git Workflow

**Branch Naming:**

- `feature/block-name-implementation` - New blocks
- `feature/flow-name-implementation` - New flows
- `fix/issue-description` - Bug fixes
- `docs/section-updates` - Documentation updates

**Commit Messages:**

We use [Conventional Commits](https://www.conventionalcommits.org/) format, enforced via pre-commit hook.

Format: `<type>(<scope>): <description>`

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting, no logic change) |
| `refactor` | Code refactoring |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `build` | Build system or dependencies |
| `ci` | CI/CD configuration |
| `chore` | Maintenance tasks |
| `revert` | Reverting a previous commit |

Examples:

```
feat(blocks): add TextSummarizerBlock for document summarization
fix(flows): correct parameter validation in QA generation flow
docs(blocks): update LLM block examples with new model config
```

### Pre-commit Hooks

Pre-commit hooks run automatically to ensure code quality. Install them once after cloning:

```bash
uv run pre-commit install                      # Ruff linting on commit
uv run pre-commit install --hook-type commit-msg  # Commit message validation
```

**What the hooks do:**

| Hook | Stage | Description |
|------|-------|-------------|
| `uv-lock` | commit | Keeps `uv.lock` in sync with `pyproject.toml` |
| `ruff` | commit | Lints Python code with auto-fix |
| `ruff-format` | commit | Formats Python code |
| `conventional-pre-commit` | commit-msg | Validates commit message format |

**Pull Request Process:**

1. Create feature branch from `main`
2. Implement changes with tests and documentation
3. Run tests locally: `uv run pytest tests/blocks tests/connectors tests/flow tests/utils`
4. Create PR with clear description
5. Address review feedback
6. Squash and merge when approved

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers get started
- Follow the project's coding standards
- Report issues responsibly

## Documentation

For comprehensive guides and examples:

- **[Development Guide](docs/development.md)** - Complete development documentation
- **[Custom Blocks](docs/blocks/custom-blocks.md)** - Building custom blocks
- **[Flow Configuration](docs/flows/yaml-configuration.md)** - YAML configuration guide
- **[Block System Overview](docs/blocks/overview.md)** - Understanding the block architecture
- **[Flow System Overview](docs/flows/overview.md)** - Understanding flow orchestration

## Getting Help

- **GitHub Issues** - Report bugs, request features
- **GitHub Discussions** - Ask questions, share ideas
- **Documentation** - Check existing docs first
- **Code Examples** - Look at existing implementations

## Documentation Guidelines

### NumPy-Style Docstrings

If you choose to add docstrings to your functions, we recommend following the NumPy docstring format for consistency with the scientific Python ecosystem.

#### Basic Structure

```python
def example_function(param1, param2=None):
    """Brief description of the function.

    Longer description providing more context about what the function does,
    its purpose, and any important behavioral notes.

    Parameters
    ----------
    param1 : str
        Description of the first parameter
    param2 : int, optional
        Description of the second parameter (default: None)

    Returns
    -------
    bool
        Description of what the function returns

    Raises
    ------
    ValueError
        When invalid input is provided

    Examples
    --------
    >>> result = example_function("hello", 42)
    >>> print(result)
    True
    """
```

#### Key Guidelines

- **Summary**: Start with a concise one-line description
- **Parameters**: Document all function parameters with types and descriptions
- **Returns**: Describe return values with types and meaning
- **Types**: Use standard Python types (`str`, `int`, `list`, `dict`, etc.)
- **Optional parameters**: Mark default parameters as "optional"
- **Examples**: Include simple usage examples when helpful

#### When to Add Docstrings

Docstrings are **optional** but recommended for:

- Public API functions and classes
- Complex functions with multiple parameters
- Functions that might be confusing to other developers
- Core framework components

#### When to Skip Docstrings

You may skip docstrings for:

- Simple utility functions with obvious behavior
- Private/internal functions (starting with `_`)
- Functions with self-explanatory names and simple parameters

**Remember**: Quality over quantity. A well-written docstring is better than a verbose one, and no docstring is better than a poor one.

Thank you for contributing to SDG Hub!
