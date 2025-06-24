# Development Guide

This guide covers development setup, testing, and contribution guidelines for SDG Hub.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
cd sdg_hub

# Install in development mode
pip install -e .[dev]
```

### Optional Dependencies

```bash
# Web interface
pip install -e .[web_interface]

# Examples dependencies  
pip install -e .[examples]

# All dependencies
pip install -e .[dev,web_interface,examples]
```

## Development Commands

### Testing

Run all tests:
```bash
pytest tests/
```

Run specific test:
```bash
pytest tests/test_filename.py
```

Run tests with coverage:
```bash
tox -e py3-unitcov
```

### Code Quality

Format code:
```bash
tox -e ruff fix
# or
./scripts/ruff.sh fix
```

Check code formatting:
```bash
tox -e ruff check
```

Run linting:
```bash
# Full pylint (slower)
tox -e lint

# Fast linting
tox -e fastlint
```

Type checking:
```bash
tox -e mypy
```

Run all checks:
```bash
make verify
```

This runs fastlint, mypy, and ruff via tox.

## Git Workflow

### Important Guidelines

- **Always create a feature branch**
- **Never push directly to main**
- Follow conventional commit messages

### Branch Creation

```bash
# Create and switch to feature branch
git checkout -b feature-branch-name

# Push to remote
git push origin feature-branch-name
```

### Commit Guidelines

Use conventional commit format:

```bash
# Examples
git commit -m "feat: add new block type for data filtering"
git commit -m "fix: resolve issue with checkpoint loading"  
git commit -m "docs: update installation instructions"
git commit -m "test: add unit tests for LLMBlock"
```

## Architecture Overview

### Core Components

1. **Blocks** (`src/sdg_hub/blocks/`)
   - `Block`: Abstract base class
   - `LLMBlock`: Language model blocks
   - `utilblocks.py`: Utility blocks

2. **Flows** (`src/sdg_hub/flow.py`)
   - YAML-based pipeline orchestration
   - Block execution management
   - Data flow coordination

3. **Registry System** (`src/sdg_hub/registry.py`)
   - `BlockRegistry`: Block type management
   - `PromptRegistry`: Prompt template management

4. **Prompts** (`src/sdg_hub/configs/`)
   - YAML-based prompt templates
   - Jinja2 templating support

### Data Flow

- Uses Hugging Face Datasets (Arrow tables)
- Supports checkpointing for reliability
- Processes data through block chains

## Block Development

### Creating a New Block

1. **Inherit from Block base class**:

```python
from sdg_hub.blocks import Block
from sdg_hub.registry import BlockRegistry
from datasets import Dataset

@BlockRegistry.register("MyCustomBlock")
class MyCustomBlock(Block):
    def generate(self, dataset: Dataset) -> Dataset:
        # Your block logic here
        return processed_dataset
```

2. **Implement required methods**:

```python
def generate(self, dataset: Dataset) -> Dataset:
    """Main processing method - required"""
    pass

def _validate(self, dataset: Dataset) -> None:
    """Input validation - optional"""
    pass

def _load_config(self, config_path: str) -> dict:
    """Configuration loading - optional"""
    pass
```

3. **Register the block**:

The `@BlockRegistry.register("BlockName")` decorator automatically registers your block.

### Block Testing

Create tests in `tests/blocks/`:

```python
import pytest
from datasets import Dataset
from your_module import MyCustomBlock

def test_my_custom_block():
    # Setup test data
    test_data = Dataset.from_dict({
        "input_column": ["test1", "test2"]
    })
    
    # Initialize block
    block = MyCustomBlock(block_config={})
    
    # Test generation
    result = block.generate(test_data)
    
    # Assertions
    assert len(result) == len(test_data)
    assert "output_column" in result.column_names
```

## Testing Conventions

### Test Structure

```
tests/
├── blocks/           # Block-specific tests
│   ├── test_llmblock.py
│   └── utilblocks/   # Utility block tests
├── flows/            # Flow-related tests
└── test_*.py         # General tests
```

### Test Data

Store test data in `testdata/` subdirectories:

```
tests/blocks/testdata/
├── test_config.yaml
└── sample_data.json
```

### Testing Best Practices

1. **Test both positive and negative cases**
2. **Include edge cases and error conditions**  
3. **Use pytest fixtures for common setup**
4. **Mock external dependencies (APIs, files)**
5. **Test configuration loading and validation**

### Example Test

```python
import pytest
from datasets import Dataset
from sdg_hub.blocks import LLMBlock

@pytest.fixture
def sample_dataset():
    return Dataset.from_dict({
        "prompt": ["Hello", "World"],
        "context": ["test1", "test2"]
    })

def test_llm_block_generation(sample_dataset):
    block = LLMBlock(
        block_config={
            "model_id": "test-model",
            "output_cols": ["response"]
        }
    )
    
    # Mock the LLM call
    with patch('sdg_hub.blocks.llmblock.some_llm_function') as mock_llm:
        mock_llm.return_value = ["response1", "response2"]
        
        result = block.generate(sample_dataset)
        
        assert "response" in result.column_names
        assert len(result) == 2
```

## Contributing

### Contribution Types

We welcome:
- **Bug reports and fixes**
- **Feature requests and implementations**  
- **Documentation improvements**
- **Test coverage improvements**
- **Performance optimizations**

### Pull Request Process

1. **Create feature branch**
2. **Make changes with tests**
3. **Run all quality checks**:
   ```bash
   make verify
   pytest tests/
   ```
4. **Update documentation if needed**
5. **Submit pull request**

### Code Review Guidelines

- **Code should be well-documented**
- **All tests must pass**
- **Follow existing code style**
- **Include appropriate error handling**
- **Update relevant documentation**

### Documentation Standards

- **Docstrings**: Use Google-style docstrings
- **Type hints**: Include type annotations
- **Examples**: Provide usage examples
- **Comments**: Explain complex logic

```python
def my_function(param1: str, param2: int) -> Dict[str, Any]:
    """Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is invalid
        
    Example:
        >>> result = my_function("test", 42)
        >>> print(result)
        {'status': 'success'}
    """
    pass
```

## Release Process

### Version Management

Versions follow semantic versioning (SemVer):
- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes

### Creating a Release

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Create release branch**
4. **Submit PR to main**
5. **Tag release after merge**

## Troubleshooting

### Common Development Issues

1. **Import errors**: Ensure development installation with `pip install -e .`
2. **Test failures**: Check Python version compatibility
3. **Linting errors**: Run `tox -e ruff fix` to auto-fix
4. **Type errors**: Run `tox -e mypy` for detailed type checking

### Getting Help

- **Issues**: Create GitHub issues for bugs/features
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check existing docs first
- **Code Review**: Request reviews from maintainers