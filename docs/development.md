# Development Guide

Welcome to SDG Hub development! This guide covers everything you need to know about contributing blocks, flows, and other improvements to the SDG Hub ecosystem.

## 🚀 Getting Started

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

3. **Verify Installation**
```bash
# Run tests to ensure everything works
tox -e py3-unit

# Run linting
make verify

# Quick lint check
tox -e fastlint
```

### Development Environment

The `[dev]` extra includes:
- **Testing frameworks** - pytest, tox
- **Linting tools** - pylint, ruff, mypy
- **Documentation tools** - docsify dependencies
- **Pre-commit hooks** - automated code quality checks

## 🧱 Contributing Blocks

### Block Contribution Workflow

1. **Plan Your Block**
   - Identify the category (llm, transform, filtering, evaluation)
   - Define clear input/output specifications
   - Check if similar functionality already exists

2. **Create the Block**
   - Follow the [Custom Blocks Guide](blocks/custom-blocks.md) for implementation
   - Place in appropriate category directory under `src/sdg_hub/core/blocks/`

3. **Add Tests**
   - Create comprehensive tests in `tests/blocks/[category]/`
   - Test both success and error cases
   - Include configuration validation tests

4. **Documentation**
   - Add docstrings following the existing patterns
   - Update relevant documentation pages
   - Include usage examples

### Block Structure Requirements

```python
# Required: Place in appropriate category directory
# src/sdg_hub/core/blocks/[category]/my_new_block.py

from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry
from typing import Any
from datasets import Dataset

@BlockRegistry.register(
    "MyNewBlock",                    # Unique block name
    "category",                      # Block category
    "Description of functionality"   # Clear description
)
class MyNewBlock(BaseBlock):
    """Comprehensive docstring with examples.
    
    This block does X, Y, and Z. It's useful for...
    
    Parameters
    ----------
    param1 : type
        Description of parameter
    param2 : type, optional
        Description of optional parameter
        
    Examples
    --------
    >>> block = MyNewBlock(
    ...     block_name="example",
    ...     input_cols=["text"],
    ...     output_cols=["result"]
    ... )
    >>> result = block.generate(dataset)
    """
    
    # Pydantic field definitions
    custom_param: str = Field(
        default="default_value",
        description="Parameter description"
    )
    
    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Implement the block's processing logic."""
        # Your implementation here
        pass
```

### Block Testing Requirements

Create comprehensive tests following this pattern:

```python
# tests/blocks/[category]/test_my_new_block.py

import pytest
from datasets import Dataset
from sdg_hub.core.utils.error_handling import MissingColumnError
# Import your custom block directly
from .my_new_block import MyNewBlock

class TestMyNewBlock:
    """Test suite for MyNewBlock."""
    
    def test_basic_functionality(self):
        """Test basic block functionality."""
        block = MyNewBlock(
            block_name="test_block",
            input_cols=["input"],
            output_cols=["output"]
        )
        
        dataset = Dataset.from_dict({
            "input": ["test1", "test2", "test3"]
        })
        
        result = block.generate(dataset)
        
        assert "output" in result.column_names
        assert len(result) == 3
    
    def test_configuration_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            MyNewBlock(
                block_name="bad_config",
                input_cols=["input"],
                output_cols=["output"],
                custom_param=""  # Invalid value
            )
    
    def test_missing_columns(self):
        """Test error handling for missing columns."""
        block = MyNewBlock(
            block_name="test_block",
            input_cols=["missing_column"],
            output_cols=["output"]
        )
        
        dataset = Dataset.from_dict({
            "other_column": ["data"]
        })
        
        with pytest.raises(MissingColumnError):
            block.generate(dataset)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test empty dataset
        # Test null values
        # Test malformed data
        pass
```

### Block Categories and Guidelines

#### LLM Blocks (`src/sdg_hub/core/blocks/llm/`)
- **Purpose**: Language model operations
- **Examples**: Chat completion, prompt building, text parsing
- **Requirements**: 
  - Support async operations when possible
  - Include proper error handling for API failures
  - Support multiple providers through LiteLLM

#### Transform Blocks (`src/sdg_hub/core/blocks/transform/`)
- **Purpose**: Data manipulation and reshaping
- **Examples**: Column operations, text processing, data reformatting
- **Requirements**:
  - Preserve data integrity
  - Handle edge cases (empty data, null values)
  - Efficient processing for large datasets

#### Filtering Blocks (`src/sdg_hub/core/blocks/filtering/`)
- **Purpose**: Quality control and data validation
- **Examples**: Value-based filtering, quality gates
- **Requirements**:
  - Clear filtering criteria
  - Comprehensive operator support
  - Good performance on large datasets


## 🌊 Contributing Flows

### Flow Contribution Workflow

1. **Design the Flow**
   - Define clear use case and objectives
   - Plan the block sequence and data flow
   - Specify required input columns and expected outputs

2. **Create Flow Directory Structure**
```
src/sdg_hub/flows/[category]/[use_case]/[variant]/
├── flow.yaml              # Main flow definition
├── prompt_template_1.yaml # Supporting prompt templates
├── prompt_template_2.yaml
└── README.md             # Flow-specific documentation
```

3. **Implement the Flow**
   - Create comprehensive `flow.yaml` with proper metadata
   - Include supporting prompt templates
   - Add parameter documentation

4. **Test the Flow**
   - Test with various input datasets
   - Validate all execution paths
   - Test parameter overrides

5. **Document the Flow**
   - Create clear README with usage examples
   - Document required input format
   - Include expected output structure

### Flow YAML Requirements

```yaml
metadata:
  name: "Descriptive Flow Name"
  description: "Comprehensive description of what this flow does and when to use it"
  version: "1.0.0"
  author: "Your Name <your.email@example.com>"
  license: "Apache-2.0"
  min_sdg_hub_version: "0.2.0"
  
  # Model recommendations
  recommended_models:
    default: "meta-llama/Llama-3.3-70B-Instruct"
    compatible: 
      - "microsoft/phi-4"
      - "mistralai/Mixtral-8x7B-Instruct-v0.1"
    experimental: []
  
  # Categorization tags
  tags:
    - "primary-use-case"
    - "domain"
    - "data-type"
  
  # Dataset requirements
  dataset_requirements:
    required_columns:
      - "column1"
      - "column2"
    description: "Clear description of expected input data format"
    min_samples: 1
    max_samples: 10000

# Runtime parameters
parameters:
  parameter_name:
    type: "string|integer|float|boolean|object"
    default: default_value
    description: "Clear description of what this parameter controls"
    # Optional validation
    min: minimum_value      # for numbers
    max: maximum_value      # for numbers
    allowed_values: [...]   # for strings

# Block sequence
blocks:
  - block_type: "BlockTypeName"
    block_config:
      block_name: "unique_descriptive_name"
      # Block-specific configuration
```


## 🔧 Development Tools and Standards

### Code Quality Standards

**Linting and Formatting**
```bash
# Run full verification suite
make verify

# Individual tools
tox -e lint        # Full pylint check
tox -e fastlint    # Quick pylint check
tox -e ruff        # Ruff formatting and fixes
tox -e mypy        # Type checking

# Format code
tox -e ruff fix
```

**Testing Standards**
```bash
# Run all tests
tox -e py3-unit

# Run with coverage
tox -e py3-unitcov

# Run specific tests
pytest tests/test_specific_file.py
pytest -k "test_pattern"
```

### Documentation Standards

**Docstring Format**
Follow NumPy-style docstrings:

```python
def my_function(param1: str, param2: int = 5) -> bool:
    """One-line summary of the function.
    
    More detailed description if needed. Explain the purpose,
    behavior, and any important notes.
    
    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, optional
        Description of param2, by default 5
        
    Returns
    -------
    bool
        Description of return value
        
    Raises
    ------
    ValueError
        When invalid parameters are provided
        
    Examples
    --------
    >>> result = my_function("test", 10)
    >>> print(result)
    True
    """
    pass
```

**Type Hints**
Use comprehensive type hints:

```python
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset

def process_data(
    dataset: Dataset,
    config: Dict[str, Any],
    filters: Optional[List[str]] = None
) -> Dataset:
    """Process dataset with configuration."""
    pass
```

### Git Workflow

**Branch Naming**
- `feature/block-name-implementation` - New blocks
- `feature/flow-name-implementation` - New flows
- `fix/issue-description` - Bug fixes
- `docs/section-updates` - Documentation updates

**Commit Messages**
Follow conventional commits:
```
type(scope): description

feat(blocks): add TextSummarizerBlock for document summarization
fix(flows): correct parameter validation in QA generation flow
docs(blocks): update LLM block examples with new model config
test(evaluation): add comprehensive tests for faithfulness evaluation
```

**Pull Request Process**
1. Create feature branch from `main`
2. Implement changes with tests and documentation
3. Run full verification: `make verify && tox -e py3-unit`
4. Create PR with clear description
5. Address review feedback
6. Squash and merge when approved

## 📋 Contribution Checklist

### For New Blocks

- [ ] Block placed in correct category directory
- [ ] Inherits from `BaseBlock` and implements `generate()`
- [ ] Registered with `@BlockRegistry.register()`
- [ ] Includes comprehensive docstring with examples
- [ ] Has proper Pydantic field validation
- [ ] Includes error handling and validation
- [ ] Has comprehensive test suite
- [ ] Tests cover success cases, error cases, and edge cases
- [ ] Documentation updated in relevant block category page
- [ ] Code passes all linting checks
- [ ] All tests pass

### For New Flows

- [ ] Flow directory structure follows conventions
- [ ] `flow.yaml` includes complete metadata
- [ ] Required input columns documented
- [ ] Expected output structure documented
- [ ] Supporting prompt templates included
- [ ] Flow-specific README created
- [ ] Integration tests written
- [ ] Dry run tests pass
- [ ] Parameter validation tests included
- [ ] Documentation updated with flow description
- [ ] Example usage provided

### General Requirements

- [ ] Code follows project style guidelines
- [ ] All new code has appropriate type hints
- [ ] Docstrings follow NumPy style
- [ ] No breaking changes to existing APIs
- [ ] Performance considerations addressed
- [ ] Security implications considered
- [ ] Backward compatibility maintained
- [ ] Change log entry added (if applicable)

## 🤝 Community Guidelines

### Getting Help

- **GitHub Issues** - Report bugs, request features
- **GitHub Discussions** - Ask questions, share ideas
- **Documentation** - Check existing docs first
- **Code Examples** - Look at existing implementations

### Best Practices

1. **Start Small** - Begin with simple contributions
2. **Ask Questions** - Don't hesitate to ask for clarification
3. **Follow Patterns** - Study existing code patterns
4. **Test Thoroughly** - Comprehensive testing is essential
5. **Document Well** - Clear documentation helps everyone
6. **Be Patient** - Code review takes time
7. **Stay Updated** - Keep up with project changes

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers get started
- Follow the project's coding standards
- Report issues responsibly

## 🚀 Advanced Contributions

### Framework Extensions

For larger architectural changes:
- Discuss in GitHub Issues first
- Create design document
- Implement incrementally
- Maintain backward compatibility
- Provide migration guide if needed

Ready to contribute? Start by exploring the codebase, running the tests, and trying out some simple improvements. The SDG Hub community welcomes your contributions! 🎉