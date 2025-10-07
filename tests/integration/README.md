# Integration Tests

This directory contains integration tests for SDG Hub notebooks and end-to-end workflows.

## Overview

Integration tests validate that complete workflows (especially notebooks) execute successfully using real components and configurations.

## Test Structure

Tests are organized to mirror the `examples/` directory structure:

```
tests/integration/
└── knowledge_tuning/
    └── enhanced_summary_knowledge_tuning/
        ├── conftest.py                     # Test fixtures and env setup
        ├── test_knowledge_generation.py    # Integration tests
        ├── test_data/                      # Minimal test data
        ├── README.md                       # Test documentation
```

## Running Integration Tests

### Via tox (Recommended)

```bash
# Run all integration tests with coverage
tox -e py3-integrationcov

# Run integration tests only (no coverage)
tox -e py3-integration
```

### Direct pytest

```bash
# Install with dev dependencies
uv pip install .[dev,examples]

# Run all integration tests
pytest tests/integration/ -v -m integration

# Run specific test suite
pytest tests/integration/knowledge_tuning/enhanced_summary_knowledge_tuning/ -v -m integration

# Run all tests except integration
pytest -m "not integration"
```

## Test Approach

Integration tests use **nbconvert** to convert notebooks to Python scripts and execute them:

1. **Convert**: Notebook → Python script (via nbconvert)
2. **Execute**: Run the script with test environment variables
3. **Validate**: Check that output datasets exist and are loadable

## Environment Configuration

Tests automatically handle both local and CI environments:

- **Local**: Reads from `.env` file in the example directory
- **CI**: Uses GitHub Actions secrets passed as environment variables

See individual test suite README files for specific configuration requirements.

## CI/CD Integration

Integration tests run automatically in GitHub Actions on changes to:
- Core SDG Hub code (`src/sdg_hub/core/**`)
- Example notebooks and flows
- Integration test code itself

See `.github/workflows/integration-test.yml` for workflow configuration.

## Writing New Integration Tests

1. Create test directory mirroring `examples/` structure
2. Add `conftest.py` with environment setup fixtures
3. Write simple tests: convert → execute → validate outputs
4. Document any required API keys or environment variables
5. Update `.github/workflows/integration-test.yml` trigger paths

## Dependencies

Integration tests require the `[dev]` and `[examples]` extras:

```bash
uv pip install .[dev,examples]
```

Key dependencies:
- `nbconvert`: Convert notebooks to Python scripts
- `pytest` and plugins: Test framework
