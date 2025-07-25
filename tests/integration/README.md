# Integration Tests

This directory contains integration tests for SDG Hub notebooks and end-to-end workflows.

## Overview

Integration tests validate that complete workflows (especially notebooks) execute successfully. These tests use [papermill](https://papermill.readthedocs.io/) to execute notebooks with test parameters and validate their outputs.

## Running Integration Tests

### Via tox (Recommended)

```bash
# Run all integration tests
tox -e py3-integration

# Run integration tests with verbose output
tox -e py3-integration -- -v
```

### Direct pytest

```bash
# Install with integration test dependencies
pip install -e .[dev,examples]

# Run integration tests
pytest tests/integration/ -v -m integration

# Run all tests except integration
pytest -m "not integration"
```

## Test Structure

- `conftest.py`: Shared fixtures and configuration
- `notebook_utils.py`: Utility functions for notebook testing
- `test_*_notebook.py`: Individual notebook integration tests

## Writing New Integration Tests

1. Create a new test file: `test_your_notebook_name.py`
2. Use the `@pytest.mark.integration` marker
3. Mock external dependencies (LLM servers, datasets) when possible
4. Use `execute_notebook_with_params()` to run notebooks with test parameters
5. Validate outputs using `validate_notebook_execution()` and custom assertions

## Dependencies

Integration tests require additional dependencies installed via the `[dev]` and `[examples]` extras:

- `papermill`: For notebook execution
- `jupyter` and `ipykernel`: For notebook execution environment

## CI/CD Integration

Integration tests can be added to GitHub Actions workflows:

```yaml
- name: Run integration tests
  run: tox -e py3-integration
```

For tests requiring LLM servers, consider:
- Using mock servers in CI
- Skipping real server tests in CI (mark with `@pytest.mark.slow`)
- Running full integration tests in a separate, scheduled workflow