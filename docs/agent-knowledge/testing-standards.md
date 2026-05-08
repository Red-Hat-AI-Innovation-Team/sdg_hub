# Testing Standards

What "tested" means in SDG Hub, and how to write tests that provide real
safety against regressions.

## Definition of "Tested"

A module is tested when all of these are true:

1. **Test file exists** at the expected path (see directory structure below)
2. **Tests cover success AND error cases** -- not just the happy path
3. **Tests assert specific values or properties**, not just "something was
   returned" (e.g., `assert result is not None` is insufficient)
4. **Coverage is at least 80%** for the module under test

## Test Directory Structure

```text
tests/
  blocks/          # Block tests, mirroring src/sdg_hub/core/blocks/
    agent/
    code/
    filtering/
    llm/
    mcp_blocks/
    parsing/
    transform/
    testdata/      # Shared test fixtures and config files
  connectors/      # Connector tests
    agent/
    code_interpreter/
    http/
  flow/            # Flow system tests
  utils/           # Utility tests
  integration/     # Integration tests (require API keys / external services)
```

## Test File Naming

| Convention | Example |
|-----------|---------|
| File name | `test_{module_name}.py` |
| Test function | `test_{behavior_being_tested}` |
| Test class (optional) | `Test{ClassName}` |

Examples:

- Module `llm_chat_block.py` gets test file `test_llm_chat_block.py`
- `test_generate_returns_expected_columns`
- `test_generate_raises_on_missing_input_column`

## Mocking LLM Clients

Blocks that call LLMs must be tested with mocked clients. Here is the
standard pattern:

```python
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sdg_hub.core.blocks.llm import LLMChatBlock


@pytest.fixture
def mock_litellm_completion():
    """Mock LiteLLM completion to avoid real API calls."""
    with patch(
        "sdg_hub.core.blocks.llm.llm_chat_block.completion"
    ) as mock_completion:
        mock_response = MagicMock()
        choice = MagicMock()
        choice.message = MagicMock(content="Mocked LLM response")
        mock_response.choices = [choice]
        mock_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Mocked LLM response",
                        "role": "assistant",
                    }
                }
            ],
        }
        mock_completion.return_value = mock_response
        yield mock_completion


def test_generate_populates_output_column(mock_litellm_completion):
    """LLMChatBlock should write LLM responses to the output column."""
    block = LLMChatBlock(
        block_name="test_llm",
        input_cols=["question"],
        output_cols=["answer"],
        model="gpt-4o",
    )
    dataset = pd.DataFrame({"question": ["What is 2+2?"]})

    result = block.generate(dataset)

    assert "answer" in result.columns
    assert result["answer"].iloc[0] == "Mocked LLM response"
    mock_litellm_completion.assert_called_once()
```

Key points:

- Patch at the import location (`sdg_hub.core.blocks.llm.llm_chat_block.completion`),
  not the source module
- Mock both `choices[0].message.content` and `model_dump()` because different
  code paths access the response differently
- Assert specific output values, not just that the mock was called

## Running Tests

```bash
# Unit tests (excludes slow/integration)
uv run pytest tests/blocks tests/connectors tests/flow tests/utils \
    -m "not (examples or slow)"

# With coverage report
uv run pytest --cov=sdg_hub --cov-report=term \
    tests/blocks tests/connectors tests/flow tests/utils

# Structural check: verify test files exist for all blocks
find src/sdg_hub/core/blocks -name "*.py" -not -name "__init__.py" \
    -not -name "base*.py" -not -name "registry.py" | while read f; do
    name=$(basename "$f" .py)
    if ! find tests/blocks -name "test_${name}.py" | grep -q .; then
        echo "MISSING TEST: $f"
    fi
done
```

## Test Marks

Use pytest marks to categorize tests that have special requirements:

| Mark | Purpose | When to use |
|------|---------|-------------|
| `@pytest.mark.integration` | Requires external services or API keys | Tests that call real APIs |
| `@pytest.mark.slow` | Takes more than a few seconds | Large dataset tests, end-to-end flows |
| `@pytest.mark.examples` | Tests for example notebooks/scripts | Example validation |

These marks are excluded from the default unit test run. Integration tests
are run separately:

```bash
# Integration tests only
uv run pytest tests/integration -v -s
```
