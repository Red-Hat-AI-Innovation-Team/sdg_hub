# Custom Blocks

This guide walks through building your own block -- from a minimal example to
registration, Pydantic validation, testing, and YAML flow usage.

## Minimal Example

Every custom block must:

1. Inherit from `BaseBlock`.
2. Implement `generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame`.
3. Register with `@BlockRegistry.register()`.

Here is a complete, working block that uppercases a text column:

```python
from typing import Any

import pandas as pd
from pydantic import Field

from sdg_hub import BaseBlock, BlockRegistry


@BlockRegistry.register(
    block_name="UpperCaseBlock",
    category="transform",
    description="Uppercases text in the input column",
)
class UpperCaseBlock(BaseBlock):
    """Converts text values in input_cols[0] to uppercase and writes to output_cols[0]."""

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        result = samples.copy()
        input_col = self.input_cols[0]
        output_col = self.output_cols[0]
        result[output_col] = result[input_col].str.upper()
        return result
```

Usage:

```python
import pandas as pd

df = pd.DataFrame({"text": ["hello world", "foo bar"]})

block = UpperCaseBlock(
    block_name="upper",
    input_cols=["text"],
    output_cols=["text_upper"],
)

result = block(df)
print(result["text_upper"].tolist())
# ['HELLO WORLD', 'FOO BAR']
```

## Pydantic Field Validation

`BaseBlock` inherits from Pydantic's `BaseModel`, so you define configurable
parameters as typed class attributes with `Field()`. Pydantic validates them at
construction time.

```python
from typing import Any

import pandas as pd
from pydantic import Field

from sdg_hub import BaseBlock, BlockRegistry


@BlockRegistry.register(
    block_name="TextTruncateBlock",
    category="transform",
    description="Truncates text to a maximum character length",
)
class TextTruncateBlock(BaseBlock):
    """Truncates text values to max_length characters, optionally adding a suffix."""

    max_length: int = Field(
        ...,
        description="Maximum number of characters to keep",
        gt=0,
    )
    suffix: str = Field(
        default="...",
        description="String appended when text is truncated",
    )

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        result = samples.copy()
        input_col = self.input_cols[0]
        output_col = self.output_cols[0]

        def truncate(text: str) -> str:
            if len(text) <= self.max_length:
                return text
            return text[: self.max_length] + self.suffix

        result[output_col] = result[input_col].apply(truncate)
        return result
```

Invalid configurations raise `ValidationError` at construction:

```python
# Raises ValidationError: max_length must be > 0
block = TextTruncateBlock(
    block_name="trunc",
    input_cols=["text"],
    output_cols=["truncated"],
    max_length=-1,
)
```

## Custom Validation

Override `_validate_custom(df)` to add block-specific checks that run before
`generate()` during the `__call__()` lifecycle:

```python
def _validate_custom(self, df: pd.DataFrame) -> None:
    if len(self.input_cols) != 1:
        raise ValueError("This block requires exactly one input column")
```

## The generate() Method in Detail

The `generate()` method signature, verified from `BaseBlock` in
`src/sdg_hub/core/blocks/base.py`:

```python
@abstractmethod
def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
```

Key points:

- **Input**: `samples` is a `pd.DataFrame`. Never modify it in place -- make a
  copy with `samples.copy()`.
- **Output**: Return a `pd.DataFrame`. It can have more rows, fewer rows, or
  different columns than the input.
- **kwargs**: Runtime overrides passed through `__call__()`. These may include
  flow parameters prefixed with `_flow_`.
- **Column access**: Use `self.input_cols` and `self.output_cols` to know which
  columns to read from and write to. These are normalized to lists or dicts by
  `BaseBlock` validators.

## Registering Your Block

Use the `@BlockRegistry.register()` decorator. The full signature:

```python
@BlockRegistry.register(
    block_name="MyBlock",           # unique name for lookup
    category="transform",          # organizational category
    description="What it does",    # human-readable description
    deprecated=False,              # mark as deprecated (default: False)
    replacement=None,              # name of replacement block if deprecated
)
```

The decorator validates that your class inherits from `BaseBlock` and has a
`generate()` method. After registration, the block appears in
`BlockRegistry.list_blocks()` and `BlockRegistry.discover_blocks()` output.

Choose a category that matches the block's purpose:

| Category | When to use |
|---|---|
| `llm` | Calls a language model |
| `parsing` | Extracts structured data from text |
| `transform` | Reshapes, renames, or manipulates columns |
| `filtering` | Removes rows based on conditions |
| `agent` | Integrates external agent frameworks |
| `mcp` | Uses MCP (Model Context Protocol) tools |

## Testing Your Block

Follow the project's existing test patterns under `tests/blocks/`. Tests use
`pytest` and operate on plain DataFrames:

```python
import pandas as pd
import pytest

from your_module import UpperCaseBlock


class TestUpperCaseBlock:
    """Tests for UpperCaseBlock."""

    def test_basic_uppercase(self):
        df = pd.DataFrame({"text": ["hello", "world"]})
        block = UpperCaseBlock(
            block_name="test_upper",
            input_cols=["text"],
            output_cols=["text_upper"],
        )
        result = block.generate(df)
        assert result["text_upper"].tolist() == ["HELLO", "WORLD"]
        # Original column is preserved
        assert "text" in result.columns

    def test_empty_string(self):
        df = pd.DataFrame({"text": ["", "hello"]})
        block = UpperCaseBlock(
            block_name="test_upper",
            input_cols=["text"],
            output_cols=["text_upper"],
        )
        result = block.generate(df)
        assert result["text_upper"].tolist() == ["", "HELLO"]

    def test_missing_column_raises(self):
        df = pd.DataFrame({"wrong_col": ["hello"]})
        block = UpperCaseBlock(
            block_name="test_upper",
            input_cols=["text"],
            output_cols=["text_upper"],
        )
        with pytest.raises(Exception):
            block(df)  # __call__ validates columns

    def test_output_collision_raises(self):
        df = pd.DataFrame({"text": ["hello"], "text_upper": ["existing"]})
        block = UpperCaseBlock(
            block_name="test_upper",
            input_cols=["text"],
            output_cols=["text_upper"],
        )
        with pytest.raises(Exception):
            block(df)  # __call__ catches collision
```

Run tests:

```bash
uv run pytest tests/blocks/test_upper_case_block.py -v
```

## Using in a YAML Flow

Once registered, your block can be used in flow YAML files by its registered
`block_name`:

```yaml
metadata:
  name: "uppercase_flow"
  version: "1.0.0"
  description: "Demo flow using a custom block"

blocks:
  - block_type: "UpperCaseBlock"
    block_config:
      block_name: "make_uppercase"
      input_cols:
        - "text"
      output_cols:
        - "text_upper"
```

The flow system resolves `block_type: "UpperCaseBlock"` through
`BlockRegistry._get()`, which looks up the class by its registered name.

## Best Practices

- **Copy the input DataFrame.** Always call `samples.copy()` at the start of
  `generate()` to avoid mutating the caller's data.
- **Use descriptive block names.** End class names with `Block` (e.g.
  `TextTruncateBlock`). Use the same name for `block_name` in the registry.
- **Validate early.** Use Pydantic `Field` constraints (`gt`, `ge`, `le`,
  `pattern`) and `_validate_custom()` to catch bad configuration before
  execution.
- **Handle edge cases.** Consider empty strings, `NaN` values, and single-row
  DataFrames.
- **Write tests.** Test both `generate()` directly and `block(df)` via
  `__call__()` to verify validation behavior.

## Next Steps

- [Block System Overview](index.md) -- architecture, categories, and registry API
- [LLM Blocks](llm-blocks.md) -- example of async LLM-powered blocks
- [Transform Blocks](transform-blocks.md) -- example of data transformation blocks
