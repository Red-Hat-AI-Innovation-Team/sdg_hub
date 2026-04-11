# Block System Overview

Blocks are the core building units of SDG Hub. Each block is a self-contained,
composable processor that takes a pandas DataFrame as input, transforms it, and
returns a new DataFrame. You chain blocks together -- either in Python or via
YAML flows -- to build data-generation pipelines.

```
DataFrame --> Block_1 --> Block_2 --> Block_3 --> DataFrame
```

Blocks operate on `pandas.DataFrame` internally. The `Flow.generate()` method
accepts both `pandas.DataFrame` and `datasets.Dataset` and handles conversion
automatically. When using blocks directly outside a flow, always pass a
`pandas.DataFrame`.

## Block Architecture

Every block inherits from `BaseBlock`, which itself inherits from both Pydantic's
`BaseModel` and Python's `ABC`:

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class BaseBlock(BaseModel, ABC):
    ...
```

**Source:** `src/sdg_hub/core/blocks/base.py`

### Core fields

All blocks share these Pydantic fields defined on `BaseBlock`:

| Field | Type | Description |
|---|---|---|
| `block_name` | `str` (required) | Unique identifier for this block instance. |
| `block_type` | `Optional[str]` | Category hint (e.g. `"llm"`, `"transform"`). Note: this is different from the `block_type` key in flow YAML files, which specifies the class name for BlockRegistry lookup (e.g. `"LLMChatBlock"`). |
| `input_cols` | `str`, `list[str]`, `dict[str, Any]`, or `None` | Columns the block reads from the input DataFrame. |
| `output_cols` | `str`, `list[str]`, `dict[str, Any]`, or `None` | Columns the block writes to the output DataFrame. |

Pydantic's `ConfigDict(extra="allow", arbitrary_types_allowed=True)` is set on
`BaseBlock`, so subclasses can accept additional fields and types like DataFrames.

### The two key methods

**`generate()` -- abstract, you implement this:**

```python
@abstractmethod
def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    ...
```

Every block subclass must override `generate()`. It receives a pandas DataFrame
and must return a pandas DataFrame.

**`__call__()` -- concrete, wraps `generate()` with validation and logging:**

```python
def __call__(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    ...
```

When you call a block as `block(df)`, the `__call__` method runs the full
lifecycle: input logging, validation, `generate()` execution, and output logging.
It also handles runtime keyword overrides -- if you pass extra `**kwargs`, they
are validated against the block's Pydantic fields and temporarily applied for
that execution.

### Serialization helpers

| Method | Signature | Description |
|---|---|---|
| `get_config()` | `def get_config(self) -> dict[str, Any]` | Returns block configuration as a dict (calls `self.model_dump()`). |
| `from_config()` | `@classmethod def from_config(cls, config: dict[str, Any]) -> BaseBlock` | Reconstruct a block from a config dict. |
| `get_info()` | `def get_info(self) -> dict[str, Any]` | Returns `get_config()` plus `block_class` key with the class name. |

## Block Categories

SDG Hub ships with blocks organized into six categories. Each category has a
dedicated documentation page.

| Category | Blocks | Doc Page |
|---|---|---|
| **llm** | `LLMChatBlock`, `PromptBuilderBlock`, `LLMResponseExtractorBlock` | [LLM Blocks](llm-blocks.md) |
| **parsing** | `TagParserBlock`, `RegexParserBlock`, `JSONParserBlock`, `TextParserBlock` (deprecated) | [Parsing Blocks](parsing-blocks.md) |
| **transform** | `TextConcatBlock`, `DuplicateColumnsBlock`, `RenameColumnsBlock`, `MeltColumnsBlock`, `RowMultiplierBlock`, `IndexBasedMapperBlock`, `SamplerBlock`, `UniformColumnValueSetter`, `JSONStructureBlock` | [Transform Blocks](transform-blocks.md) |
| **filtering** | `ColumnValueFilterBlock` | [Filtering Blocks](filtering-blocks.md) |
| **agent** | `AgentBlock`, `AgentResponseExtractorBlock` | [Agent Blocks](agent-blocks.md) |
| **mcp** | `MCPAgentBlock` | [Agent Blocks](agent-blocks.md) |

`TextParserBlock` is deprecated -- use `TagParserBlock` or `RegexParserBlock`
instead.

## Block Lifecycle

When you call a block via `__call__()` (i.e. `block(df)`), the following steps
execute in order:

### 1. Input logging

`_log_input_data(df)` prints a Rich-formatted panel summarizing the input:
block type, row count, column names, and expected output columns.

### 2. DataFrame validation

`_validate_dataframe(df)` runs three checks:

- **Empty check** -- raises `EmptyDatasetError` if the DataFrame has zero rows.
- **Input column check** -- raises `MissingColumnError` if any column listed in
  `input_cols` is missing from the DataFrame.
- **Output column collision check** -- raises `OutputColumnCollisionError` if any
  column listed in `output_cols` already exists in the DataFrame.

Subclasses can add additional checks by overriding `_validate_custom(df)`.

### 3. `generate()` execution

The abstract `generate()` method runs your processing logic.

### 4. Output logging

`_log_output_data(input_df, output_df)` prints a summary showing row count
changes, added columns, removed columns, and final column list.

### Calling `generate()` directly

You can call `block.generate(df)` directly to skip validation and logging. This
is useful in tests or when you handle validation yourself, but in production
pipelines you should use `block(df)`.

## Column Handling

### Input formats

The `input_cols` and `output_cols` fields accept several formats. Pydantic
validators on `BaseBlock` normalize them automatically:

| You pass | Stored as |
|---|---|
| `None` | `[]` |
| `"text"` | `["text"]` |
| `["text", "title"]` | `["text", "title"]` |
| `{"text": "renamed_text"}` | `{"text": "renamed_text"}` |

### Validation behavior

- **Input columns**: The block checks that every key (for dicts) or element (for
  lists) in `input_cols` exists as a column in the DataFrame. Missing columns
  raise `MissingColumnError`.
- **Output columns**: The block checks that no key/element in `output_cols`
  already exists in the DataFrame. Collisions raise `OutputColumnCollisionError`.

```python
import pandas as pd
from sdg_hub.core.blocks import TextConcatBlock

df = pd.DataFrame({"a": ["hello"], "b": ["world"]})

block = TextConcatBlock(
    block_name="concat",
    input_cols=["a", "b"],
    output_cols=["combined"],
    separator=" ",
)

result = block(df)
# result has columns: a, b, combined
```

## BlockRegistry API

The `BlockRegistry` class (in `src/sdg_hub/core/blocks/registry.py`) provides
block discovery and organization. All methods are `@classmethod`.

### `@BlockRegistry.register()`

Decorator to register a block class:

```python
from sdg_hub import BaseBlock, BlockRegistry

@BlockRegistry.register(
    block_name="MyBlock",
    category="transform",
    description="Does something useful",
    deprecated=False,       # optional, default False
    replacement=None,       # optional, name of replacement block
)
class MyBlock(BaseBlock):
    ...
```

The decorator validates that the class inherits from `BaseBlock` and stores a
`BlockMetadata` dataclass with the provided information.

### `BlockRegistry.discover_blocks()`

```python
@classmethod
def discover_blocks(cls) -> None:
```

Prints a Rich-formatted table of all registered blocks to the console, sorted by
category then name. Shows block name, category, and description. Deprecated
blocks are marked with a warning indicator.

### `BlockRegistry.list_blocks()`

```python
@classmethod
def list_blocks(
    cls,
    category: Optional[str] = None,
    *,
    grouped: bool = False,
    include_deprecated: bool = True,
) -> list[str] | dict[str, list[str]]:
```

Returns registered block names. Behavior depends on arguments:

- `list_blocks()` -- flat sorted list of all block names.
- `list_blocks(category="llm")` -- list of block names in the `"llm"` category.
- `list_blocks(grouped=True)` -- dict mapping category names to sorted lists of
  block names.
- `list_blocks(include_deprecated=False)` -- excludes deprecated blocks.

### `BlockRegistry.categories()`

```python
@classmethod
def categories(cls) -> list[str]:
```

Returns a sorted list of all category names (e.g. `["agent", "filtering", "llm",
"mcp", "parsing", "transform"]`).

### `BlockMetadata` dataclass

Each registered block is stored as a `BlockMetadata` instance:

```python
@dataclass
class BlockMetadata:
    name: str                          # registered block name
    block_class: type                  # the actual class
    category: str                      # category string
    description: str = ""              # human-readable description
    deprecated: bool = False           # deprecation flag
    replacement: Optional[str] = None  # suggested replacement name
```

## Next Steps

- [LLM Blocks](llm-blocks.md) -- generate text with language models
- [Parsing Blocks](parsing-blocks.md) -- extract structured data from text
- [Transform Blocks](transform-blocks.md) -- reshape and manipulate columns
- [Filtering Blocks](filtering-blocks.md) -- filter rows by column values
- [Agent Blocks](agent-blocks.md) -- integrate external agent frameworks
- [Custom Blocks](custom-blocks.md) -- build your own blocks
