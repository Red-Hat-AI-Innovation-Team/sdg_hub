# Block Invariants

Rules every block in SDG Hub must follow. A block is a composable processing
unit that transforms a pandas DataFrame.

## Required Structure

1. **Inherit from `BaseBlock`** (`src/sdg_hub/core/blocks/base.py`)
2. **Implement `generate()`** -- the method that receives and returns a DataFrame
3. **Use Pydantic fields** for all configuration (no bare `__init__` params)
4. **Declare `input_cols` and `output_cols`** so the framework can validate
   column presence before execution
5. **Register with `@BlockRegistry.register(name, category, description)`**
   so the block is discoverable via `BlockRegistry.discover_blocks()`
6. **Have a test file** at `tests/blocks/{category}/test_{name}.py`

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Class name | PascalCase, ends in `Block` | `LLMChatBlock` |
| File name | snake_case | `llm_chat_block.py` |
| Registration name | PascalCase (matches class) | `"LLMChatBlock"` |
| Test file | `test_` + snake_case module name | `test_llm_chat_block.py` |

## Block Categories

| Category | Directory | What belongs here |
|----------|-----------|-------------------|
| `llm` | `src/sdg_hub/core/blocks/llm/` | LLM chat, prompt building, response extraction |
| `parsing` | `src/sdg_hub/core/blocks/parsing/` | Tag, regex, and JSON parsing |
| `transform` | `src/sdg_hub/core/blocks/transform/` | Column renaming, text concat, melting, sampling, duplication |
| `filtering` | `src/sdg_hub/core/blocks/filtering/` | Row filtering by column value |
| `agent` | `src/sdg_hub/core/blocks/agent/` | Agent invocation and response extraction |
| `mcp` | `src/sdg_hub/core/blocks/mcp/` | MCP-based agentic tool-use |
| `code` | `src/sdg_hub/core/blocks/code/` | Code interpretation and execution |

## Registration Example

```python
from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry


@BlockRegistry.register(
    "MyNewBlock",
    "transform",
    "Applies a custom transformation to the dataset",
)
class MyNewBlock(BaseBlock):
    """One-line description of what this block does.

    Detailed explanation of the transformation, including any
    edge cases or assumptions about the input data.
    """

    my_param: str = Field(..., description="What this parameter controls")

    def generate(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # Implementation here
        return dataset
```

## Checklist Before Merging

- [ ] Class inherits from `BaseBlock`
- [ ] `generate()` method implemented
- [ ] All config uses Pydantic `Field()`
- [ ] `input_cols` and `output_cols` declared (if applicable)
- [ ] `@BlockRegistry.register(name, category, description)` decorator present
- [ ] Test file exists at `tests/blocks/{category}/test_{name}.py`
- [ ] Tests cover success and error cases
- [ ] Class has a docstring
