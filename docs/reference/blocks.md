# Blocks API Reference

This page provides auto-generated API documentation from source docstrings.
When reading raw markdown, refer to the quick reference below.

## Quick Reference

### Base Classes

| Class | Import | Description |
|-------|--------|-------------|
| BaseBlock | `from sdg_hub import BaseBlock` | Abstract base class for all blocks with Pydantic validation |
| BlockRegistry | `from sdg_hub import BlockRegistry` | Registry for block discovery and metadata |

### LLM Blocks

| Class | Import | Description |
|-------|--------|-------------|
| LLMChatBlock | `from sdg_hub.core.blocks import LLMChatBlock` | Unified LLM chat block supporting 100+ providers via LiteLLM |
| PromptBuilderBlock | `from sdg_hub.core.blocks import PromptBuilderBlock` | Formats prompts into structured chat messages or plain text |
| LLMResponseExtractorBlock | `from sdg_hub.core.blocks import LLMResponseExtractorBlock` | Extracts specified fields from LLM response objects |
| LLMErrorHandler | `from sdg_hub.core.blocks.llm.error_handler import LLMErrorHandler` | Centralized error handling for LLM operations across all providers |

### Parsing Blocks

| Class | Import | Description |
|-------|--------|-------------|
| TagParserBlock | `from sdg_hub.core.blocks import TagParserBlock` | Parses text content using start/end tags |
| JSONParserBlock | `from sdg_hub.core.blocks import JSONParserBlock` | Parses JSON from text and expands fields into separate columns |
| RegexParserBlock | `from sdg_hub.core.blocks import RegexParserBlock` | Parses text content using regex patterns |
| TextParserBlock | `from sdg_hub.core.blocks import TextParserBlock` | DEPRECATED: Use TagParserBlock or RegexParserBlock |

### Transform Blocks

| Class | Import | Description |
|-------|--------|-------------|
| TextConcatBlock | `from sdg_hub.core.blocks import TextConcatBlock` | Combines multiple columns into a single column using a separator |
| DuplicateColumnsBlock | `from sdg_hub.core.blocks import DuplicateColumnsBlock` | Duplicates existing columns with new names according to a mapping |
| RenameColumnsBlock | `from sdg_hub.core.blocks import RenameColumnsBlock` | Renames columns in a dataset according to a mapping |
| MeltColumnsBlock | `from sdg_hub.core.blocks import MeltColumnsBlock` | Transforms wide dataset format into long format by melting columns into rows |
| IndexBasedMapperBlock | `from sdg_hub.core.blocks import IndexBasedMapperBlock` | Maps values from source columns to output based on choice columns |
| UniformColumnValueSetter | `from sdg_hub.core.blocks import UniformColumnValueSetter` | Replaces all values in a column with a summary statistic (mode, mean, median) |
| JSONStructureBlock | `from sdg_hub.core.blocks.transform.json_structure_block import JSONStructureBlock` | Combines multiple columns into a single JSON object column |
| RowMultiplierBlock | `from sdg_hub.core.blocks.transform.row_multiplier import RowMultiplierBlock` | Duplicates each row a configurable number of times |
| SamplerBlock | `from sdg_hub.core.blocks.transform.sampler import SamplerBlock` | Randomly samples n values from a list column |

### Filtering Blocks

| Class | Import | Description |
|-------|--------|-------------|
| ColumnValueFilterBlock | `from sdg_hub.core.blocks import ColumnValueFilterBlock` | Filters datasets based on column values using comparison operations |

### Agent Blocks

| Class | Import | Description |
|-------|--------|-------------|
| AgentBlock | `from sdg_hub.core.blocks import AgentBlock` | Execute agent frameworks (Langflow, etc.) on DataFrame rows |
| AgentResponseExtractorBlock | `from sdg_hub.core.blocks.agent.agent_response_extractor_block import AgentResponseExtractorBlock` | Extracts text content from agent framework responses |

### MCP Blocks

| Class | Import | Description |
|-------|--------|-------------|
| MCPAgentBlock | `from sdg_hub.core.blocks import MCPAgentBlock` | LLM agent with remote MCP tools in an agentic loop |

---

## Detailed API (auto-generated)

### Base Classes

::: sdg_hub.core.blocks.base.BaseBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.registry.BlockRegistry
    options:
      members_order: source
      show_source: false

---

### LLM Blocks

::: sdg_hub.core.blocks.llm.llm_chat_block.LLMChatBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.llm.prompt_builder_block.PromptBuilderBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.llm.llm_response_extractor_block.LLMResponseExtractorBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.llm.error_handler.LLMErrorHandler
    options:
      members_order: source
      show_source: false

---

### Parsing Blocks

::: sdg_hub.core.blocks.parsing.tag_parser_block.TagParserBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.parsing.json_parser_block.JSONParserBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.parsing.regex_parser_block.RegexParserBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.parsing.text_parser_block.TextParserBlock
    options:
      members_order: source
      show_source: false

---

### Transform Blocks

::: sdg_hub.core.blocks.transform.text_concat.TextConcatBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.transform.duplicate_columns.DuplicateColumnsBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.transform.rename_columns.RenameColumnsBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.transform.melt_columns.MeltColumnsBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.transform.index_based_mapper.IndexBasedMapperBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.transform.uniform_col_val_setter.UniformColumnValueSetter
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.transform.json_structure_block.JSONStructureBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.transform.row_multiplier.RowMultiplierBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.transform.sampler.SamplerBlock
    options:
      members_order: source
      show_source: false

---

### Filtering Blocks

::: sdg_hub.core.blocks.filtering.column_value_filter.ColumnValueFilterBlock
    options:
      members_order: source
      show_source: false

---

### Agent Blocks

::: sdg_hub.core.blocks.agent.agent_block.AgentBlock
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.blocks.agent.agent_response_extractor_block.AgentResponseExtractorBlock
    options:
      members_order: source
      show_source: false

---

### MCP Blocks

::: sdg_hub.core.blocks.mcp.mcp_agent_block.MCPAgentBlock
    options:
      members_order: source
      show_source: false
