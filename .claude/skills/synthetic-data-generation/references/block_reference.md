# Block Reference

All available blocks organized by category. Use `BlockRegistry.discover_blocks()` to verify the latest list.

## Table of Contents

- [LLM Blocks](#llm-blocks) -- LLMChatBlock, PromptBuilderBlock, LLMResponseExtractorBlock
- [Parsing Blocks](#parsing-blocks) -- TagParserBlock, RegexParserBlock, JSONParserBlock
- [Transform Blocks](#transform-blocks) -- TextConcatBlock, RenameColumnsBlock, DuplicateColumnsBlock, MeltColumnsBlock, JSONStructureBlock, RowMultiplierBlock, SamplerBlock, IndexBasedMapperBlock, UniformColumnValueSetter
- [Filtering Blocks](#filtering-blocks) -- ColumnValueFilterBlock
- [Agent Blocks](#agent-blocks) -- AgentBlock, AgentResponseExtractorBlock
- [MCP Blocks](#mcp-blocks) -- MCPAgentBlock

---

## LLM Blocks

### LLMChatBlock

Call LLM APIs (100+ providers via LiteLLM).

```yaml
- block_type: "LLMChatBlock"
  block_config:
    block_name: "generate"
    input_cols: "messages"
    output_cols: "response"
    temperature: 0.7
    max_tokens: 1024
    top_p: 1.0
    async_mode: true
    # model/api_key set via flow.set_model_config() or directly:
    # model: "openai/gpt-4o-mini"
    # api_key: "sk-..."
```

**Input format:** Column must contain list of message dicts:
```python
[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
```

### PromptBuilderBlock

Build chat messages from Jinja templates. Supports both structured (YAML) and plain text templates.

```yaml
- block_type: "PromptBuilderBlock"
  block_config:
    block_name: "build_prompt"
    input_cols: ["document", "query"]
    output_cols: "messages"
    prompt_config_path: "prompts/template.yaml"  # Relative to flow.yaml
```

**Prompt template format (YAML):**
```yaml
# prompts/template.yaml
- role: system
  content: |
    You are a helpful assistant.

- role: user
  content: |
    Document: {document}
    Query: {query}
```

### LLMResponseExtractorBlock

Extract fields from LLM response objects (tool calls, content, metadata).

```yaml
- block_type: "LLMResponseExtractorBlock"
  block_config:
    block_name: "extract"
    input_cols: "raw_response"
    output_cols: "content"
    extract_content: true
    field_prefix: "result_"
```

---

## Parsing Blocks

### TagParserBlock

Parse text using start/end tag pairs. Preferred for structured LLM outputs.

```yaml
- block_type: "TagParserBlock"
  block_config:
    block_name: "parse_tags"
    input_cols: "response"
    output_cols: ["question", "answer"]
    start_tags: ["<question>", "<answer>"]
    end_tags: ["</question>", "</answer>"]
```

### RegexParserBlock

Parse text using regex patterns with capture groups.

```yaml
- block_type: "RegexParserBlock"
  block_config:
    block_name: "parse_regex"
    input_cols: "response"
    output_cols: ["question", "answer"]
    pattern: "Question:\\s*(.+?)\\s*Answer:\\s*(.+)"
    flags: "DOTALL"    # DOTALL, IGNORECASE, MULTILINE
```

### JSONParserBlock

Parse JSON from text and expand fields into columns.

```yaml
- block_type: "JSONParserBlock"
  block_config:
    block_name: "parse_json"
    input_cols: "response"
    output_cols: ["name", "score"]
```

---

## Transform Blocks

### TextConcatBlock

Concatenate multiple columns into one.

```yaml
- block_type: "TextConcatBlock"
  block_config:
    block_name: "concat"
    input_cols: ["title", "body"]
    output_cols: "full_text"
    separator: "\n\n"
```

### RenameColumnsBlock

Rename columns in-place.

```yaml
- block_type: "RenameColumnsBlock"
  block_config:
    block_name: "rename"
    input_cols:
      question: generated_question
      response: generated_answer
```

### DuplicateColumnsBlock

Copy columns with new names.

```yaml
- block_type: "DuplicateColumnsBlock"
  block_config:
    block_name: "dup"
    input_cols: "document"
    output_cols: "base_document"
```

### MeltColumnsBlock

Reshape wide format to long format (one row per value).

```yaml
- block_type: "MeltColumnsBlock"
  block_config:
    block_name: "melt"
    input_cols: ["summary_a", "summary_b"]
    output_cols: "summary"
    id_vars: ["document", "domain"]
```

Before: `| doc | sum_a | sum_b |` -> After: `| doc | summary |` (2 rows per doc)

### JSONStructureBlock

Build JSON objects from column values.

```yaml
- block_type: "JSONStructureBlock"
  block_config:
    block_name: "build_json"
    input_cols: ["question", "answer"]
    output_cols: "qa_pair"
    structure:
      q: "{question}"
      a: "{answer}"
```

### RowMultiplierBlock

Duplicate each row N times.

```yaml
- block_type: "RowMultiplierBlock"
  block_config:
    block_name: "multiply"
    input_cols: "document"
    output_cols: "document"
    n: 3
```

### SamplerBlock

Randomly sample N values from a list column.

```yaml
- block_type: "SamplerBlock"
  block_config:
    block_name: "sample"
    input_cols: "items"
    output_cols: "sampled_items"
    n: 5
```

### IndexBasedMapperBlock

Map values using a lookup dictionary.

```yaml
- block_type: "IndexBasedMapperBlock"
  block_config:
    block_name: "map"
    input_cols: "label_idx"
    output_cols: "label"
    mapping:
      0: "negative"
      1: "neutral"
      2: "positive"
```

### UniformColumnValueSetter

Set all values in a column to a single value or summary statistic (mode, mean, median).

```yaml
- block_type: "UniformColumnValueSetter"
  block_config:
    block_name: "set_source"
    output_cols: "source"
    value: "generated"
```

---

## Filtering Blocks

### ColumnValueFilterBlock

Filter rows by column values.

```yaml
- block_type: "ColumnValueFilterBlock"
  block_config:
    block_name: "filter"
    input_cols: "score"
    filter_value: [4, 5]
    operation: "in"
    convert_dtype: "int"
```

**Operations:** `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `in`, `contains`

---

## Agent Blocks

### AgentBlock

Execute external agent frameworks (Langflow, LangGraph) on each row.

```yaml
- block_type: "AgentBlock"
  block_config:
    block_name: "agent"
    input_cols: ["question"]
    output_cols: ["agent_response"]
    agent_framework: "langflow"      # or "langgraph"
    agent_url: "http://localhost:7860/api/v1/run/my-flow"
    agent_api_key: "your-key"
    extract_response: true
```

Agent config can also be set at runtime via `flow.set_agent_config()`.

**Supported connectors:**
- `langflow` -- Langflow visual LLM app builder
- `langgraph` -- LangGraph stateful multi-actor agents (supports `assistant_id` and `run_config`)

### AgentResponseExtractorBlock

Extract structured data from agent responses.

```yaml
- block_type: "AgentResponseExtractorBlock"
  block_config:
    block_name: "extract_agent"
    input_cols: "agent_response"
    output_cols: ["text", "tool_trace"]
    extract_tool_trace: true
    field_prefix: "agent_"
```

---

## MCP Blocks

### MCPAgentBlock

LLM agent with remote MCP tools in an agentic loop. The LLM calls tools from the MCP server iteratively, producing complete conversation traces including all tool calls and results.

```yaml
- block_type: "MCPAgentBlock"
  block_config:
    block_name: "mcp_agent"
    input_cols: "messages"
    output_cols: "agent_trace"
    mcp_server_url: "http://localhost:3000/mcp"
    max_iterations: 10
```

Output includes the full agent trace: messages, tool calls, and tool results -- useful for generating tool-use training data.

---

## Discovering Blocks at Runtime

```python
from sdg_hub.core.blocks import BlockRegistry

# Rich table of all blocks
BlockRegistry.discover_blocks()

# List by category
BlockRegistry.list_blocks(category="llm")
BlockRegistry.list_blocks(category="transform")
BlockRegistry.list_blocks(category="parsing")
BlockRegistry.list_blocks(category="filtering")
BlockRegistry.list_blocks(category="agent")
BlockRegistry.list_blocks(category="mcp")

# All categories
BlockRegistry.categories()

# Grouped view
BlockRegistry.list_blocks(grouped=True)
```
