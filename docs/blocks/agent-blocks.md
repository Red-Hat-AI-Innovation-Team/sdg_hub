# Agent Blocks

Agent blocks integrate external agent frameworks and tool-calling LLMs into data generation pipelines. They send dataset rows to remote agents or MCP servers, collect responses, and optionally extract structured fields from those responses.

---

## AgentBlock

Executes external agent frameworks (such as Langflow or LangGraph) on each row of a DataFrame. Each row's content is sent as a message to the agent endpoint and the raw response is stored in an output column. The block connects to agent frameworks through the connector registry using the `agent_framework` parameter.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `agent_framework` | `str` | required | Connector name from the registry (e.g., `"langflow"`, `"langgraph"`) |
| `agent_url` | `str` | required | API endpoint URL for the agent |
| `agent_api_key` | `str` or `null` | `null` | API key for authentication |
| `input_cols` | `dict[str, str]` or `list[str]` | required | Input column specification. Dict form: `{"messages": "column_name"}`. List form: first element is used as the messages column. |
| `output_cols` | `list[str]` | `None` (inherited from BaseBlock) | Output column for storing raw agent responses. If not specified, `"agent_response"` is used as the fallback column name at runtime. |
| `timeout` | `float` | `120.0` | Request timeout in seconds (must be > 0) |
| `max_retries` | `int` | `3` | Maximum retry attempts (must be >= 0) |
| `session_id_col` | `str` or `null` | `null` | Column containing session IDs. If not set, UUIDs are generated per row. |
| `async_mode` | `bool` | `false` | Use async execution for better throughput with large datasets |
| `max_concurrency` | `int` | `10` | Maximum concurrent requests in async mode (must be > 0) |
| `connector_kwargs` | `dict[str, Any]` | `{}` | Extra keyword arguments passed to the connector constructor (e.g., `assistant_id` for LangGraph) |

### Input Format

The messages column accepts three formats:

- **Plain text string** -- wrapped as `[{"role": "user", "content": "..."}]`
- **Single message dict** -- wrapped in a list: `[{"role": "user", "content": "..."}]`
- **List of message dicts** -- used as-is for multi-turn conversations

### Python Example

```python
from sdg_hub.core.blocks import AgentBlock
import pandas as pd

block = AgentBlock(
    block_name="qa_agent",
    agent_framework="langflow",
    agent_url="http://localhost:7860/api/v1/run/qa-flow",
    agent_api_key="your-api-key",
    input_cols={"messages": "question"},
    output_cols=["agent_response"],
    timeout=60.0,
    max_retries=2,
)

dataset = pd.DataFrame({
    "question": [
        "What is machine learning?",
        "Explain neural networks.",
    ],
})

result = block(dataset)
# result["agent_response"] contains raw response dicts from the agent
```

### Python Example -- Async Mode

```python
from sdg_hub.core.blocks import AgentBlock
import pandas as pd

block = AgentBlock(
    block_name="batch_agent",
    agent_framework="langflow",
    agent_url="http://localhost:7860/api/v1/run/qa-flow",
    input_cols=["question"],
    output_cols=["response"],
    async_mode=True,
    max_concurrency=20,
)

dataset = pd.DataFrame({
    "question": ["Q1", "Q2", "Q3", "Q4", "Q5"],
})

result = block(dataset)
```

### YAML Example

```yaml
- block_type: "AgentBlock"
  block_config:
    block_name: "qa_agent"
    agent_framework: "langflow"
    agent_url: "http://localhost:7860/api/v1/run/qa-flow"
    agent_api_key: "${LANGFLOW_API_KEY}"
    input_cols:
      messages: "question"
    output_cols:
      - "agent_response"
    timeout: 60.0
    max_retries: 2
```

### YAML Example -- LangGraph with connector_kwargs

```yaml
- block_type: "AgentBlock"
  block_config:
    block_name: "langgraph_agent"
    agent_framework: "langgraph"
    agent_url: "http://localhost:8123"
    input_cols:
      messages: "query"
    output_cols:
      - "agent_response"
    connector_kwargs:
      assistant_id: "my-assistant"
```

---

## AgentResponseExtractorBlock

Extracts text content, session IDs, and tool traces from raw agent framework response objects. Designed to run after `AgentBlock` to parse framework-specific response structures into flat columns. Delegates parsing to the connector class registered for the specified `agent_framework`.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `agent_framework` | `str` | required | Agent framework whose response format to parse (e.g., `"langflow"`) |
| `input_cols` | `list[str]` | required | Single input column containing response objects (dict or list of dicts) |
| `output_cols` | `list[str]` | auto-derived | Automatically computed from enabled extraction fields and prefix |
| `extract_text` | `bool` | `true` | Extract text content from responses |
| `extract_session_id` | `bool` | `false` | Extract session ID from responses |
| `extract_tool_trace` | `bool` | `false` | Extract the full tool call trace (for Langflow: content_blocks with tool_use entries) |
| `expand_lists` | `bool` | `true` | Expand list inputs into individual rows (`true`) or preserve as lists (`false`) |
| `field_prefix` | `str` | `""` | Prefix for output field names. Empty default uses `block_name_` as prefix. Example: `"agent_"` produces `"agent_text"`, `"agent_session_id"`. |

At least one of `extract_text`, `extract_session_id`, or `extract_tool_trace` must be enabled.

### Python Example

```python
from sdg_hub.core.blocks import AgentResponseExtractorBlock
import pandas as pd

block = AgentResponseExtractorBlock(
    block_name="extract_response",
    agent_framework="langflow",
    input_cols=["agent_response"],
    extract_text=True,
    extract_session_id=True,
    field_prefix="lf_",
)

# Assume agent_response column contains raw Langflow response dicts
result = block(dataset)
# result now has columns: "lf_text", "lf_session_id"
```

### YAML Example

```yaml
- block_type: "AgentResponseExtractorBlock"
  block_config:
    block_name: "extract_response"
    agent_framework: "langflow"
    input_cols:
      - "agent_response"
    extract_text: true
    extract_session_id: true
    extract_tool_trace: false
    expand_lists: true
    field_prefix: "lf_"
```

### Pipeline Pattern -- AgentBlock followed by Extractor

A common pattern is to chain `AgentBlock` with `AgentResponseExtractorBlock`:

```yaml
blocks:
  - block_type: "AgentBlock"
    block_config:
      block_name: "run_agent"
      agent_framework: "langflow"
      agent_url: "http://localhost:7860/api/v1/run/qa-flow"
      input_cols:
        messages: "question"
      output_cols:
        - "raw_response"

  - block_type: "AgentResponseExtractorBlock"
    block_config:
      block_name: "extract_text"
      agent_framework: "langflow"
      input_cols:
        - "raw_response"
      extract_text: true
      field_prefix: "agent_"
```

This produces an `agent_text` column containing the extracted text from each agent response.

---

## MCPAgentBlock

Runs an agentic loop where an LLM calls tools provided by a remote MCP (Model Context Protocol) server. The block connects via streamable HTTP, fetches available tools, and iteratively calls the LLM until a final text response is generated or the iteration limit is reached.

Uses LiteLLM for LLM calls, supporting all major providers.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `mcp_server_url` | `str` | required | URL of the remote MCP server (e.g., `"https://mcp.deepwiki.com/mcp"`) |
| `mcp_headers` | `dict[str, str]` or `null` | `null` | HTTP headers for MCP server authentication |
| `model` | `str` | required | Model identifier in LiteLLM format (e.g., `"openai/gpt-4o"`) |
| `api_key` | `SecretStr` or `null` | `null` | API key for the LLM provider. Falls back to environment variables. |
| `api_base` | `str` or `null` | `null` | Base URL for the LLM API |
| `max_iterations` | `int` | `10` | Maximum number of agentic loop iterations (must be >= 1) |
| `system_prompt` | `str` or `null` | `null` | System prompt prepended to conversations |
| `input_cols` | `list[str]` | required | Exactly one input column containing queries |
| `output_cols` | `list[str]` | required | Exactly one output column for agent trace dictionaries |

### Output Format

Each output cell contains a dictionary with three keys:

| Key | Type | Description |
|-----|------|-------------|
| `messages` | `list[dict]` | Full conversation history including user, assistant, and tool messages with all tool calls and results |
| `iterations` | `int` | Number of agentic loop iterations completed |
| `max_iterations_reached` | `bool` | Whether the loop hit the iteration limit without producing a final response |

### Python Example

```python
from sdg_hub.core.blocks import MCPAgentBlock
import pandas as pd

block = MCPAgentBlock(
    block_name="research_agent",
    mcp_server_url="https://mcp.deepwiki.com/mcp",
    model="openai/gpt-4o",
    max_iterations=5,
    system_prompt="You are a helpful research assistant.",
    input_cols=["question"],
    output_cols=["agent_trace"],
)

dataset = pd.DataFrame({
    "question": [
        "What is the architecture of the Transformer model?",
        "How does BERT handle tokenization?",
    ],
})

result = block(dataset)

# Access the full trace for the first row
trace = result["agent_trace"].iloc[0]
print(trace["iterations"])           # Number of iterations completed
print(trace["max_iterations_reached"])  # False if finished normally
print(trace["messages"][-1]["content"])  # Final assistant response
```

### Python Example -- Custom MCP Server with Authentication

```python
from sdg_hub.core.blocks import MCPAgentBlock
import pandas as pd

block = MCPAgentBlock(
    block_name="internal_agent",
    mcp_server_url="https://internal-mcp.company.com/mcp",
    mcp_headers={"Authorization": "Bearer your-token"},
    model="openai/gpt-4o",
    api_key="your-openai-key",
    max_iterations=10,
    input_cols=["query"],
    output_cols=["trace"],
)
```

### YAML Example

```yaml
- block_type: "MCPAgentBlock"
  block_config:
    block_name: "research_agent"
    mcp_server_url: "https://mcp.deepwiki.com/mcp"
    model: "openai/gpt-4o"
    max_iterations: 5
    system_prompt: "You are a helpful research assistant."
    input_cols:
      - "question"
    output_cols:
      - "agent_trace"
```

### YAML Example -- With Authentication

```yaml
- block_type: "MCPAgentBlock"
  block_config:
    block_name: "internal_agent"
    mcp_server_url: "https://internal-mcp.company.com/mcp"
    mcp_headers:
      Authorization: "Bearer ${MCP_TOKEN}"
    model: "openai/gpt-4o"
    api_key: "${OPENAI_API_KEY}"
    api_base: "https://api.openai.com/v1"
    max_iterations: 10
    system_prompt: "Answer questions using available tools."
    input_cols:
      - "query"
    output_cols:
      - "trace"
```

Behavior notes:

- The block requires an async-compatible environment. In Jupyter notebooks, apply `nest_asyncio.apply()` before running.
- Tool call failures are logged as warnings and the error is passed back to the LLM as a tool result so it can recover.
- The `api_key` and `mcp_headers` fields are excluded from serialization for security.
- All MCP tools are automatically converted to OpenAI function-calling format.
