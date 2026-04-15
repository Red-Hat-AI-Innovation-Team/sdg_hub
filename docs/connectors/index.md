# Connector System

The connector system provides a pluggable interface for communicating with external agent frameworks. Connectors handle request formatting, HTTP transport, response parsing, and error handling for each supported framework.

Connectors are used by `AgentBlock` and `AgentResponseExtractorBlock` to integrate agent frameworks (Langflow, LangGraph, etc.) into data generation pipelines.

## Architecture Overview

The connector system has four layers:

```
ConnectorRegistry          # Discovery and lookup by name
    |
BaseConnector              # Abstract base with execute() / aexecute()
    |
BaseAgentConnector         # Agent-specific: build_request(), parse_response(), send()
    |
LangflowConnector          # Framework-specific implementation
LangGraphConnector
```

All classes are importable from the top-level connectors package:

```python
from sdg_hub.core.connectors import (
    BaseConnector,
    ConnectorConfig,
    BaseAgentConnector,
    LangflowConnector,
    LangGraphConnector,
    ConnectorRegistry,
    HttpClient,
    ConnectorError,
    ConnectorHTTPError,
)
```

---

## ConnectorConfig

`ConnectorConfig` is a Pydantic `BaseModel` that holds connection parameters shared by all connectors. It is defined in `src/sdg_hub/core/connectors/base.py`.

```python
from sdg_hub.core.connectors import ConnectorConfig

config = ConnectorConfig(
    url="http://localhost:7860/api/v1/run/my-flow",
    api_key="your-api-key",
    timeout=60.0,
    max_retries=5,
)
```

### Fields

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `Optional[str]` | `None` | Base URL for the external service. |
| `api_key` | `Optional[str]` | `None` | API key for authentication. |
| `timeout` | `float` | `120.0` | Request timeout in seconds. Must be greater than 0. |
| `max_retries` | `int` | `3` | Maximum number of retry attempts. Must be >= 0. |

`ConnectorConfig` uses `model_config = ConfigDict(extra="allow")`, so additional framework-specific fields can be passed without raising validation errors.

---

## BaseConnector

`BaseConnector` is the abstract base class for all connectors. It is defined in `src/sdg_hub/core/connectors/base.py`.

It inherits from both `pydantic.BaseModel` and `abc.ABC`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | `ConnectorConfig` | Required. The connector configuration. |

### Abstract Methods

```python
@abstractmethod
def execute(self, request: Any) -> Any:
    """Execute a synchronous request.

    Parameters
    ----------
    request : Any
        The request to execute (format depends on connector type).

    Returns
    -------
    Any
        The response from the external service.
    """
```

### Concrete Methods

```python
async def aexecute(self, request: Any) -> Any:
    """Execute an asynchronous request.

    Default implementation wraps sync execute in a thread via asyncio.to_thread().
    Subclasses should override for true async support.

    Parameters
    ----------
    request : Any
        The request to execute.

    Returns
    -------
    Any
        The response from the external service.
    """
```

---

## ConnectorRegistry

`ConnectorRegistry` is a class-level registry for connector classes. It is defined in `src/sdg_hub/core/connectors/registry.py`.

All methods are `@classmethod`.

### API

#### `register(name: str)` -- Decorator

Registers a connector class under the given name. The decorated class must inherit from `BaseConnector`.

```python
from sdg_hub.core.connectors import ConnectorRegistry, BaseConnector

@ConnectorRegistry.register("my_connector")
class MyConnector(BaseConnector):
    def execute(self, request):
        return {"result": request.get("input")}
```

Raises `ConnectorError` if:

- The argument is not a class.
- The class does not inherit from `BaseConnector`.

#### `get(name: str) -> type`

Returns the connector class registered under `name`.

```python
connector_class = ConnectorRegistry.get("langflow")
config = ConnectorConfig(url="http://localhost:7860/api/v1/run/flow")
connector = connector_class(config=config)
```

Raises `ConnectorError` if the name is not found. The error message includes a list of available connectors.

#### `list_all() -> list[str]`

Returns a sorted list of all registered connector names.

```python
available = ConnectorRegistry.list_all()
# ['langflow', 'langgraph']
```

#### `clear() -> None`

Clears all registered connectors. Intended for testing only.

```python
ConnectorRegistry.clear()
```

---

## BaseAgentConnector

`BaseAgentConnector` extends `BaseConnector` with an agent-specific interface for communicating with agent frameworks. It is defined in `src/sdg_hub/core/connectors/agent/base.py`.

It uses an async-first pattern: the core logic lives in `_send_async()`, and the synchronous `send()` method delegates to it.

### Abstract Methods

Subclasses must implement these two methods:

```python
@abstractmethod
def build_request(
    self,
    messages: list[dict[str, Any]],
    session_id: str,
) -> dict[str, Any]:
    """Build framework-specific request payload.

    Parameters
    ----------
    messages : list[dict]
        List of messages in standard format:
        [{"role": "user", "content": "Hello"}, ...]
    session_id : str
        Session identifier for conversation tracking.

    Returns
    -------
    dict
        Framework-specific request payload.
    """
```

```python
@abstractmethod
def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate framework response.

    Parameters
    ----------
    response : dict
        Raw response from the framework.

    Returns
    -------
    dict
        Validated response dict.

    Raises
    ------
    ConnectorError
        If the response is invalid or cannot be parsed.
    """
```

### Concrete Methods

#### `send(messages, session_id, async_mode=False)`

Send messages to the agent. This is the primary entry point.

```python
def send(
    self,
    messages: list[dict[str, Any]],
    session_id: str,
    async_mode: bool = False,
) -> dict | Coroutine:
    """Send messages to the agent.

    Parameters
    ----------
    messages : list[dict]
        Messages to send, in format:
        [{"role": "user", "content": "Hello"}, ...]
    session_id : str
        Session identifier for conversation tracking.
    async_mode : bool, optional
        If True, returns a coroutine. If False (default), runs synchronously.

    Returns
    -------
    dict or Coroutine[dict]
        Response dict, or coroutine if async_mode=True.
    """
```

When `async_mode=False` (default), this method handles event loop detection automatically:

- If already in an async context, it uses a `ThreadPoolExecutor` to avoid blocking.
- If no event loop is running, it creates one with `asyncio.run()`.

#### `asend(messages, session_id)`

Async convenience wrapper that directly awaits the internal `_send_async()`.

```python
async def asend(
    self,
    messages: list[dict[str, Any]],
    session_id: str,
) -> dict[str, Any]:
    """Async send - convenience wrapper.

    Parameters
    ----------
    messages : list[dict]
        Messages to send.
    session_id : str
        Session identifier.

    Returns
    -------
    dict
        Response from the agent.
    """
```

#### `execute(request)`

Implements the `BaseConnector.execute()` interface by delegating to `send()`.

```python
def execute(self, request: dict[str, Any]) -> dict[str, Any]:
    """Execute a request (BaseConnector interface).

    Parameters
    ----------
    request : dict
        Request containing 'messages' and 'session_id' keys.

    Returns
    -------
    dict
        Response from the agent.
    """
```

### Response Field Extraction Class Methods

These class methods allow extracting fields from framework-specific responses without instantiating a connector. They are used by `AgentResponseExtractorBlock`.

The base class returns `None` for all extractors. Subclasses override to provide framework-specific parsing.

```python
@classmethod
def extract_text(cls, response: dict[str, Any]) -> str | None:
    """Extract text content from a framework response."""

@classmethod
def extract_session_id(cls, response: dict[str, Any]) -> str | None:
    """Extract session ID from a framework response."""

@classmethod
def extract_tool_trace(cls, response: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract tool call trace from a framework response."""
```

### Internal Methods

#### `_build_headers() -> dict[str, str]`

Builds HTTP headers. The base implementation sets `Content-Type: application/json` and adds `Authorization: Bearer {api_key}` if an API key is configured. Subclasses override this for framework-specific authentication headers.

#### `_get_http_client() -> HttpClient`

Lazily creates and caches an `HttpClient` instance using `config.timeout` and `config.max_retries`.

---

## Built-in Connectors

### LangflowConnector

Registered name: `"langflow"`

Source: `src/sdg_hub/core/connectors/agent/langflow.py`

Connector for [Langflow](https://github.com/langflow-ai/langflow), a visual framework for building LLM-powered applications.

#### How It Works

Langflow expects a single string input (not a message array). The connector extracts the last user message from the standard message list and sends it as `input_value`.

**Request format** (sent to Langflow API):

```json
{
    "output_type": "chat",
    "input_type": "chat",
    "input_value": "the last user message content",
    "session_id": "session-123"
}
```

**Authentication**: Uses `x-api-key` header (not `Authorization: Bearer`). The `_build_headers()` method is overridden to set this.

**Response parsing**: Returns the raw response dict unchanged after validating it is a dict.

#### Configuration Example

```python
from sdg_hub.core.connectors import ConnectorConfig, LangflowConnector

config = ConnectorConfig(
    url="http://localhost:7860/api/v1/run/my-flow",
    api_key="your-api-key",
)
connector = LangflowConnector(config=config)
response = connector.send(
    messages=[{"role": "user", "content": "Hello!"}],
    session_id="session-123",
)
```

#### Response Field Extraction

The `LangflowConnector` overrides all three extraction class methods:

- **`extract_text(response)`**: Navigates `outputs[0].outputs[0].results.message.text`. Returns `""` if the text field is explicitly `None`.
- **`extract_session_id(response)`**: Reads `response["session_id"]`. Returns `""` if explicitly `None`.
- **`extract_tool_trace(response)`**: Looks for content blocks at two paths:
    - `outputs[0].outputs[0].results.message.data.content_blocks`
    - `outputs[0].outputs[0].results.message.content_blocks`

    Returns the `contents` list from the first matching content block (the "Agent Steps" block with structured `tool_use` entries).

---

### LangGraphConnector

Registered name: `"langgraph"`

Source: `src/sdg_hub/core/connectors/agent/langgraph.py`

Connector for [LangGraph](https://github.com/langchain-ai/langgraph), a framework for building stateful, multi-actor applications with LLMs.

#### How It Works

LangGraphConnector uses a two-step thread-based flow by overriding `_send_async()`:

1. **Create a thread**: `POST {base_url}/threads` with `session_id` in metadata.
2. **Run the agent**: `POST {base_url}/threads/{thread_id}/runs/wait` with the message payload.

Each call creates a new LangGraph thread. The `session_id` is stored as thread metadata for traceability but does not cause thread reuse.

**Run request format** (sent to `/threads/{thread_id}/runs/wait`):

```json
{
    "assistant_id": "agent",
    "input": {
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    },
    "config": {}
}
```

The `config` key is only included when `run_config` is non-empty.

**Authentication**: Uses `x-api-key` header, same as Langflow.

**Response parsing**: Validates the response is a non-empty dict. Logs a warning if no `messages` key is present.

#### Additional Fields

`LangGraphConnector` defines two extra Pydantic fields beyond those inherited from `BaseAgentConnector`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `assistant_id` | `str` | `"agent"` | The assistant ID or graph name to run. Must be non-empty. |
| `run_config` | `dict[str, Any]` | `{}` | Optional configuration dict passed as the `config` key in the run payload. Use for runtime parameters via `configurable`, e.g., `{"configurable": {"model": "gpt-4o"}}`. |

#### Configuration Example

```python
from sdg_hub.core.connectors import ConnectorConfig, LangGraphConnector

config = ConnectorConfig(
    url="http://localhost:2024",
    api_key="your-api-key",
)
connector = LangGraphConnector(
    config=config,
    assistant_id="my-agent",
    run_config={"configurable": {"model": "gpt-4o"}},
)
response = connector.send(
    messages=[{"role": "user", "content": "Hello!"}],
    session_id="session-123",
)
```

#### Response Field Extraction

- **`extract_text(response)`**: Finds the last message with `type` or `role` equal to `"ai"` or `"assistant"` in `response["messages"]` and returns its `content`. Returns `""` if content is explicitly `None`.
- **`extract_session_id(response)`**: Always returns `None`. LangGraph uses thread-based state with no top-level session ID in the run response.
- **`extract_tool_trace(response)`**: Iterates through `response["messages"]` and collects:
    - AI messages with `tool_calls` as `{"type": "tool_use", "tool_calls": [...]}`.
    - Tool result messages as `{"type": "tool_result", "name": "...", "content": "...", "tool_call_id": "..."}`.

---

## Integration with AgentBlock

`AgentBlock` is the primary consumer of connectors. The `agent_framework` parameter selects which connector to use via `ConnectorRegistry.get()`.

Source: `src/sdg_hub/core/blocks/agent/agent_block.py`

### How AgentBlock Uses Connectors

```python
from sdg_hub.core.blocks import AgentBlock

block = AgentBlock(
    block_name="qa_agent",
    agent_framework="langflow",
    agent_url="http://localhost:7860/api/v1/run/qa-flow",
    agent_api_key="your-api-key",
    input_cols={"messages": "question"},
    output_cols=["response"],
)
result_df = block.generate(df)
```

Internally, `AgentBlock._get_connector()` does the following:

1. Calls `ConnectorRegistry.get(self.agent_framework)` to get the connector class.
2. Creates a `ConnectorConfig` from `agent_url`, `agent_api_key`, `timeout`, and `max_retries`.
3. Instantiates the connector class with the config and any `connector_kwargs`.
4. Caches the connector instance, invalidating on config changes.

The `connector_kwargs` field allows passing framework-specific parameters to the connector constructor. For example, to set `assistant_id` for LangGraph:

```python
block = AgentBlock(
    block_name="graph_agent",
    agent_framework="langgraph",
    agent_url="http://localhost:2024",
    input_cols={"messages": "question"},
    output_cols=["response"],
    connector_kwargs={"assistant_id": "my-agent"},
)
```

### YAML Configuration

```yaml
- block_type: AgentBlock
  block_config:
    block_name: my_agent
    agent_framework: langflow
    agent_url: http://localhost:7860/api/v1/run/my-flow
    agent_api_key: ${LANGFLOW_API_KEY}
    input_cols:
      messages: messages_col
    output_cols:
      - agent_response
```

For LangGraph with extra connector parameters:

```yaml
- block_type: AgentBlock
  block_config:
    block_name: graph_agent
    agent_framework: langgraph
    agent_url: http://localhost:2024
    agent_api_key: ${LANGGRAPH_API_KEY}
    connector_kwargs:
      assistant_id: my-agent
      run_config:
        configurable:
          model: gpt-4o
    input_cols:
      messages: question
    output_cols:
      - agent_response
```

---

## Runtime Configuration with `flow.set_agent_config()`

When using flows, agent blocks can be configured at runtime using `flow.set_agent_config()`. This supports credential-free YAML definitions where URLs and API keys are injected at runtime rather than hardcoded.

Source: `src/sdg_hub/core/flow/agent_config.py`

```python
def set_agent_config(
    flow: "Flow",
    agent_framework: Optional[str] = None,
    agent_url: Optional[str] = None,
    agent_api_key: Optional[str] = None,
    blocks: Optional[list[str]] = None,
    **kwargs: Any,
) -> None:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_framework` | `Optional[str]` | `None` | Connector name (e.g., `"langflow"`). |
| `agent_url` | `Optional[str]` | `None` | Agent API endpoint URL. |
| `agent_api_key` | `Optional[str]` | `None` | Agent API key. |
| `blocks` | `Optional[list[str]]` | `None` | Specific block names to target. If `None`, auto-detects all agent blocks. |
| `**kwargs` | `Any` | -- | Additional parameters (e.g., `timeout`, `max_retries`). |

### Usage

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")

# Configure all agent blocks at once
flow.set_agent_config(
    agent_framework="langflow",
    agent_url="http://localhost:7860/api/v1/run/my-flow",
    agent_api_key="your_key",
)

# Or target specific blocks
flow.set_agent_config(
    agent_url="http://other-endpoint:7860/api/v1/run/other-flow",
    blocks=["my_agent_block"],
)

result = flow.generate(dataset)
```

The method auto-detects blocks with `block_type == "agent"` when no `blocks` list is provided. It sets attributes directly on the block instances, and the cached connector in `AgentBlock._get_connector()` is invalidated on the next call because the config key changes.

### Related Helper Functions

These are defined in `src/sdg_hub/core/flow/agent_config.py`:

| Function | Signature | Description |
|----------|-----------|-------------|
| `detect_agent_blocks` | `(flow: Flow) -> list[str]` | Returns block names where `block_type == "agent"`. |
| `is_agent_config_required` | `(flow: Flow) -> bool` | Returns `True` if the flow contains any agent blocks. |
| `is_agent_config_set` | `(flow: Flow) -> bool` | Returns `True` if agent config has been set or is not required. |
| `reset_agent_config` | `(flow: Flow) -> None` | Resets the agent config flag. Call `set_agent_config()` again before `generate()`. |

---

## HttpClient

`HttpClient` provides HTTP transport with automatic retries using tenacity. It is defined in `src/sdg_hub/core/connectors/http/client.py`.

```python
from sdg_hub.core.connectors import HttpClient

client = HttpClient(timeout=60.0, max_retries=3)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `float` | `120.0` | Request timeout in seconds. |
| `max_retries` | `int` | `3` | Maximum number of retry attempts. |

### Methods

#### `async post(url, payload, headers=None) -> dict`

Async POST request with exponential backoff retry on `httpx.TimeoutException` and `httpx.ConnectError`.

```python
async def post(
    self,
    url: str,
    payload: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
```

#### `post_sync(url, payload, headers=None) -> dict`

Synchronous POST request with the same retry logic.

```python
def post_sync(
    self,
    url: str,
    payload: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
```

Both methods raise:

- `ConnectorHTTPError` for HTTP error status codes (includes the URL, status code, and first 500 characters of the response body).
- `ConnectorError` for connection failures and timeouts.

Retry configuration: `stop_after_attempt(max_retries + 1)` with `wait_exponential(multiplier=1, min=1, max=60)`.

---

## Exceptions

Connector exceptions are defined in `src/sdg_hub/core/connectors/exceptions.py`. Both inherit from `SDGHubError` (the project-wide base exception).

### ConnectorError

Base exception for all connector-related errors: configuration errors, connection failures, timeouts, and response parsing errors.

```python
from sdg_hub.core.connectors import ConnectorError

try:
    connector = ConnectorRegistry.get("nonexistent")
except ConnectorError as e:
    print(e)  # "Connector 'nonexistent' not found. Available: langflow, langgraph"
```

### ConnectorHTTPError

Raised when an HTTP request returns an error status code. Subclass of `ConnectorError`.

```python
from sdg_hub.core.connectors import ConnectorHTTPError

class ConnectorHTTPError(ConnectorError):
    def __init__(self, url: str, status_code: int, message: Optional[str] = None):
        self.url = url
        self.status_code = status_code
```

The error message format is `HTTP {status_code} error from '{url}': {message}`.

---

## Creating Custom Connectors

To add a new agent framework connector:

1. Create a new file in `src/sdg_hub/core/connectors/agent/`.
2. Inherit from `BaseAgentConnector`.
3. Implement `build_request()` and `parse_response()`.
4. Register with `@ConnectorRegistry.register("name")`.
5. Optionally override `_build_headers()` for custom authentication.
6. Optionally override the `extract_*` class methods for response field extraction.

### Complete Example

```python
# src/sdg_hub/core/connectors/agent/my_framework.py

from typing import Any

from sdg_hub.core.connectors import BaseAgentConnector, ConnectorError, ConnectorRegistry


@ConnectorRegistry.register("my_framework")
class MyFrameworkConnector(BaseAgentConnector):
    """Connector for MyFramework agent API."""

    def _build_headers(self) -> dict[str, str]:
        """MyFramework uses Bearer token authentication."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def build_request(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        """Convert standard messages to MyFramework format."""
        return {
            "conversation_id": session_id,
            "messages": [
                {"sender": msg["role"], "text": msg["content"]}
                for msg in messages
            ],
        }

    def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Validate and return the response."""
        if not isinstance(response, dict):
            raise ConnectorError(
                f"Expected dict response, got {type(response).__name__}"
            )
        return response

    @classmethod
    def extract_text(cls, response: dict[str, Any]) -> str | None:
        """Extract text from MyFramework response."""
        return response.get("reply", {}).get("text")

    @classmethod
    def extract_session_id(cls, response: dict[str, Any]) -> str | None:
        """Extract session ID from MyFramework response."""
        return response.get("conversation_id")
```

After creating the file, add the import to `src/sdg_hub/core/connectors/agent/__init__.py`:

```python
from .my_framework import MyFrameworkConnector
```

The connector is then available via the registry:

```python
from sdg_hub.core.connectors import ConnectorRegistry

connector_class = ConnectorRegistry.get("my_framework")
```

And usable in `AgentBlock`:

```python
from sdg_hub.core.blocks import AgentBlock

block = AgentBlock(
    block_name="my_agent",
    agent_framework="my_framework",
    agent_url="http://localhost:8080/api/chat",
    agent_api_key="your-api-key",
    input_cols={"messages": "question"},
    output_cols=["response"],
)
```
