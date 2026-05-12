# Connector Invariants

Rules every connector in SDG Hub must follow. Connectors integrate with
external services -- agent frameworks and code interpreters.

## Connector Types

SDG Hub has two connector families, each with its own base class.

### Agent Connectors

For integrating with agent frameworks (Langflow, LangGraph, etc.).

| Requirement | Detail |
|-------------|--------|
| Base class | `BaseAgentConnector` (`src/sdg_hub/core/connectors/agent/base.py`) |
| Implement | `build_request()` -- convert `ChatAgentRequest` to framework-specific format |
| Implement | `parse_response()` -- convert framework response to `ChatAgentResponse` |
| Register | `@ConnectorRegistry.register("name")` |
| Test file | `tests/connectors/agent/test_{name}.py` |

### Code Interpreter Connectors

For executing code in sandboxed environments.

| Requirement | Detail |
|-------------|--------|
| Base class | `BaseCodeInterpreterConnector` (`src/sdg_hub/core/connectors/code_interpreter/base.py`) |
| Implement | `execute_code()` -- execute code and return `CodeExecutionResult` |
| Register | `@ConnectorRegistry.register("name")` |
| Test file | `tests/connectors/code_interpreter/test_{name}.py` |

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Class name | PascalCase, ends in `Connector` | `LangflowConnector` |
| File name | snake_case | `langflow.py` |
| Registration name | snake_case | `"langflow"` |
| Test file | `test_` + file name | `test_langflow.py` |

## Existing Connectors

| Name | Type | Class | File |
|------|------|-------|------|
| `langflow` | agent | `LangflowConnector` | `src/sdg_hub/core/connectors/agent/langflow.py` |
| `langgraph` | agent | `LangGraphConnector` | `src/sdg_hub/core/connectors/agent/langgraph.py` |
| `generic_http` | agent | `GenericHTTPConnector` | `src/sdg_hub/core/connectors/agent/generic_http.py` |
| `monty` | code_interpreter | `MontyConnector` | `src/sdg_hub/core/connectors/code_interpreter/monty.py` |

## Registration Example

```python
from sdg_hub.core.connectors.agent.base import BaseAgentConnector
from sdg_hub.core.connectors.registry import ConnectorRegistry


@ConnectorRegistry.register("my_agent")
class MyAgentConnector(BaseAgentConnector):
    """Connects to the My Agent framework."""

    def build_request(self, request: ChatAgentRequest) -> dict:
        # Convert to framework-specific format
        ...

    def parse_response(self, response: dict) -> ChatAgentResponse:
        # Convert back to standard format
        ...
```

## Checklist Before Merging

- [ ] Class inherits from the correct base (`BaseAgentConnector` or `BaseCodeInterpreterConnector`)
- [ ] Required abstract methods implemented
- [ ] `@ConnectorRegistry.register("name")` decorator present
- [ ] Test file exists at `tests/connectors/{type}/test_{name}.py`
- [ ] Tests cover success and error cases — see [testing-standards.md](testing-standards.md)
- [ ] Class has a docstring
