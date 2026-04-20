# SPDX-License-Identifier: Apache-2.0
"""Connectors subsystem for external service integrations.

Example
-------
>>> from sdg_hub.core.connectors import (
...     ConnectorConfig,
...     ConnectorRegistry,
...     DEFAULT_LANGFLOW_URL,
...     LangflowConnector,
... )
>>>
>>> # Using the registry
>>> connector_class = ConnectorRegistry.get("langflow")
>>> config = ConnectorConfig(url=DEFAULT_LANGFLOW_URL)
>>> connector = connector_class(config=config)
>>>
>>> # Direct instantiation
>>> connector = LangflowConnector(config=config)
>>> response = connector.send(
...     messages=[{"role": "user", "content": "Hello!"}],
...     session_id="session-123",
... )
"""

# Import agent module to register connectors
from .agent import (
    BaseAgentConnector,
    GenericHTTPConnector,
    LangflowConnector,
    LangGraphConnector,
)
from .base import BaseConnector, ConnectorConfig
from .code_interpreter import (
    BaseCodeInterpreterConnector,
    CodeExecutionResult,
    MontyConnector,
)
from .exceptions import ConnectorError, ConnectorHTTPError
from .http import HttpClient
from .registry import ConnectorRegistry

# Default Langflow API endpoint URL for local development.
DEFAULT_LANGFLOW_URL = "http://localhost:7860/api/v1/run/flow"

# Default LangGraph API endpoint URL for local development.
DEFAULT_LANGGRAPH_URL = "http://localhost:2024"

__all__ = [
    # Base classes
    "BaseConnector",
    "ConnectorConfig",
    # Agent connectors
    "BaseAgentConnector",
    "LangflowConnector",
    "GenericHTTPConnector",
    "LangGraphConnector",
    # Code interpreter connectors
    "BaseCodeInterpreterConnector",
    "CodeExecutionResult",
    "MontyConnector",
    # Registry
    "ConnectorRegistry",
    # HTTP utilities
    "HttpClient",
    # Exceptions
    "ConnectorError",
    "ConnectorHTTPError",
    # Default URLs
    "DEFAULT_LANGFLOW_URL",
    "DEFAULT_LANGGRAPH_URL",
]
