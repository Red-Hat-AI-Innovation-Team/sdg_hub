# SPDX-License-Identifier: Apache-2.0
"""Connectors subsystem for external service integrations.

Example
-------
>>> from sdg_hub.core.connectors import (
...     ConnectorConfig,
...     ConnectorRegistry,
...     LangflowConnector,
... )
>>>
>>> # Using the registry
>>> connector_class = ConnectorRegistry.get("langflow")
>>> config = ConnectorConfig(url="http://localhost:7860/api/v1/run/flow")
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
from .agent import BaseAgentConnector, LangflowConnector, LangGraphConnector
from .base import BaseConnector, ConnectorConfig
from .exceptions import ConnectorError, ConnectorHTTPError
from .http import HttpClient
from .registry import ConnectorRegistry

__all__ = [
    # Base classes
    "BaseConnector",
    "ConnectorConfig",
    # Agent connectors
    "BaseAgentConnector",
    "LangflowConnector",
    "LangGraphConnector",
    # Registry
    "ConnectorRegistry",
    # HTTP utilities
    "HttpClient",
    # Exceptions
    "ConnectorError",
    "ConnectorHTTPError",
]
