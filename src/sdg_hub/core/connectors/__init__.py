# SPDX-License-Identifier: Apache-2.0
"""Connectors subsystem for external service integrations.

This module provides a scalable, extensible system for integrating with
third-party services including agent frameworks, vector databases,
storage services, and APIs.

Architecture Philosophy:
- Connectors handle external service communication
- Blocks handle DataFrame integration
- Registry provides discovery and instantiation

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
from . import agent as agent  # noqa: F401
from .base import BaseConnector, ConnectorConfig
from .exceptions import (
    ConnectorConfigError,
    ConnectorConnectionError,
    ConnectorError,
    ConnectorHTTPError,
    ConnectorResponseError,
    ConnectorTimeoutError,
)
from .http import HttpClient
from .registry import ConnectorMetadata, ConnectorRegistry

__all__ = [
    # Base classes
    "BaseConnector",
    "ConnectorConfig",
    # Registry
    "ConnectorRegistry",
    "ConnectorMetadata",
    # HTTP utilities
    "HttpClient",
    # Exceptions
    "ConnectorError",
    "ConnectorConfigError",
    "ConnectorConnectionError",
    "ConnectorTimeoutError",
    "ConnectorHTTPError",
    "ConnectorResponseError",
]
