# SPDX-License-Identifier: Apache-2.0
"""Registry for connector classes."""

from __future__ import annotations

from typing import TYPE_CHECKING
import inspect

from ..utils.logger_config import setup_logger
from .exceptions import ConnectorError

if TYPE_CHECKING:
    from .base import BaseConnector

logger = setup_logger(__name__)


class ConnectorRegistry:
    """Global registry for connector classes.

    Simple registry for registering and retrieving connectors by name.

    Example
    -------
    >>> @ConnectorRegistry.register("my_connector")
    ... class MyConnector(BaseConnector):
    ...     pass
    ...
    >>> connector_class = ConnectorRegistry.get("my_connector")
    """

    _connectors: dict[str, type[BaseConnector]] = {}

    @classmethod
    def register(cls, name: str):
        """Register a connector class.

        Parameters
        ----------
        name : str
            Name under which to register the connector.

        Returns
        -------
        callable
            Decorator function that registers the class.

        Example
        -------
        >>> @ConnectorRegistry.register("langflow")
        ... class LangflowConnector(BaseAgentConnector):
        ...     pass
        """

        def decorator(connector_class: type[BaseConnector]) -> type[BaseConnector]:
            # Validate the class
            if not inspect.isclass(connector_class):
                raise ConnectorError(f"Expected a class, got {type(connector_class)}")

            # Check for BaseConnector inheritance
            from .base import BaseConnector

            if not issubclass(connector_class, BaseConnector):
                raise ConnectorError(
                    f"Connector class '{connector_class.__name__}' "
                    "must inherit from BaseConnector"
                )

            cls._connectors[name] = connector_class
            logger.debug(f"Registered connector '{name}' ({connector_class.__name__})")

            return connector_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseConnector]:
        """Get a connector class by name.

        Parameters
        ----------
        name : str
            Name of the connector to retrieve.

        Returns
        -------
        type[BaseConnector]
            The connector class.

        Raises
        ------
        ConnectorError
            If the connector is not found.
        """
        if name not in cls._connectors:
            available = sorted(cls._connectors.keys())
            error_msg = f"Connector '{name}' not found."
            if available:
                error_msg += f" Available: {', '.join(available)}"
            raise ConnectorError(error_msg)

        return cls._connectors[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """Get all registered connector names.

        Returns
        -------
        list[str]
            Sorted list of all connector names.
        """
        return sorted(cls._connectors.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered connectors. Primarily for testing."""
        cls._connectors.clear()
