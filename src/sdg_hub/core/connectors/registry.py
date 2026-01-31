# SPDX-License-Identifier: Apache-2.0
"""Registry for connector classes with metadata and discovery."""

from dataclasses import dataclass
from difflib import get_close_matches
import inspect

from rich.console import Console
from rich.table import Table

from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)
console = Console()


@dataclass
class ConnectorMetadata:
    """Metadata for registered connectors.

    Parameters
    ----------
    name : str
        The registered name of the connector.
    connector_class : type
        The actual connector class.
    category : str
        Category for organization (e.g., 'agent', 'retrieval', 'storage').
    description : str, optional
        Human-readable description of what the connector does.
    supports_async : bool
        Whether the connector supports async operations.
    supports_streaming : bool
        Whether the connector supports streaming responses.
    supports_batch : bool
        Whether the connector supports batch operations.
    """

    name: str
    connector_class: type
    category: str
    description: str = ""
    supports_async: bool = False
    supports_streaming: bool = False
    supports_batch: bool = False

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if not self.name:
            raise ValueError("Connector name cannot be empty")
        if not inspect.isclass(self.connector_class):
            raise ValueError("connector_class must be a class")


class ConnectorRegistry:
    """Global registry for all connector types.

    This registry provides a centralized location for registering and
    discovering connectors. It supports categorization, metadata,
    and helpful error messages.

    Example
    -------
    >>> @ConnectorRegistry.register(
    ...     "my_connector",
    ...     category="agent",
    ...     description="My custom connector"
    ... )
    ... class MyConnector(BaseConnector):
    ...     pass
    ...
    >>> connector_class = ConnectorRegistry.get("my_connector")
    """

    _metadata: dict[str, ConnectorMetadata] = {}
    _by_category: dict[str, set[str]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        category: str,
        description: str = "",
        supports_async: bool = False,
        supports_streaming: bool = False,
        supports_batch: bool = False,
    ):
        """Register a connector class with metadata.

        Parameters
        ----------
        name : str
            Name under which to register the connector.
        category : str
            Category for organization (e.g., 'agent', 'retrieval', 'storage').
        description : str, optional
            Human-readable description of the connector.
        supports_async : bool, optional
            Whether the connector supports async operations.
        supports_streaming : bool, optional
            Whether the connector supports streaming responses.
        supports_batch : bool, optional
            Whether the connector supports batch operations.

        Returns
        -------
        callable
            Decorator function that registers the class.

        Example
        -------
        >>> @ConnectorRegistry.register("langflow", category="agent")
        ... class LangflowConnector(BaseAgentConnector):
        ...     pass
        """

        def decorator(connector_class: type) -> type:
            # Validate the class
            cls._validate_connector_class(connector_class)

            # Create metadata
            metadata = ConnectorMetadata(
                name=name,
                connector_class=connector_class,
                category=category,
                description=description,
                supports_async=supports_async,
                supports_streaming=supports_streaming,
                supports_batch=supports_batch,
            )

            # Register the metadata
            cls._metadata[name] = metadata

            # Update category index
            cls._by_category.setdefault(category, set()).add(name)

            logger.debug(
                f"Registered connector '{name}' "
                f"({connector_class.__name__}) in category '{category}'"
            )

            return connector_class

        return decorator

    @classmethod
    def _validate_connector_class(cls, connector_class: type) -> None:
        """Validate that a class is a proper connector class.

        Parameters
        ----------
        connector_class : type
            The class to validate.

        Raises
        ------
        ValueError
            If the class is not a valid connector class.
        """
        if not inspect.isclass(connector_class):
            raise ValueError(f"Expected a class, got {type(connector_class)}")

        # Check for BaseConnector inheritance
        try:
            from .base import BaseConnector

            if not issubclass(connector_class, BaseConnector):
                raise ValueError(
                    f"Connector class '{connector_class.__name__}' "
                    "must inherit from BaseConnector"
                )
        except ImportError:
            # BaseConnector not available, check for execute method
            if not hasattr(connector_class, "execute"):
                raise ValueError(
                    f"Connector class '{connector_class.__name__}' "
                    "must implement 'execute' method"
                )

    @classmethod
    def get(cls, name: str) -> type:
        """Get a connector class by name.

        Parameters
        ----------
        name : str
            Name of the connector to retrieve.

        Returns
        -------
        type
            The connector class.

        Raises
        ------
        KeyError
            If the connector is not found, with helpful suggestions.
        """
        if name not in cls._metadata:
            available = list(cls._metadata.keys())
            suggestions = get_close_matches(name, available, n=3, cutoff=0.6)

            error_msg = f"Connector '{name}' not found in registry."

            if suggestions:
                error_msg += f" Did you mean: {', '.join(suggestions)}?"

            if available:
                error_msg += f"\nAvailable connectors: {', '.join(sorted(available))}"

            if cls._by_category:
                error_msg += (
                    f"\nCategories: {', '.join(sorted(cls._by_category.keys()))}"
                )

            logger.error(error_msg)
            raise KeyError(error_msg)

        return cls._metadata[name].connector_class

    @classmethod
    def get_metadata(cls, name: str) -> ConnectorMetadata:
        """Get metadata for a connector.

        Parameters
        ----------
        name : str
            Name of the connector.

        Returns
        -------
        ConnectorMetadata
            Metadata for the connector.

        Raises
        ------
        KeyError
            If the connector is not found.
        """
        if name not in cls._metadata:
            raise KeyError(f"Connector '{name}' not found")
        return cls._metadata[name]

    @classmethod
    def list_by_category(cls, category: str) -> list[str]:
        """Get all connectors in a specific category.

        Parameters
        ----------
        category : str
            The category to filter by.

        Returns
        -------
        list[str]
            Sorted list of connector names in the category.
        """
        return sorted(cls._by_category.get(category, []))

    @classmethod
    def categories(cls) -> list[str]:
        """Get all available categories.

        Returns
        -------
        list[str]
            Sorted list of categories.
        """
        return sorted(cls._by_category.keys())

    @classmethod
    def list_all(cls) -> list[str]:
        """Get all registered connector names.

        Returns
        -------
        list[str]
            Sorted list of all connector names.
        """
        return sorted(cls._metadata.keys())

    @classmethod
    def discover(cls) -> None:
        """Print a Rich-formatted table of all available connectors."""
        if not cls._metadata:
            console.print("[yellow]No connectors registered yet.[/yellow]")
            return

        table = Table(
            title="Available Connectors",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Connector Name", style="cyan", no_wrap=True)
        table.add_column("Category", style="green")
        table.add_column("Async", style="yellow", justify="center")
        table.add_column("Description", style="white")

        # Sort by category, then by name
        sorted_connectors = sorted(
            cls._metadata.items(), key=lambda x: (x[1].category, x[0])
        )

        for name, metadata in sorted_connectors:
            description = metadata.description or "No description"
            async_support = "\u2713" if metadata.supports_async else ""
            table.add_row(name, metadata.category, async_support, description)

        console.print(table)

        # Summary
        total = len(cls._metadata)
        num_categories = len(cls._by_category)
        console.print(
            f"\n[bold]Summary:[/bold] {total} connectors "
            f"across {num_categories} categories"
        )

    @classmethod
    def clear(cls) -> None:
        """Clear all registered connectors. Primarily for testing."""
        cls._metadata.clear()
        cls._by_category.clear()
