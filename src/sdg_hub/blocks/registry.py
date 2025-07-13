# SPDX-License-Identifier: Apache-2.0
"""Enhanced BlockRegistry with metadata and better error handling.

This module provides a clean registry system for blocks with metadata,
categorization, and improved error handling.
"""

# Standard
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Dict, List, Optional, Set, Type
import inspect

# Third Party
from rich.console import Console
from rich.table import Table

# Local
from ..logger_config import setup_logger

logger = setup_logger(__name__)
console = Console()


@dataclass
class BlockMetadata:
    """Metadata for registered blocks.

    Parameters
    ----------
    name : str
        The registered name of the block.
    block_class : Type
        The actual block class.
    category : str
        Category for organization (e.g., 'llm', 'utility', 'filtering').
    description : str, optional
        Human-readable description of what the block does.
    deprecated : bool, optional
        Whether this block is deprecated.
    replacement : str, optional
        Suggested replacement if deprecated.
    """

    name: str
    block_class: Type
    category: str
    description: str = ""
    deprecated: bool = False
    replacement: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if not self.name:
            raise ValueError("Block name cannot be empty")
        if not inspect.isclass(self.block_class):
            raise ValueError("block_class must be a class")


class BlockRegistry:
    """Registry for block classes with metadata and enhanced error handling."""

    _metadata: Dict[str, BlockMetadata] = {}
    _categories: Dict[str, Set[str]] = {}

    @classmethod
    def register(
        cls,
        block_name: str,
        category: str,
        description: str = "",
        deprecated: bool = False,
        replacement: Optional[str] = None,
    ):
        """Register a block class with metadata.

        Parameters
        ----------
        block_name : str
            Name under which to register the block.
        category : str
            Category for organization.
        description : str, optional
            Human-readable description of the block.
        deprecated : bool, optional
            Whether this block is deprecated.
        replacement : str, optional
            Suggested replacement if deprecated.

        Returns
        -------
        callable
            Decorator function.
        """

        def decorator(block_class: Type) -> Type:
            # Validate the class
            cls._validate_block_class(block_class)

            # Create metadata
            metadata = BlockMetadata(
                name=block_name,
                block_class=block_class,
                category=category,
                description=description,
                deprecated=deprecated,
                replacement=replacement,
            )

            # Register the metadata
            cls._metadata[block_name] = metadata

            # Update category index
            if category not in cls._categories:
                cls._categories[category] = set()
            cls._categories[category].add(block_name)

            logger.debug(
                f"Registered block '{block_name}' "
                f"({block_class.__name__}) in category '{category}'"
            )

            if deprecated:
                warning_msg = f"Block '{block_name}' is deprecated."
                if replacement:
                    warning_msg += f" Use '{replacement}' instead."
                logger.warning(warning_msg)

            return block_class

        return decorator

    @classmethod
    def _validate_block_class(cls, block_class: Type) -> None:
        """Validate that a class is a proper block class.

        Parameters
        ----------
        block_class : Type
            The class to validate.

        Raises
        ------
        ValueError
            If the class is not a valid block class.
        """
        if not inspect.isclass(block_class):
            raise ValueError(f"Expected a class, got {type(block_class)}")

        # Validate BaseBlock inheritance
        try:
            # Local
            from .base import BaseBlock

            if not issubclass(block_class, BaseBlock):
                raise ValueError(
                    f"Block class '{block_class.__name__}' must inherit from BaseBlock"
                )
        except ImportError as exc:
            # BaseBlock not available, check for generate method
            if not hasattr(block_class, "generate"):
                raise ValueError(
                    f"Block class '{block_class.__name__}' must implement 'generate' method"
                ) from exc

    @classmethod
    def get(cls, block_name: str) -> Type:
        """Get a block class with enhanced error handling.

        Parameters
        ----------
        block_name : str
            Name of the block to retrieve.

        Returns
        -------
        Type
            The block class.

        Raises
        ------
        KeyError
            If the block is not found, with helpful suggestions.
        """
        if block_name not in cls._metadata:
            available_blocks = list(cls._metadata.keys())
            suggestions = get_close_matches(
                block_name, available_blocks, n=3, cutoff=0.6
            )

            error_msg = f"Block '{block_name}' not found in registry."

            if suggestions:
                error_msg += f" Did you mean: {', '.join(suggestions)}?"

            if available_blocks:
                error_msg += (
                    f"\nAvailable blocks: {', '.join(sorted(available_blocks))}"
                )

            if cls._categories:
                error_msg += (
                    f"\nCategories: {', '.join(sorted(cls._categories.keys()))}"
                )

            logger.error(error_msg)
            raise KeyError(error_msg)

        metadata = cls._metadata[block_name]

        if metadata.deprecated:
            warning_msg = f"Block '{block_name}' is deprecated."
            if metadata.replacement:
                warning_msg += f" Use '{metadata.replacement}' instead."
            logger.warning(warning_msg)

        return metadata.block_class

    @classmethod
    def info(cls, block_name: str) -> BlockMetadata:
        """Get metadata for a specific block.

        Parameters
        ----------
        block_name : str
            Name of the block.

        Returns
        -------
        BlockMetadata
            The block's metadata.

        Raises
        ------
        KeyError
            If the block is not found.
        """
        if block_name not in cls._metadata:
            raise KeyError(f"Block '{block_name}' not found in registry.")
        return cls._metadata[block_name]

    @classmethod
    def categories(cls) -> List[str]:
        """Get all available categories.

        Returns
        -------
        List[str]
            Sorted list of categories.
        """
        return sorted(cls._categories.keys())

    @classmethod
    def category(cls, category: str) -> List[str]:
        """Get all blocks in a specific category.

        Parameters
        ----------
        category : str
            The category to filter by.

        Returns
        -------
        List[str]
            List of block names in the category.

        Raises
        ------
        KeyError
            If the category doesn't exist.
        """
        if category not in cls._categories:
            available_categories = sorted(cls._categories.keys())
            raise KeyError(
                f"Category '{category}' not found. "
                f"Available categories: {', '.join(available_categories)}"
            )
        return sorted(cls._categories[category])

    @classmethod
    def all(cls) -> Dict[str, List[str]]:
        """List all blocks organized by category.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping categories to lists of block names.
        """
        return {
            category: sorted(blocks) for category, blocks in cls._categories.items()
        }

    @classmethod
    def show(cls) -> None:
        """Print a Rich-formatted table of all available blocks."""
        if not cls._metadata:
            console.print("[yellow]No blocks registered yet.[/yellow]")
            return

        table = Table(
            title="Available Blocks", show_header=True, header_style="bold magenta"
        )
        table.add_column("Block Name", style="cyan", no_wrap=True)
        table.add_column("Category", style="green")
        table.add_column("Description", style="white")

        # Sort blocks by category, then by name
        sorted_blocks = sorted(
            cls._metadata.items(), key=lambda x: (x[1].category, x[0])
        )

        for name, metadata in sorted_blocks:
            description = metadata.description or "No description"

            # Show deprecated blocks with a warning indicator in the name
            block_name = f"⚠️ {name}" if metadata.deprecated else name

            table.add_row(block_name, metadata.category, description)

        console.print(table)

        # Show summary
        total_blocks = len(cls._metadata)
        total_categories = len(cls._categories)
        deprecated_count = sum(1 for m in cls._metadata.values() if m.deprecated)

        console.print(
            f"\n[bold]Summary:[/bold] {total_blocks} blocks across {total_categories} categories"
        )
        if deprecated_count > 0:
            console.print(f"[yellow]⚠️  {deprecated_count} deprecated blocks[/yellow]")
