# SPDX-License-Identifier: Apache-2.0
"""Flow registry for managing contributed flows."""

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import os

# Third Party
import yaml

# Local
from ..logger_config import setup_logger
from .metadata import FlowMetadata

logger = setup_logger(__name__)


@dataclass
class FlowRegistryEntry:
    """Entry in the flow registry.

    Parameters
    ----------
    path : str
        Path to the flow YAML file.
    metadata : FlowMetadata
        Flow metadata extracted from the file.
    """

    path: str
    metadata: FlowMetadata


class FlowRegistry:
    """Registry for managing contributed flows."""

    _entries: Dict[str, FlowRegistryEntry] = {}
    _search_paths: List[str] = []

    @classmethod
    def register_search_path(cls, path: str) -> None:
        """Add a directory to search for flows.

        Parameters
        ----------
        path : str
            Path to directory containing flow YAML files.
        """
        if path not in cls._search_paths:
            cls._search_paths.append(path)
            logger.debug(f"Added flow search path: {path}")

    @classmethod
    def _discover_flows(cls, force_refresh: bool = False) -> None:
        """Discover and register flows from search paths (private method).

        Parameters
        ----------
        force_refresh : bool, optional
            Whether to force refresh the registry.
        """
        if cls._entries and not force_refresh:
            return

        cls._entries.clear()

        for search_path in cls._search_paths:
            if not os.path.exists(search_path):
                logger.warning(f"Flow search path does not exist: {search_path}")
                continue

            cls._discover_flows_in_directory(search_path)

        logger.info(f"Discovered {len(cls._entries)} flows")

    @classmethod
    def _discover_flows_in_directory(cls, directory: str) -> None:
        """Discover flows in a specific directory."""
        path = Path(directory)

        for yaml_file in path.rglob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    flow_config = yaml.safe_load(f)

                # Check if this is a flow file
                if "metadata" in flow_config and "blocks" in flow_config:
                    metadata_dict = flow_config["metadata"]
                    metadata = FlowMetadata(**metadata_dict)

                    entry = FlowRegistryEntry(path=str(yaml_file), metadata=metadata)

                    cls._entries[metadata.name] = entry
                    logger.debug(f"Registered flow: {metadata.name} from {yaml_file}")

            except Exception as exc:
                logger.debug(f"Skipped {yaml_file}: {exc}")

    @classmethod
    def get_flow_path(cls, flow_name: str) -> Optional[str]:
        """Get the path to a registered flow.

        Parameters
        ----------
        flow_name : str
            Name of the flow to find.

        Returns
        -------
        Optional[str]
            Path to the flow file, or None if not found.
        """
        cls._discover_flows()

        if flow_name in cls._entries:
            return cls._entries[flow_name].path
        return None

    @classmethod
    def get_flow_metadata(cls, flow_name: str) -> Optional[FlowMetadata]:
        """Get metadata for a registered flow.

        Parameters
        ----------
        flow_name : str
            Name of the flow.

        Returns
        -------
        Optional[FlowMetadata]
            Flow metadata, or None if not found.
        """
        cls._discover_flows()

        if flow_name in cls._entries:
            return cls._entries[flow_name].metadata
        return None

    @classmethod
    def list_flows(cls) -> List[str]:
        """List all registered flow names.

        Returns
        -------
        List[str]
            List of flow names.
        """
        cls._discover_flows()
        return list(cls._entries.keys())

    @classmethod
    def search_flows(
        cls, tag: Optional[str] = None, author: Optional[str] = None
    ) -> List[str]:
        """Search flows by criteria.

        Parameters
        ----------
        tag : Optional[str]
            Tag to filter by.
        author : Optional[str]
            Author to filter by.

        Returns
        -------
        List[str]
            List of matching flow names.
        """
        cls._discover_flows()

        matching_flows = []

        for name, entry in cls._entries.items():
            metadata = entry.metadata

            # Filter by tag
            if tag and tag not in metadata.tags:
                continue

            # Filter by author
            if author and author.lower() not in metadata.author.lower():
                continue

            matching_flows.append(name)

        return matching_flows

    @classmethod
    def get_flows_by_category(cls) -> Dict[str, List[str]]:
        """Get flows organized by their primary tag.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping tags to flow names.
        """
        cls._discover_flows()

        categories = {}

        for name, entry in cls._entries.items():
            metadata = entry.metadata

            # Use first tag as primary category, or "uncategorized"
            category = metadata.tags[0] if metadata.tags else "uncategorized"

            if category not in categories:
                categories[category] = []

            categories[category].append(name)

        return categories

    @classmethod
    def discover_flows(cls, show_all_columns: bool = False) -> None:
        """Discover and display all flows in a formatted table.
        
        This is the main public API for flow discovery. It finds all flows
        in registered search paths and displays them in a beautiful Rich table.
        
        Parameters
        ----------
        show_all_columns : bool, optional
            Whether to show extended table with all columns, by default False
        """
        cls._discover_flows()
        
        if not cls._entries:
            print("No flows discovered. Try adding search paths with register_search_path()")
            print("Note: Only flows with 'metadata' section are discoverable.")
            return
        
        # Prepare data with fallbacks
        flow_data = []
        for name, entry in cls._entries.items():
            metadata = entry.metadata
            flow_data.append({
                "name": name,
                "author": metadata.author or "Unknown",
                "tags": ", ".join(metadata.tags) if metadata.tags else "-", 
                "description": metadata.description or "No description",
                "version": metadata.version,
                "cost": metadata.estimated_cost,
            })
        
        # Sort by name for consistency
        flow_data.sort(key=lambda x: x["name"])
        
        # Display Rich table
        # Third Party
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        
        # Add columns
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Author", style="green")
        
        if show_all_columns:
            table.add_column("Version", style="blue")
            table.add_column("Cost", style="yellow")
        
        table.add_column("Tags", style="dim")
        table.add_column("Description")
        
        # Add rows
        for flow in flow_data:
            if show_all_columns:
                table.add_row(
                    flow["name"],
                    flow["author"], 
                    flow["version"],
                    flow["cost"],
                    flow["tags"],
                    flow["description"]
                )
            else:
                table.add_row(
                    flow["name"],
                    flow["author"], 
                    flow["tags"],
                    flow["description"]
                )
        
        console.print(table)
