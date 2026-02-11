# SPDX-License-Identifier: Apache-2.0
"""Display and info helper functions for Flow class."""

# Standard
from typing import TYPE_CHECKING, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# Third Party
import pandas as pd

# Local
from .metadata import DatasetRequirements

if TYPE_CHECKING:
    from .base import Flow


def get_flow_info(flow: "Flow") -> dict[str, Any]:
    """Get information about the flow.

    Parameters
    ----------
    flow : Flow
        The flow instance to get info from.

    Returns
    -------
    dict[str, Any]
        Dictionary containing flow metadata and block information.
    """
    return {
        "metadata": flow.metadata.model_dump(),
        "blocks": [
            {
                "block_class": block.__class__.__name__,
                "block_name": block.block_name,
                "input_cols": getattr(block, "input_cols", None),
                "output_cols": getattr(block, "output_cols", None),
            }
            for block in flow.blocks
        ],
        "total_blocks": len(flow.blocks),
        "block_names": [block.block_name for block in flow.blocks],
    }


def get_dataset_requirements(flow: "Flow") -> Optional[DatasetRequirements]:
    """Get the dataset requirements for this flow.

    Parameters
    ----------
    flow : Flow
        The flow instance.

    Returns
    -------
    Optional[DatasetRequirements]
        Dataset requirements object or None if not defined.

    Examples
    --------
    >>> flow = Flow.from_yaml("path/to/flow.yaml")
    >>> requirements = flow.get_dataset_requirements()
    >>> if requirements:
    ...     print(f"Required columns: {requirements.required_columns}")
    """
    return flow.metadata.dataset_requirements


def get_dataset_schema(flow: "Flow") -> pd.DataFrame:
    """Get an empty dataset with the correct schema for this flow.

    Parameters
    ----------
    flow : Flow
        The flow instance.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with the correct schema/features for this flow.
        Users can add data to this dataset or use it to validate their own dataset schema.

    Examples
    --------
    >>> flow = Flow.from_yaml("path/to/flow.yaml")
    >>> schema_dataset = flow.get_dataset_schema()
    >>>
    >>> # Add your data using pandas concat
    >>> new_row = pd.DataFrame([{
    ...     "document": "Your document text",
    ...     "domain": "Computer Science",
    ...     "icl_document": "Example document"
    ... }])
    >>> schema_dataset = pd.concat([schema_dataset, new_row], ignore_index=True)
    >>>
    >>> # Or validate your existing dataset schema
    >>> my_dataset = pd.DataFrame(my_data)
    >>> if set(my_dataset.columns) == set(schema_dataset.columns):
    ...     print("Schema matches!")
    """
    requirements = get_dataset_requirements(flow)

    if requirements is None:
        # Return empty dataframe with no schema requirements
        return pd.DataFrame({})

    # Build schema with column names and dtypes
    schema = {}

    # Process required columns
    for col_name in requirements.required_columns:
        col_type = requirements.column_types.get(col_name, "string")
        schema[col_name] = map_column_type_to_dtype(col_type)

    # Process optional columns
    for col_name in requirements.optional_columns:
        col_type = requirements.column_types.get(col_name, "string")
        schema[col_name] = map_column_type_to_dtype(col_type)

    # Create empty dataframe with the correct dtypes
    empty_data = {
        col_name: pd.Series([], dtype=dtype) for col_name, dtype in schema.items()
    }

    return pd.DataFrame(empty_data)


def map_column_type_to_dtype(col_type: str) -> str:
    """Map column type string to pandas dtype.

    Parameters
    ----------
    col_type : str
        Column type string (e.g., "string", "int", "float").

    Returns
    -------
    str
        Pandas dtype string.
    """
    # Map common type names to pandas dtypes
    if col_type in ["str", "string", "text"]:
        return "object"  # pandas uses 'object' for strings
    elif col_type in ["int", "integer"]:
        return "Int64"  # nullable integer
    elif col_type in ["float", "number"]:
        return "float64"
    elif col_type in ["bool", "boolean"]:
        return "boolean"  # nullable boolean
    else:
        # Default to object (string) for unknown types
        return "object"


def print_flow_info(flow: "Flow") -> None:
    """Print an interactive summary of the Flow in the console.

    The summary contains:
    1. Flow metadata (name, version, author, description)
    2. A table of all blocks with their input and output columns

    Parameters
    ----------
    flow : Flow
        The flow instance to print info for.

    Notes
    -----
    Uses the `rich` library for colourised output; install with
    `pip install rich` if not already present.

    Returns
    -------
    None
    """
    console = Console()

    # Create main tree structure
    flow_tree = Tree(f"[bold bright_blue]{flow.metadata.name}[/bold bright_blue] Flow")

    # Metadata section
    metadata_branch = flow_tree.add("[bold bright_green]Metadata[/bold bright_green]")
    metadata_branch.add(f"Version: [bright_cyan]{flow.metadata.version}[/bright_cyan]")
    metadata_branch.add(f"Author: [bright_cyan]{flow.metadata.author}[/bright_cyan]")
    if flow.metadata.description:
        metadata_branch.add(f"Description: [white]{flow.metadata.description}[/white]")

    # Blocks overview
    flow_tree.add(
        f"[bold bright_magenta]Blocks[/bold bright_magenta] ({len(flow.blocks)} total)"
    )

    # Create blocks table
    blocks_table = Table(show_header=True, header_style="bold bright_white")
    blocks_table.add_column("Block Name", style="bright_cyan")
    blocks_table.add_column("Type", style="bright_green")
    blocks_table.add_column("Input Cols", style="bright_yellow")
    blocks_table.add_column("Output Cols", style="bright_red")

    for block in flow.blocks:
        input_cols = getattr(block, "input_cols", None)
        output_cols = getattr(block, "output_cols", None)

        blocks_table.add_row(
            block.block_name,
            block.__class__.__name__,
            str(input_cols) if input_cols else "[bright_black]None[/bright_black]",
            str(output_cols) if output_cols else "[bright_black]None[/bright_black]",
        )

    # Print everything
    console.print()
    console.print(
        Panel(
            flow_tree,
            title="[bold bright_white]Flow Information[/bold bright_white]",
            border_style="bright_blue",
        )
    )
    console.print()
    console.print(
        Panel(
            blocks_table,
            title="[bold bright_white]Block Details[/bold bright_white]",
            border_style="bright_magenta",
        )
    )
    console.print()
