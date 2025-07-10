# SPDX-License-Identifier: Apache-2.0
"""Enhanced base block implementation with standardized patterns.

This module provides a comprehensive base class for all blocks in the system,
with unified constructor patterns, column handling, and common functionality.
"""

# Standard
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

# Third Party
from datasets import Dataset
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Local
from ..logger_config import setup_logger
from ..utils.error_handling import (
    EmptyDatasetError,
    MissingColumnError,
    OutputColumnCollisionError,
)

logger = setup_logger(__name__)
console = Console()


class BaseBlock(ABC):
    """Enhanced base class with standardized patterns for all blocks.

    This class provides a unified interface for block construction, column handling,
    and configuration loading. All blocks should inherit from this class
    to ensure consistent behavior across the system.

    Parameters
    ----------
    block_name : str
        Unique identifier for this block instance.
    input_cols : Optional[Union[str, List[str]]], optional
        Input column name(s). Can be a single string or list of strings.
        If None, block may not require input columns.
    output_cols : Optional[Union[str, List[str]]], optional
        Output column name(s). Can be a single string or list of strings.
        If None, block may not produce output columns.
    **kwargs : Any
        Additional block-specific parameters.

    Attributes
    ----------
    block_name : str
        Name of the block instance.
    input_cols : Union[List[str], Dict[str, Any]]
        Normalized input column specification.
    output_cols : Union[List[str], Dict[str, Any]]
        Normalized output column specification.
    """

    def __init__(
        self,
        block_name: str,
        input_cols: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        output_cols: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize base block with standardized parameters."""
        self._block_name = block_name
        self._input_cols = self._normalize_columns(input_cols)
        self._output_cols = self._normalize_columns(output_cols)
        # Store additional kwargs for subclass access
        self.kwargs = kwargs

        logger.debug(
            f"Initialized {self.__class__.__name__} '{block_name}'",
            extra={
                "block_name": block_name,
                "input_cols": self._input_cols,
                "output_cols": self._output_cols,
            },
        )

    @property
    def block_name(self) -> str:
        """Get the block name (immutable)."""
        return self._block_name

    @property
    def input_cols(self) -> Union[List[str], Dict[str, Any]]:
        """Get the input column specification (immutable)."""
        if isinstance(self._input_cols, dict):
            return deepcopy(self._input_cols)
        return self._input_cols.copy()

    @property
    def output_cols(self) -> Union[List[str], Dict[str, Any]]:
        """Get the output column specification (immutable)."""
        if isinstance(self._output_cols, dict):
            return deepcopy(self._output_cols)
        return self._output_cols.copy()

    def _normalize_columns(
        self, cols: Optional[Union[str, List[str], Dict[str, Any]]]
    ) -> Union[List[str], Dict[str, Any]]:
        """Normalize column specifications to appropriate format.

        Parameters
        ----------
        cols : Optional[Union[str, List[str], Dict[str, Any]]]
            Column specification as string, list of strings, dictionary keys, or None.

        Returns
        -------
        Union[List[str], Dict[str, Any]]
            Normalized column specification. Returns empty list if cols is None,
            single-item list for strings, copy of list for lists, or deep copy
            of dictionary for dictionaries.

        Raises
        ------
        ValueError
            If cols is not None, str, List[str], or Dict.
        """
        if cols is None:
            return []
        if isinstance(cols, str):
            return [cols]
        if isinstance(cols, list):
            return cols.copy()
        if isinstance(cols, dict):
            return deepcopy(cols)

        # This will only be reached if cols is not None, str, list, or dict
        raise ValueError(
            f"Invalid column specification: {cols} (type: {type(cols)}). Must be str, List[str], Dict, or None."
        )

    def _validate_columns(self, dataset: Dataset) -> None:
        """Validate that required input columns exist in dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to validate against.

        Raises
        ------
        MissingColumnError
            If required input columns are missing from the dataset.
        """
        if not self._input_cols:
            return

        # Get column names to check based on type
        if isinstance(self._input_cols, dict):
            columns_to_check = list(self._input_cols.keys())
        else:
            columns_to_check = self._input_cols

        missing_columns = [
            col for col in columns_to_check if col not in dataset.column_names
        ]

        if missing_columns:
            logger.error(
                f"[MissingColumnError] Missing required input columns in dataset: {missing_columns}",
                extra={
                    "block_name": self.block_name,
                    "missing_columns": missing_columns,
                    "available_columns": dataset.column_names,
                },
            )
            raise MissingColumnError(
                block_name=self.block_name,
                missing_columns=missing_columns,
                available_columns=dataset.column_names,
            )

    def _validate_dataset_not_empty(self, dataset: Dataset) -> None:
        """Validate that the dataset is not empty.

        Parameters
        ----------
        dataset : Dataset
            The dataset to validate.

        Raises
        ------
        EmptyDatasetError
            If the dataset is empty.
        """
        if len(dataset) == 0:
            logger.error(
                f"[EmptyDatasetError] Empty dataset provided to block '{self.block_name}'",
                extra={"block_name": self.block_name},
            )
            raise EmptyDatasetError(block_name=self.block_name)

    def _validate_output_columns(self, dataset: Dataset) -> None:
        """Validate that output columns won't overwrite existing data.

        Parameters
        ----------
        dataset : Dataset
            The dataset to validate against.

        Raises
        ------
        OutputColumnCollisionError
            If output columns would overwrite existing columns.
        """
        if not self._output_cols:
            return

        # Get column names to check based on type
        if isinstance(self._output_cols, dict):
            columns_to_check = list(self._output_cols.keys())
        else:
            columns_to_check = self._output_cols

        collision_columns = [
            col for col in columns_to_check if col in dataset.column_names
        ]

        if collision_columns:
            logger.error(
                f"[OutputColumnCollisionError] Output columns would overwrite existing data: {collision_columns}",
                extra={
                    "block_name": self.block_name,
                    "collision_columns": collision_columns,
                    "existing_columns": dataset.column_names,
                },
            )
            raise OutputColumnCollisionError(
                block_name=self.block_name,
                collision_columns=collision_columns,
                existing_columns=dataset.column_names,
            )

    def _validate_dataset(self, dataset: Dataset) -> None:
        """Perform comprehensive dataset validation.

        Parameters
        ----------
        dataset : Dataset
            The dataset to validate.

        Raises
        ------
        EmptyDatasetError
            If the dataset is empty.
        MissingColumnError
            If required input columns are missing.
        OutputColumnCollisionError
            If output columns would overwrite existing data.
        """
        self._validate_dataset_not_empty(dataset)
        self._validate_columns(dataset)
        self._validate_output_columns(dataset)

    def _validate_custom(self, dataset: Dataset) -> None:
        """Hook for subclasses to add custom validation logic.

        Override this method in subclasses to add block-specific validation
        that goes beyond the standard column and dataset validation.

        Parameters
        ----------
        dataset : Dataset
            The dataset to validate.

        Raises
        ------
        BlockValidationError
            If custom validation fails.
        """
        # Base implementation does nothing - subclasses can override
        pass

    def _log_input_data(self, dataset: Dataset) -> None:
        """Log information about input dataset using Rich panels.

        Parameters
        ----------
        dataset : Dataset
            The input dataset to log information about.
        """
        row_count = len(dataset)
        columns = dataset.column_names

        # Create input panel content
        content = Text()
        content.append("📊 Processing Input Data\n", style="bold blue")
        content.append("Block Type: ", style="dim")
        content.append(f"{self.__class__.__name__}\n", style="cyan")
        content.append("Input Rows: ", style="dim")
        content.append(f"{row_count:,}\n", style="bold cyan")
        content.append("Input Columns: ", style="dim")
        content.append(f"{len(columns)}\n", style="cyan")
        content.append("Column Names: ", style="dim")
        content.append(f"{', '.join(columns)}\n", style="white")

        if self._output_cols:
            content.append("Expected Output Columns: ", style="dim")
            if isinstance(self._output_cols, dict):
                content.append(f"{', '.join(self._output_cols.keys())}", style="green")
            else:
                content.append(f"{', '.join(self._output_cols)}", style="green")
        else:
            content.append("Expected Output Columns: ", style="dim")
            content.append("None specified", style="dim")

        # Create and log panel
        panel = Panel(
            content,
            title=f"[bold]{self.block_name}[/bold]",
            border_style="blue",
            padding=(0, 1),
        )

        console.print(panel)

    def _log_output_data(self, input_dataset: Dataset, output_dataset: Dataset) -> None:
        """Log information about output dataset and changes using Rich panels.

        Parameters
        ----------
        input_dataset : Dataset
            The original input dataset.
        output_dataset : Dataset
            The generated output dataset.
        """
        input_rows = len(input_dataset)
        output_rows = len(output_dataset)
        input_columns = set(input_dataset.column_names)
        output_columns = set(output_dataset.column_names)

        # Calculate changes
        rows_added = output_rows - input_rows
        columns_added = list(output_columns - input_columns)
        columns_removed = list(input_columns - output_columns)

        # Create output panel content
        content = Text()
        content.append("✅ Processing Complete\n", style="bold green")

        # Row changes
        content.append("Rows: ", style="dim")
        content.append(f"{input_rows:,}", style="cyan")
        content.append(" → ", style="dim")
        content.append(f"{output_rows:,}", style="cyan")

        if rows_added > 0:
            content.append(f" (+{rows_added:,})", style="bold green")
        elif rows_added < 0:
            content.append(f" ({rows_added:,})", style="bold red")
        else:
            content.append(" (no change)", style="dim")
        content.append("\n")

        # Column changes
        content.append("Columns: ", style="dim")
        content.append(f"{len(input_columns)}", style="cyan")
        content.append(" → ", style="dim")
        content.append(f"{len(output_columns)}", style="cyan")

        total_column_changes = len(columns_added) + len(columns_removed)
        if total_column_changes > 0:
            content.append(f" ({total_column_changes} changes)", style="yellow")
        else:
            content.append(" (no changes)", style="dim")
        content.append("\n")

        # Detail column changes
        if columns_added:
            content.append("🟢 Added: ", style="bold green")
            content.append(f"{', '.join(columns_added)}\n", style="green")

        if columns_removed:
            content.append("🔴 Removed: ", style="bold red")
            content.append(f"{', '.join(columns_removed)}\n", style="red")

        if not columns_added and not columns_removed:
            content.append("➡️  No column changes\n", style="dim")

        # Final columns
        content.append("📋 Final Columns: ", style="dim")
        content.append(f"{', '.join(sorted(output_columns))}", style="white")

        # Create and log panel
        panel = Panel(
            content,
            title=f"[bold green]{self.block_name} - Complete[/bold green]",
            border_style="green",
            padding=(0, 1),
        )

        console.print(panel)

    @abstractmethod
    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate output from input samples.

        This method must be implemented by all subclasses to define the specific
        processing logic for the block. Column validation is automatically performed
        before this method is called.

        Parameters
        ----------
        samples : Dataset
            Input dataset to process.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        Dataset
            Processed dataset with generated outputs.
        """

    def __call__(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Call the block with automatic dataset validation and Rich logging.

        This method performs comprehensive validation, logs input data,
        calls generate(), and logs output data with Rich panels.

        Parameters
        ----------
        samples : Dataset
            Input dataset to process.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        Dataset
            Processed dataset with generated outputs.

        Raises
        ------
        EmptyDatasetError
            If the dataset is empty.
        MissingColumnError
            If required input columns are missing from the dataset.
        OutputColumnCollisionError
            If output columns would overwrite existing data.
        """
        # Log input data with Rich panel
        self._log_input_data(samples)

        # Perform comprehensive dataset validation
        self._validate_dataset(samples)

        # Allow subclasses to add custom validation
        self._validate_custom(samples)

        # Call the actual generate method
        output_dataset = self.generate(samples, **kwargs)

        # Log output data with Rich panel
        self._log_output_data(samples, output_dataset)

        return output_dataset

    def __repr__(self) -> str:
        """String representation of the block."""
        return (
            f"{self.__class__.__name__}(name='{self.block_name}', "
            f"input_cols={self.input_cols}, output_cols={self.output_cols})"
        )

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the block.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing block information.
        """
        return {
            "block_name": self.block_name,
            "block_type": self.__class__.__name__,
            "input_cols": self.input_cols,
            "output_cols": self.output_cols,
        }
