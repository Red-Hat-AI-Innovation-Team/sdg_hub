# SPDX-License-Identifier: Apache-2.0
"""Column dependency tracking for memory optimization."""

# Standard
from typing import Any, Union

# Local
from ..blocks.base import BaseBlock


class ColumnDependencyTracker:
    """Tracks when columns are last used to enable early dropping.

    This class analyzes a flow's blocks to determine when columns are no longer
    needed and can be safely dropped to reduce memory usage.
    """

    def __init__(
        self,
        blocks: list[BaseBlock],
        columns_to_keep: set[str],
        original_columns: set[str],
    ):
        """Initialize dependency tracker.

        Parameters
        ----------
        blocks : list[BaseBlock]
            Ordered list of blocks in the flow
        columns_to_keep : set[str]
            Columns that must be preserved in final output
        original_columns : set[str]
            Original input columns (auto-preserved)
        """
        self._preserved: frozenset[str] = frozenset(columns_to_keep | original_columns)
        self._last_consumer: dict[
            str, int
        ] = {}  # col -> last block index that reads it
        self._build_last_consumer_map(blocks)

    def _build_last_consumer_map(self, blocks: list[BaseBlock]) -> None:
        """Build map of each column to the last block index that reads it."""
        for i, block in enumerate(blocks):
            for col in self._extract_input_columns(block.input_cols):
                self._last_consumer[col] = i

    @staticmethod
    def _extract_input_columns(
        input_cols: Union[str, list[str], dict[str, Any], None],
    ) -> list[str]:
        """Extract actual column names that a block reads.

        Parameters
        ----------
        input_cols : Union[str, list[str], dict[str, Any], None]
            Block's input_cols specification

        Returns
        -------
        list[str]
            List of column names the block reads from
        """
        if input_cols is None:
            return []
        if isinstance(input_cols, str):
            return [input_cols]
        if isinstance(input_cols, list):
            return input_cols
        if isinstance(input_cols, dict):
            # For dict inputs, keys are the columns being read
            return list(input_cols.keys())
        raise ValueError(
            f"Unexpected input_cols type: {type(input_cols)}. "
            f"Expected str, list, dict, or None."
        )

    def get_droppable_columns(
        self, block_index: int, current_columns: set[str]
    ) -> list[str]:
        """Determine which columns can be dropped after a block completes.

        Parameters
        ----------
        block_index : int
            Index of the block that just completed
        current_columns : set[str]
            Currently available columns in the dataset

        Returns
        -------
        list[str]
            Columns that can be safely dropped
        """
        droppable = []
        for col in current_columns:
            if col in self._preserved:
                continue
            # Only drop columns with a known last consumer. Columns that no
            # block declares as input_cols (e.g. custom blocks reading columns
            # programmatically) are deferred to _cleanup_final_columns.
            last = self._last_consumer.get(col)
            if last is not None and last <= block_index:
                droppable.append(col)
        return droppable
