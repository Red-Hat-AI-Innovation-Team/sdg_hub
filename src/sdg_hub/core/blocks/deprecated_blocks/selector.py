# SPDX-License-Identifier: Apache-2.0
"""DEPRECATED: SelectorBlock for backward compatibility.

This module provides a deprecated wrapper for the old SelectorBlock interface.
Use transform.IndexBasedMapperBlock instead.
"""

# Standard
from typing import Any
import warnings

# Third Party
from datasets import Dataset

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry
from ..transform.index_based_mapper import IndexBasedMapperBlock

logger = setup_logger(__name__)


@BlockRegistry.register(
    "SelectorBlock",
    "deprecated",
    "DEPRECATED: Use IndexBasedMapperBlock instead. Selects and maps values from one column to another",
)
class SelectorBlock(BaseBlock):
    """DEPRECATED: Block for selecting and mapping values from one column to another.

    .. deprecated::
        Use `sdg_hub.blocks.transform.IndexBasedMapperBlock` instead.
        This class will be removed in a future version.

    This block uses a mapping dictionary to select values from one column and
    store them in a new output column based on a choice column's value.

    Parameters
    ----------
    block_name : str
        Name of the block.
    choice_map : Dict[str, str]
        Dictionary mapping choice values to column names.
    choice_col : str
        Name of the column containing choice values.
    output_col : str
        Name of the column to store selected values.
    **batch_kwargs : Dict[str, Any]
        Additional keyword arguments for batch processing.
    """

    def __init__(
        self,
        block_name: str,
        choice_map: dict[str, str],
        choice_col: str,
        output_col: str,
        **batch_kwargs: dict[str, Any],
    ) -> None:
        warnings.warn(
            "SelectorBlock is deprecated. Use sdg_hub.blocks.transform.IndexBasedMapperBlock instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Initialize with dummy values for BaseBlock validation
        # We need all columns referenced in choice_map as input, plus the choice column
        all_input_cols = list(choice_map.values()) + [choice_col]

        super().__init__(
            block_name=block_name, input_cols=all_input_cols, output_cols=[output_col]
        )

        # Create the new implementation
        self._impl = IndexBasedMapperBlock(
            block_name=block_name,
            input_cols=all_input_cols,
            output_cols=[output_col],
            choice_map=choice_map,
            choice_cols=[choice_col],
        )

    def generate(self, samples: Dataset, **kwargs) -> Dataset:
        """Generate a new dataset with selected values.

        Parameters
        ----------
        samples : Dataset
            Input dataset to process.

        Returns
        -------
        Dataset
            Dataset with selected values stored in output column.
        """
        return self._impl.generate(samples)
