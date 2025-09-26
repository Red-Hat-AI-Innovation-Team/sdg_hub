# SPDX-License-Identifier: Apache-2.0
"""Rename columns block for dataset column renaming operations.

This module provides a block for renaming columns in datasets according
to a mapping specification.
"""

# Standard
from typing import Any

# Third Party
from datasets import Dataset
from pydantic import field_validator

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "RenameColumnsBlock",
    "transform",
    "Renames columns in a dataset according to a mapping specification",
)
class RenameColumnsBlock(BaseBlock):
    """Block for renaming columns in a dataset.

    This block renames columns in a dataset according to a mapping specification.
    The mapping is provided through input_cols as a dictionary.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : Dict[str, str]
        Dictionary mapping existing column names to new column names.
        Keys are existing column names, values are new column names.
    """

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols(cls, v):
        """Validate that input_cols is a non-empty dict."""
        if not v:
            raise ValueError("input_cols cannot be empty")
        if not isinstance(v, dict):
            raise ValueError(
                "input_cols must be a dictionary mapping old column names to new column names"
            )
        return v

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate a dataset with renamed columns.

        Parameters
        ----------
        samples : Dataset
            Input dataset to rename columns in.

        Returns
        -------
        Dataset
            Dataset with renamed columns.
        """
        # Rename columns using HuggingFace datasets method
        return samples.rename_columns(self.input_cols)
