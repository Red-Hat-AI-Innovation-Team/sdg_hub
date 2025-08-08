# SPDX-License-Identifier: Apache-2.0
"""Uniform column value setter block for replacing a column with a single statistic.

This block sets all values in a column to a single summary statistic:
mode, min, max, mean, or median.
"""

# Standard
from typing import Any, Literal

# Third Party
from datasets import Dataset
from pydantic import field_validator
import numpy as np

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "UniformColumnValueSetter",
    "transform",
    "Replaces all values in a column with a single summary statistic (e.g., mode, mean, median)",
)
class UniformColumnValueSetter(BaseBlock):
    """Block that replaces all values in a column with a single aggregate value.

    Supported strategies include: mode, min, max, mean, median.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Must specify exactly one input column.
    output_cols : Union[str, List[str]]
        Output column list. Ignored — modifies in place.
    reduction_strategy : Literal["mode", "min", "max", "mean", "median"]
        Strategy used to compute the replacement value.
    """

    reduction_strategy: Literal["mode", "min", "max", "mean", "median"] = "mode"

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols_single(cls, v):
        if not v or len(v) != 1:
            raise ValueError(
                "UniformColumnValueSetter requires exactly one input column"
            )
        return v

    def model_post_init(self, __context: Any) -> None:
        if hasattr(super(), "model_post_init"):
            super().model_post_init(__context)

        if self.output_cols and len(self.output_cols) > 0:
            logger.warning(
                f"UniformColumnValueSetter modifies columns in-place. "
                f"Specified output_cols {self.output_cols} will be ignored."
            )
        self.output_cols = []
        self.col_name = self.input_cols[0]

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        df = samples.to_pandas()

        if df.empty:
            raise ValueError("Cannot compute reduction for empty dataset")

        col = df[self.col_name]

        strategy = self.reduction_strategy
        if strategy == "mode":
            value = col.mode().iloc[0] if not col.mode().empty else None
        elif strategy == "min":
            value = col.min()
        elif strategy == "max":
            value = col.max()
        elif strategy == "mean":
            value = col.mean()
        elif strategy == "median":
            value = col.median()
        else:
            raise ValueError(f"Unsupported reduction strategy: {strategy}")

        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ValueError(
                f"Could not compute {strategy} for column '{self.col_name}'"
            )

        logger.info(
            f"Replacing all values in column '{self.col_name}' with {strategy} value: '{value}'"
        )

        df[self.col_name] = value
        return Dataset.from_pandas(df)
