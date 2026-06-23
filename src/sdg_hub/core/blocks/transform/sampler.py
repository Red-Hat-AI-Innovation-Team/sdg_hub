# SPDX-License-Identifier: Apache-2.0
"""Sampler block for randomly sampling values from list columns or across rows.

This module provides a block for sampling a specified number of values
from list or set columns in each row of a dataset (cell mode), or from
scalar values across all rows of a column (column mode).
"""

# Standard
from typing import Any, Literal, Optional, cast

from pydantic import Field, field_validator, model_validator

# Third Party
import numpy as np
import pandas as pd

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "SamplerBlock",
    "transform",
    "Randomly samples values from a list column (cell mode) or across rows (column mode)",
)
class SamplerBlock(BaseBlock):
    """Block for randomly sampling values from list columns or across rows.

    In ``cell`` mode (default), this block samples a specified number of
    values from each row's list/set and outputs the sampled values to a
    new column.

    In ``column`` mode, this block samples scalar values from a column
    across all rows and writes each sample into a separate output column,
    useful for constructing few-shot example sets for LLM prompts.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : list[str]
        Single input column to sample from.
    output_cols : list[str]
        Cell mode: single output column. Column mode: one output column
        per sample (length must equal ``num_samples``).
    num_samples : int
        Number of values to sample.
    random_seed : int, optional
        Random seed for reproducibility.
    return_scalar : bool
        Cell mode only. When num_samples=1, return scalar instead of list.
    source : str
        ``"cell"`` samples from a list within each row;
        ``"column"`` samples scalar values from the column across rows.
    exclude_self : bool
        Column mode only. Exclude the current row's value from the pool.
    exclude_by_value : bool
        Column mode only. When True and exclude_self is True, exclude all
        pool entries matching the current row's value, not just its index.
    replace : bool
        Sample with replacement (True) or without (False).
    sample_range : list[int], optional
        Column mode only. Restrict the sampling pool to rows
        ``[start, end)``. Default ``None`` uses all rows.
    """

    block_type: str = "transform"

    num_samples: int = Field(
        default=5, description="Number of values to randomly sample"
    )
    random_seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )
    return_scalar: bool = Field(
        default=False,
        description="When num_samples=1, return scalar value instead of single-element list",
    )
    source: Literal["cell", "column"] = Field(
        default="cell",
        description="Sampling source: 'cell' samples from a list within each row; "
        "'column' samples scalar values from the column across rows",
    )
    exclude_self: bool = Field(
        default=True,
        description="When source='column', exclude the current row's value from the pool",
    )
    exclude_by_value: bool = Field(
        default=False,
        description="When source='column' and exclude_self=True, exclude all pool entries "
        "matching the current row's value (not just the current index). "
        "Use after RowMultiplierBlock to avoid sampling duplicated copies of the same row.",
    )
    replace: bool = Field(
        default=False,
        description="Sample with replacement (True) or without (False)",
    )
    sample_range: Optional[list[int]] = Field(
        default=None,
        description="When source='column', restrict sampling pool to rows [start, end). "
        "Default None uses all rows.",
    )

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols(cls, v: list[str]) -> list[str]:
        """Validate that exactly one input column is specified."""
        if not v or len(v) != 1:
            raise ValueError("SamplerBlock requires exactly one input column")
        return v

    @field_validator("num_samples", mode="after")
    @classmethod
    def validate_num_samples(cls, v: int) -> int:
        """Validate that num_samples is at least 1."""
        if v < 1:
            raise ValueError("num_samples must be at least 1")
        return v

    @field_validator("sample_range", mode="after")
    @classmethod
    def validate_sample_range(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        """Validate sample_range is a valid [start, end) pair."""
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError("sample_range must be a 2-element list [start, end)")
        start, end = v
        if start < 0 or end < 0:
            raise ValueError("sample_range values must be non-negative")
        if start >= end:
            raise ValueError("sample_range start must be less than end")
        return v

    @model_validator(mode="after")
    def validate_output_cols_for_source(self) -> "SamplerBlock":
        """Validate output_cols length based on source mode."""
        output_cols = cast(list[str], self.output_cols)
        if self.source == "cell":
            if not output_cols or len(output_cols) != 1:
                raise ValueError(
                    "SamplerBlock requires exactly one output column in cell mode"
                )
        else:
            if not output_cols:
                raise ValueError(
                    "SamplerBlock requires at least one output column in column mode"
                )
            if len(output_cols) == 1 and self.num_samples > 1:
                base = output_cols[0]
                self.output_cols = [
                    f"{base}_{i}" for i in range(1, self.num_samples + 1)
                ]
            elif len(output_cols) != self.num_samples:
                raise ValueError(
                    f"output_cols length ({len(output_cols)}) must match "
                    f"num_samples ({self.num_samples}) in column mode"
                )
        return self

    @model_validator(mode="after")
    def validate_mode_specific_params(self) -> "SamplerBlock":
        """Reject mode-irrelevant parameters set to non-default values."""
        if self.source == "cell":
            if self.sample_range is not None:
                raise ValueError(
                    "sample_range is only valid in column mode (source='column')"
                )
            if self.exclude_by_value:
                raise ValueError(
                    "exclude_by_value is only valid in column mode (source='column')"
                )
        else:
            if self.return_scalar:
                raise ValueError(
                    "return_scalar is only valid in cell mode (source='cell')"
                )
        if self.exclude_by_value and not self.exclude_self:
            raise ValueError("exclude_by_value requires exclude_self=True")
        return self

    def _validate_custom(self, dataset: pd.DataFrame) -> None:
        """Validate dataset constraints for column mode."""
        if self.source != "column":
            return

        input_col = cast(list[str], self.input_cols)[0]
        non_null = dataset[input_col].dropna()
        if not non_null.empty:
            first_val = non_null.iloc[0]
            if isinstance(first_val, (list, dict, set, np.ndarray)):
                raise ValueError(
                    f"Column '{input_col}' contains {type(first_val).__name__} values. "
                    f"Column mode requires scalar values. "
                    f"Use source='cell' for list/dict/set columns."
                )

        if self.sample_range is not None:
            start, end = self.sample_range
            if end > len(dataset):
                raise ValueError(
                    f"sample_range end ({end}) exceeds dataset length ({len(dataset)})"
                )
            pool_size = end - start
        else:
            pool_size = len(dataset)

        if not self.replace:
            min_required = self.num_samples + (1 if self.exclude_self else 0)
            if pool_size < min_required:
                raise ValueError(
                    f"Sampling pool has {pool_size} rows but needs at least "
                    f"{min_required} to sample {self.num_samples} values "
                    f"without replacement"
                    f"{' with exclude_self=True' if self.exclude_self else ''}"
                )

    def _sample_values(self, values: Any, rng: np.random.Generator) -> list[Any]:
        """Sample values from a list or set.

        Parameters
        ----------
        values : Any
            The list, set, or other iterable to sample from.
        rng : np.random.Generator
            Random number generator for sampling.

        Returns
        -------
        list[Any]
            Sampled values as a list.
        """
        if values is None:
            return []

        # Handle dictionary input (weighted sampling)
        if isinstance(values, dict):
            if len(values) == 0:
                return []
            items = list(values.keys())
            weights = np.array(list(values.values()), dtype=float)
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("Weights must be finite and non-negative")
            mask = weights > 0
            items = [items[i] for i in range(len(items)) if mask[i]]
            weights = weights[mask]
            if len(items) == 0:
                return []
            p = weights / weights.sum()
            n = self.num_samples if self.replace else min(self.num_samples, len(items))
            indices = rng.choice(len(items), size=n, replace=self.replace, p=p)
            return [items[i] for i in indices]

        # Convert to list if it's a set or other iterable
        if isinstance(values, set):
            try:
                values = sorted(values)
            except TypeError:
                values = list(values)
        elif not isinstance(values, (list, np.ndarray)):
            try:
                values = list(values)
            except TypeError:
                return []

        if len(values) == 0:
            return []

        n = self.num_samples if self.replace else min(self.num_samples, len(values))
        indices = rng.choice(len(values), size=n, replace=self.replace)
        return [values[i] for i in indices]

    def _generate_column_mode(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Sample values from a column across all rows.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            Dataset with sampled columns added.
        """
        input_col = cast(list[str], self.input_cols)[0]
        output_cols = cast(list[str], self.output_cols)

        all_values = samples[input_col].to_numpy()
        if self.sample_range is not None:
            start, end = self.sample_range
            pool_values = all_values[start:end]
            pool_indices = np.arange(start, end)
        else:
            pool_values = all_values
            pool_indices = np.arange(len(all_values))

        n_rows = len(samples)
        rng = np.random.default_rng(self.random_seed)
        result = samples.copy()

        sampled = np.empty((n_rows, self.num_samples), dtype=object)

        for idx in range(n_rows):
            if self.exclude_self:
                if self.exclude_by_value:
                    val = all_values[idx]
                    if pd.isna(val):
                        mask = pd.notna(pool_values)
                    else:
                        mask = pool_values != val
                else:
                    mask = pool_indices != idx
                row_pool = pool_values[mask]
            else:
                row_pool = pool_values

            if len(row_pool) == 0:
                raise ValueError(
                    f"Row {idx} has no eligible pool entries after exclusion"
                )

            if len(row_pool) < self.num_samples and not self.replace:
                raise ValueError(
                    f"Row {idx} has only {len(row_pool)} eligible pool entries "
                    f"after exclusion, but num_samples={self.num_samples} "
                    f"(without replacement)"
                )

            indices = rng.choice(
                len(row_pool), size=self.num_samples, replace=self.replace
            )
            sampled[idx] = row_pool[indices]

        for j, col in enumerate(output_cols):
            result[col] = sampled[:, j]

        return result

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate a dataset with sampled values.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset to process.

        Returns
        -------
        pd.DataFrame
            Dataset with sampled values in output column(s).
        """
        if self.source == "column":
            return self._generate_column_mode(samples)

        input_cols = cast(list[str], self.input_cols)
        output_cols = cast(list[str], self.output_cols)
        input_col = input_cols[0]
        output_col = output_cols[0]

        result = samples.copy()

        rng = np.random.default_rng(self.random_seed)

        result[output_col] = result[input_col].apply(
            lambda x: self._sample_values(x, rng)
        )

        # Unwrap scalar values when return_scalar=True and num_samples=1
        if self.return_scalar and self.num_samples == 1:
            result[output_col] = result[output_col].apply(lambda x: x[0] if x else None)

        return result
