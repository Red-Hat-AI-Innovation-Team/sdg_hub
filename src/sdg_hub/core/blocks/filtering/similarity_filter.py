# SPDX-License-Identifier: Apache-2.0
"""Similarity filter block for removing near-duplicate rows.

This module provides a block that performs greedy sequential deduplication
based on text similarity, comparing each row against previously retained
rows and dropping entries whose similarity exceeds a configurable threshold.
"""

# Standard
from difflib import SequenceMatcher
from typing import Any, Optional, cast

from pydantic import Field, field_validator

# Third Party
import pandas as pd

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "SimilarityFilterBlock",
    "filtering",
    "Filters near-duplicate rows based on text similarity within optional groups",
)
class SimilarityFilterBlock(BaseBlock):
    """A block that removes near-duplicate rows based on text similarity.

    Performs greedy sequential deduplication within optional groups
    (e.g., per document). Each row is compared against previously
    retained rows and dropped if similarity exceeds the threshold.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Column(s) to compare for similarity. The first column is used
        for comparison.
    threshold : float
        Similarity threshold (0.0 to 1.0). Rows with similarity above
        this value to any kept row are dropped. Default 0.85.
    group_by : Optional[str]
        Column to group by before deduplication. If set, similarity is
        only compared within each group.
    """

    block_type: str = "filtering"

    threshold: float = Field(
        0.85,
        description="Similarity threshold (0-1). Rows above this are dropped.",
        ge=0.0,
        le=1.0,
    )
    group_by: Optional[str] = Field(
        None,
        description="Column to group by before comparing similarity.",
    )

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols_not_empty(cls, v: list[str]) -> list[str]:
        """Validate that we have at least one input column."""
        if not v or len(v) == 0:
            raise ValueError("SimilarityFilterBlock requires at least one input column")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Initialize derived attributes after Pydantic validation."""
        super().model_post_init(__context)
        if self.output_cols is None:
            self.output_cols = []

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Compute similarity ratio between two strings."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _deduplicate_group(self, group: pd.DataFrame, col: str) -> pd.DataFrame:
        """Remove near-duplicate rows within a single group."""
        kept_indices: list[Any] = []
        kept_texts: list[str] = []

        for idx, row in group.iterrows():
            text = str(row[col])
            is_duplicate = any(
                self._similarity(text, kept) > self.threshold for kept in kept_texts
            )
            if not is_duplicate:
                kept_indices.append(idx)
                kept_texts.append(text)

        return group.loc[kept_indices]

    def generate(self, samples: pd.DataFrame, **_kwargs: Any) -> pd.DataFrame:
        """Filter near-duplicate rows based on text similarity.

        Parameters
        ----------
        samples : pd.DataFrame
            The input dataset.

        Returns
        -------
        pd.DataFrame
            Dataset with near-duplicates removed.
        """
        input_cols = cast(list[str], self.input_cols)
        compare_col = input_cols[0]

        original_len = len(samples)

        if self.group_by and self.group_by not in samples.columns:
            logger.warning(
                "SimilarityFilterBlock '%s': group_by column '%s' not found, "
                "applying global deduplication",
                self.block_name,
                self.group_by,
            )

        if self.group_by and self.group_by in samples.columns:
            groups = []
            for _, group in samples.groupby(self.group_by, dropna=False):
                groups.append(self._deduplicate_group(group, compare_col))
            result = (
                pd.concat(groups, ignore_index=True) if groups else samples.iloc[:0]
            )
        else:
            result = self._deduplicate_group(samples, compare_col)

        removed = original_len - len(result)
        if removed > 0:
            logger.info(
                "SimilarityFilterBlock '%s': removed %d near-duplicates "
                "(threshold=%.2f), %d -> %d rows",
                self.block_name,
                removed,
                self.threshold,
                original_len,
                len(result),
            )
        else:
            logger.info(
                "SimilarityFilterBlock '%s': no near-duplicates found "
                "(threshold=%.2f, %d rows examined)",
                self.block_name,
                self.threshold,
                original_len,
            )

        return result.reset_index(drop=True)
