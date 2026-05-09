# SPDX-License-Identifier: Apache-2.0
"""Tests for SimilarityFilterBlock."""

# Third Party
import numpy as np
import pandas as pd
import pytest

# Local
from sdg_hub.core.blocks.filtering.similarity_filter import SimilarityFilterBlock
from sdg_hub.core.utils.error_handling import EmptyDatasetError, MissingColumnError


@pytest.fixture
def make_block():
    """Factory fixture for creating SimilarityFilterBlock instances."""

    def _make(**kwargs):
        defaults = {
            "block_name": "test_sim_filter",
            "input_cols": ["text"],
            "threshold": 0.85,
        }
        defaults.update(kwargs)
        return SimilarityFilterBlock(**defaults)

    return _make


class TestSimilarityFilterBlock:
    """Tests for similarity-based deduplication."""

    def test_keeps_unique_rows(self, make_block):
        """Completely different rows should all be kept."""
        block = make_block()
        df = pd.DataFrame(
            {"text": ["alpha bravo charlie", "delta echo foxtrot", "golf hotel india"]}
        )
        result = block(df)
        assert len(result) == 3

    def test_removes_exact_duplicates(self, make_block):
        """Identical rows should be deduplicated to one."""
        block = make_block(threshold=0.8)
        df = pd.DataFrame({"text": ["hello world", "hello world", "hello world"]})
        result = block(df)
        assert len(result) == 1

    def test_removes_near_duplicates(self, make_block):
        """Rows differing by a single word should be caught at high threshold."""
        block = make_block(threshold=0.7)
        df = pd.DataFrame(
            {
                "text": [
                    "What is photosynthesis and how does it work?",
                    "What is photosynthesis and how does it function?",
                    "Explain the process of sourdough bread making.",
                ]
            }
        )
        result = block(df)
        assert len(result) == 2

    def test_group_by_isolates_groups(self, make_block):
        """Duplicates in different groups should both be kept."""
        block = make_block(threshold=0.8, group_by="doc_id")
        df = pd.DataFrame(
            {
                "text": ["same text here", "same text here"],
                "doc_id": ["doc_a", "doc_b"],
            }
        )
        result = block(df)
        assert len(result) == 2

    def test_group_by_deduplicates_within_group(self, make_block):
        """Duplicates within the same group should be removed."""
        block = make_block(threshold=0.8, group_by="doc_id")
        df = pd.DataFrame(
            {
                "text": ["same text here", "same text here"],
                "doc_id": ["doc_a", "doc_a"],
            }
        )
        result = block(df)
        assert len(result) == 1

    def test_empty_dataframe_raises(self, make_block):
        """Empty input should raise EmptyDatasetError via __call__."""
        block = make_block()
        df = pd.DataFrame({"text": []})
        with pytest.raises(EmptyDatasetError):
            block(df)

    def test_missing_column_raises(self, make_block):
        """Should raise MissingColumnError when input column is missing."""
        block = make_block(input_cols=["nonexistent"])
        df = pd.DataFrame({"other_col": ["a", "b"]})
        with pytest.raises(MissingColumnError):
            block(df)

    def test_low_threshold_removes_more(self, make_block):
        """A lower threshold should be more aggressive."""
        texts = [
            "What is photosynthesis?",
            "What is the process of photosynthesis?",
            "Explain sourdough bread.",
        ]
        strict = make_block(threshold=0.5)
        lenient = make_block(threshold=0.95)
        df = pd.DataFrame({"text": texts})
        assert len(strict(df)) <= len(lenient(df))

    def test_threshold_zero_filters_any_similarity(self, make_block):
        """Threshold of 0 filters any row with non-zero similarity to kept rows."""
        df = pd.DataFrame({"text": ["aaa", "aaa!", "bbb"]})
        block = make_block(threshold=0.0)
        result = block(df)
        assert len(result) == 2
        assert list(result["text"]) == ["aaa", "bbb"]

    def test_threshold_one_keeps_all(self, make_block):
        """Threshold=1.0 keeps everything since similarity > 1.0 is impossible."""
        block = make_block(threshold=1.0)
        df = pd.DataFrame({"text": ["aaa", "aab", "aaa"]})
        result = block(df)
        assert len(result) == 3

    def test_invalid_threshold_rejected(self):
        """Threshold outside 0-1 should be rejected by Pydantic."""
        with pytest.raises(ValueError):
            SimilarityFilterBlock(
                block_name="test",
                input_cols=["text"],
                threshold=1.5,
            )

    def test_empty_input_cols_rejected(self):
        """Empty input_cols should be rejected by validator."""
        with pytest.raises(ValueError, match="at least one input column"):
            SimilarityFilterBlock(
                block_name="test",
                input_cols=[],
            )

    def test_empty_string_similarity(self, make_block):
        """Two empty strings should be treated as identical and deduplicated."""
        block = make_block(threshold=0.8)
        df = pd.DataFrame({"text": ["", "", "actual content"]})
        result = block(df)
        assert len(result) == 2

    def test_nan_in_group_by_column_preserved(self, make_block):
        """Rows with NaN in group_by column should not be silently dropped."""
        block = make_block(threshold=0.8, group_by="doc_id")
        df = pd.DataFrame(
            {
                "text": ["hello world", "different text", "unique stuff"],
                "doc_id": ["doc_a", np.nan, "doc_b"],
            }
        )
        result = block(df)
        assert len(result) == 3

    def test_nan_in_comparison_column(self, make_block):
        """NaN/None values in the comparison column are coerced via str()."""
        block = make_block(threshold=0.8)
        df = pd.DataFrame({"text": [None, None, "actual content"]})
        result = block(df)
        assert len(result) == 2

    def test_first_occurrence_retained(self, make_block):
        """When duplicates are found, the first occurrence should be kept."""
        block = make_block(threshold=0.8)
        df = pd.DataFrame(
            {
                "text": ["hello world", "hello world!", "completely different"],
                "id": [1, 2, 3],
            }
        )
        result = block(df)
        assert len(result) == 2
        assert list(result["id"]) == [1, 3]

    def test_group_by_missing_column_warns(self, make_block, caplog):
        """When group_by column is missing, should warn and deduplicate globally."""
        block = make_block(threshold=0.8, group_by="missing_col")
        df = pd.DataFrame({"text": ["hello world", "hello world", "different"]})
        result = block(df)
        assert len(result) == 2
        assert "group_by column 'missing_col' not found" in caplog.text

    def test_single_row_dataframe(self, make_block):
        """A single-row DataFrame should pass through unchanged."""
        block = make_block()
        df = pd.DataFrame({"text": ["only row"]})
        result = block(df)
        assert len(result) == 1
        assert result["text"].iloc[0] == "only row"

    def test_uses_first_input_col_only(self, make_block):
        """When multiple input_cols given, only the first is used for comparison."""
        block = make_block(input_cols=["text", "other"])
        df = pd.DataFrame(
            {
                "text": ["hello", "hello"],
                "other": ["world", "universe"],
            }
        )
        result = block(df)
        assert len(result) == 1
