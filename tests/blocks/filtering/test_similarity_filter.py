# SPDX-License-Identifier: Apache-2.0
"""Tests for SimilarityFilterBlock."""

# Third Party
import pandas as pd
import pytest

# Local
from sdg_hub.core.blocks.filtering.similarity_filter import SimilarityFilterBlock


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
        result = block.generate(df)
        assert len(result) == 3

    def test_removes_exact_duplicates(self, make_block):
        """Identical rows should be deduplicated to one."""
        block = make_block(threshold=0.8)
        df = pd.DataFrame({"text": ["hello world", "hello world", "hello world"]})
        result = block.generate(df)
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
        result = block.generate(df)
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
        result = block.generate(df)
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
        result = block.generate(df)
        assert len(result) == 1

    def test_empty_dataframe(self, make_block):
        """Empty input should return empty output."""
        block = make_block()
        df = pd.DataFrame({"text": []})
        result = block.generate(df)
        assert len(result) == 0

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
        assert len(strict.generate(df)) <= len(lenient.generate(df))

    def test_threshold_boundaries(self, make_block):
        """Threshold of 0 filters any row with non-zero similarity to kept rows."""
        df = pd.DataFrame({"text": ["aaa", "aaa!", "bbb"]})
        block_zero = make_block(threshold=0.0)
        result = block_zero.generate(df)
        # "aaa" kept, "aaa!" filtered (similarity > 0), "bbb" kept (similarity = 0)
        assert len(result) == 2

    def test_missing_column_raises_error(self, make_block):
        """Should raise KeyError when input column is missing."""
        block = make_block(input_cols=["nonexistent"])
        df = pd.DataFrame({"other_col": ["a", "b"]})
        with pytest.raises(KeyError):
            block.generate(df)

    def test_invalid_threshold_rejected(self):
        """Threshold outside 0-1 should be rejected by Pydantic."""
        with pytest.raises(ValueError):
            SimilarityFilterBlock(
                block_name="test",
                input_cols=["text"],
                threshold=1.5,
            )

    def test_group_by_missing_column_warns(self, make_block, caplog):
        """When group_by column is missing, should warn and deduplicate globally."""
        block = make_block(threshold=0.8, group_by="missing_col")
        df = pd.DataFrame({"text": ["hello world", "hello world", "different"]})
        result = block.generate(df)
        assert len(result) == 2
        assert "group_by column 'missing_col' not found" in caplog.text
