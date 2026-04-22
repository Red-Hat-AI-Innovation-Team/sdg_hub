# SPDX-License-Identifier: Apache-2.0
"""Tests for DuplicateColumnsBlock."""

import pandas as pd
import pytest

from sdg_hub.core.blocks.transform.duplicate_columns import DuplicateColumnsBlock


class TestDuplicateColumnsBlock:
    def test_single_column(self):
        block = DuplicateColumnsBlock(block_name="dup", input_cols={"source": "target"})
        df = pd.DataFrame({"source": ["a", "b", "c"]})
        result = block.generate(df)

        assert "target" in result.columns
        assert result["target"].tolist() == ["a", "b", "c"]
        assert result["source"].tolist() == ["a", "b", "c"]

    def test_multiple_columns(self):
        block = DuplicateColumnsBlock(
            block_name="dup",
            input_cols={"col_a": "copy_a", "col_b": "copy_b"},
        )
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        result = block.generate(df)

        assert result["copy_a"].tolist() == [1, 2]
        assert result["copy_b"].tolist() == [3, 4]

    def test_original_unmodified(self):
        block = DuplicateColumnsBlock(block_name="dup", input_cols={"source": "target"})
        df = pd.DataFrame({"source": ["a", "b"]})
        original = df.copy()
        block.generate(df)

        pd.testing.assert_frame_equal(df, original)

    def test_preserves_other_columns(self):
        block = DuplicateColumnsBlock(block_name="dup", input_cols={"source": "target"})
        df = pd.DataFrame({"source": [1], "other": [2]})
        result = block.generate(df)

        assert result["other"].tolist() == [2]

    def test_output_cols_auto_set(self):
        block = DuplicateColumnsBlock(block_name="dup", input_cols={"a": "b", "c": "d"})
        assert block.output_cols == ["b", "d"]

    def test_missing_source_column_raises(self):
        block = DuplicateColumnsBlock(
            block_name="dup", input_cols={"missing": "target"}
        )
        df = pd.DataFrame({"other": ["a"]})

        with pytest.raises(ValueError, match="Source column 'missing' not found"):
            block.generate(df)

    def test_empty_input_cols_raises(self):
        with pytest.raises(ValueError, match="input_cols cannot be empty"):
            DuplicateColumnsBlock(block_name="dup", input_cols={})

    def test_list_input_cols_raises(self):
        with pytest.raises(ValueError):
            DuplicateColumnsBlock(block_name="dup", input_cols=["col_a"])
