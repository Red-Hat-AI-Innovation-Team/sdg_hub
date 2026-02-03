# SPDX-License-Identifier: Apache-2.0
"""Tests for the DuplicateColumnsBlock."""

# Third Party
from pydantic import ValidationError

# First Party
from sdg_hub.core.blocks.transform.duplicate_columns import DuplicateColumnsBlock
import pandas as pd
import pytest


class TestDuplicateColumnsBlock:
    """Tests for DuplicateColumnsBlock class."""

    def test_duplicate_single_column(self):
        """Test duplicating a single column."""
        data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        dataset = pd.DataFrame(data)

        block = DuplicateColumnsBlock(
            block_name="test_duplicate", input_cols={"col1": "col1_copy"}
        )

        result = block.generate(dataset)

        assert "col1" in result.columns.tolist()
        assert "col1_copy" in result.columns.tolist()
        assert "col2" in result.columns.tolist()
        assert result["col1"].tolist() == result["col1_copy"].tolist()

    def test_duplicate_multiple_columns(self):
        """Test duplicating multiple columns."""
        data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [4.0, 5.0, 6.0]}
        dataset = pd.DataFrame(data)

        block = DuplicateColumnsBlock(
            block_name="test_multi_duplicate",
            input_cols={"col1": "col1_dup", "col2": "col2_dup"},
        )

        result = block.generate(dataset)

        assert "col1_dup" in result.columns.tolist()
        assert "col2_dup" in result.columns.tolist()
        assert result["col1"].tolist() == result["col1_dup"].tolist()
        assert result["col2"].tolist() == result["col2_dup"].tolist()
        assert result["col3"].tolist() == [4.0, 5.0, 6.0]  # Original preserved

    def test_duplicate_preserves_original(self):
        """Test that original columns are preserved."""
        data = {"original": [10, 20, 30]}
        dataset = pd.DataFrame(data)

        block = DuplicateColumnsBlock(
            block_name="test_preserve", input_cols={"original": "copy"}
        )

        result = block.generate(dataset)

        assert "original" in result.columns.tolist()
        assert "copy" in result.columns.tolist()
        assert result["original"].tolist() == [10, 20, 30]
        assert result["copy"].tolist() == [10, 20, 30]

    def test_duplicate_nonexistent_column_raises_error(self):
        """Test that duplicating a non-existent column raises an error."""
        data = {"col1": [1, 2, 3]}
        dataset = pd.DataFrame(data)

        block = DuplicateColumnsBlock(
            block_name="test_nonexistent", input_cols={"nonexistent": "copy"}
        )

        with pytest.raises(ValueError, match="Source column 'nonexistent' not found"):
            block.generate(dataset)

    def test_empty_input_cols_raises_error(self):
        """Test that empty input_cols raises validation error."""
        with pytest.raises(ValidationError, match="input_cols cannot be empty"):
            DuplicateColumnsBlock(block_name="test_empty", input_cols={})

    def test_output_cols_auto_set(self):
        """Test that output_cols is automatically set from input_cols values."""
        block = DuplicateColumnsBlock(
            block_name="test_output_cols",
            input_cols={"col1": "col1_copy", "col2": "col2_copy"},
        )

        assert block.output_cols == ["col1_copy", "col2_copy"]

    def test_output_cols_can_be_overridden(self):
        """Test that output_cols can be manually overridden."""
        block = DuplicateColumnsBlock(
            block_name="test_override",
            input_cols={"col1": "col1_copy"},
            output_cols=["custom_output"],
        )

        # When explicitly provided, output_cols should be preserved
        assert block.output_cols == ["custom_output"]

    def test_duplicate_with_various_dtypes(self):
        """Test duplicating columns with various data types."""
        data = {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
        dataset = pd.DataFrame(data)

        block = DuplicateColumnsBlock(
            block_name="test_dtypes",
            input_cols={
                "int_col": "int_copy",
                "float_col": "float_copy",
                "str_col": "str_copy",
                "bool_col": "bool_copy",
            },
        )

        result = block.generate(dataset)

        assert result["int_copy"].tolist() == [1, 2, 3]
        assert result["float_copy"].tolist() == [1.1, 2.2, 3.3]
        assert result["str_copy"].tolist() == ["a", "b", "c"]
        assert result["bool_copy"].tolist() == [True, False, True]

    def test_duplicate_does_not_modify_original_dataframe(self):
        """Test that the original DataFrame is not modified."""
        data = {"col1": [1, 2, 3]}
        dataset = pd.DataFrame(data)
        original_columns = dataset.columns.tolist()

        block = DuplicateColumnsBlock(
            block_name="test_no_modify", input_cols={"col1": "col1_copy"}
        )

        block.generate(dataset)

        # Original should be unchanged
        assert dataset.columns.tolist() == original_columns
        assert "col1_copy" not in dataset.columns.tolist()

    def test_block_type_is_transform(self):
        """Test that block_type is correctly set to 'transform'."""
        block = DuplicateColumnsBlock(
            block_name="test_type", input_cols={"col1": "col2"}
        )

        assert block.block_type == "transform"

    def test_duplicate_with_none_values(self):
        """Test duplicating columns containing None values."""
        data = {"col1": [1, None, 3]}
        dataset = pd.DataFrame(data)

        block = DuplicateColumnsBlock(
            block_name="test_none", input_cols={"col1": "col1_copy"}
        )

        result = block.generate(dataset)

        assert result["col1_copy"].tolist()[0] == 1
        assert pd.isna(result["col1_copy"].tolist()[1])
        assert result["col1_copy"].tolist()[2] == 3
