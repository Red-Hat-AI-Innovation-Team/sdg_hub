"""Tests for the RenameColumnsBlock."""

# Third Party
# First Party
from sdg_hub.core.blocks.transform.rename_columns import RenameColumnsBlock
import pandas as pd
import pytest


def test_rename_columns_basic():
    """Test basic column renaming functionality."""
    # Create a sample dataset
    data = {
        "document": ["doc1", "doc2"],
        "summary": ["sum1", "sum2"],
        "other": ["other1", "other2"],
    }
    dataset = pd.DataFrame(data)

    # Initialize the block with column mapping (no chained renames)
    block = RenameColumnsBlock(
        block_name="test_rename",
        input_cols={"document": "raw_document", "summary": "summary_text"},
    )

    # Apply the transformation
    result = block.generate(dataset)

    # Verify the results
    assert "raw_document" in result.columns.tolist()
    assert "summary_text" in result.columns.tolist()
    assert "other" in result.columns.tolist()
    assert "document" not in result.columns.tolist()
    assert "summary" not in result.columns.tolist()

    # Check values are preserved
    assert result["raw_document"].tolist() == ["doc1", "doc2"]
    assert result["summary_text"].tolist() == ["sum1", "sum2"]
    assert result["other"].tolist() == ["other1", "other2"]


def test_rename_columns_nonexistent_column():
    """Test behavior when trying to rename a non-existent column."""
    data = {"col1": [1, 2]}
    dataset = pd.DataFrame(data)

    block = RenameColumnsBlock(
        block_name="test_nonexistent", input_cols={"nonexistent": "new_col"}
    )

    # Should raise ValueError when trying to rename non-existent column
    with pytest.raises(
        ValueError, match=r"Original column names \['nonexistent'\] not in the dataset"
    ):
        block.generate(dataset)


def test_rename_columns_overwrite():
    """Test that renaming to an existing column name raises an error."""
    data = {"col1": [1, 2], "col2": [3, 4]}
    dataset = pd.DataFrame(data)

    block = RenameColumnsBlock(block_name="test_overwrite", input_cols={"col1": "col2"})

    # Should raise ValueError to prevent creating datasets with duplicate column names
    with pytest.raises(
        ValueError, match="Cannot rename to existing column names: \\['col2'\\]"
    ):
        block.generate(dataset)


def test_rename_columns_preserve_data():
    """Test that data values are preserved after renaming."""
    data = {"old_name": ["value1", "value2", "value3"], "other_col": [1, 2, 3]}
    dataset = pd.DataFrame(data)

    block = RenameColumnsBlock(
        block_name="test_preserve", input_cols={"old_name": "new_name"}
    )

    result = block.generate(dataset)

    # Check that values are preserved
    assert result["new_name"].equals(dataset["old_name"])
    assert result["other_col"].equals(dataset["other_col"])
