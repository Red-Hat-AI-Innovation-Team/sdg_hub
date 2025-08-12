"""Tests for the DuplicateColumns block functionality."""

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.deprecated_blocks import DuplicateColumns
import pytest


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {"document": ["doc1", "doc2", "doc3"], "other_col": ["val1", "val2", "val3"]}
    )


def test_duplicate_single_column(sample_dataset):
    """Test duplicating a single column."""
    block = DuplicateColumns(
        block_name="test_duplicate", columns_map={"document": "base_document"}
    )

    result = block.generate(sample_dataset)

    # Check that original columns are preserved
    assert "document" in result.column_names
    assert "other_col" in result.column_names
    # Check that new column is added
    assert "base_document" in result.column_names
    # Check that values are correctly duplicated
    assert result["document"] == result["base_document"]


def test_duplicate_multiple_columns(sample_dataset):
    """Test duplicating multiple columns."""
    block = DuplicateColumns(
        block_name="test_duplicate_multiple",
        columns_map={"document": "base_document", "other_col": "duplicate_other_col"},
    )

    result = block.generate(sample_dataset)

    # Check all columns exist
    assert "document" in result.column_names
    assert "other_col" in result.column_names
    assert "base_document" in result.column_names
    assert "duplicate_other_col" in result.column_names
    # Check values are correctly duplicated
    assert result["document"] == result["base_document"]
    assert result["other_col"] == result["duplicate_other_col"]
    # Check that the mapping is exact (no extra columns)
    columns_map = {"document": "base_document", "other_col": "duplicate_other_col"}
    assert len(result.column_names) == len(sample_dataset.column_names) + len(
        columns_map
    )


def test_empty_columns_map():
    """Test with empty columns map."""
    # New implementation doesn't allow empty columns_map
    with pytest.raises(Exception):  # ValidationError or similar
        DuplicateColumns(block_name="test_empty", columns_map={})


def test_nonexistent_column():
    """Test attempting to duplicate a non-existent column."""
    dataset = Dataset.from_dict({"existing_col": ["val1", "val2"]})
    block = DuplicateColumns(
        block_name="test_nonexistent", columns_map={"nonexistent_col": "new_col"}
    )

    with pytest.raises(
        ValueError, match="Source column 'nonexistent_col' not found in dataset"
    ):
        block.generate(dataset)


def test_duplicate_with_complex_data():
    """Test duplicating columns with complex data types."""
    dataset = Dataset.from_dict(
        {
            "numbers": [1, 2, 3],
            "lists": [[1, 2], [3, 4], [5, 6]],
            "dicts": [{"a": 1}, {"b": 2}, {"c": 3}],
        }
    )

    block = DuplicateColumns(
        block_name="test_complex",
        columns_map={
            "numbers": "duplicate_numbers",
            "lists": "duplicate_lists",
            "dicts": "duplicate_dicts",
        },
    )

    result = block.generate(dataset)

    # Check all columns exist
    assert "numbers" in result.column_names
    assert "lists" in result.column_names
    assert "dicts" in result.column_names
    assert "duplicate_numbers" in result.column_names
    assert "duplicate_lists" in result.column_names
    assert "duplicate_dicts" in result.column_names

    # Check values are correctly duplicated
    assert result["numbers"] == result["duplicate_numbers"]
    assert result["lists"] == result["duplicate_lists"]
    assert result["dicts"] == result["duplicate_dicts"]
