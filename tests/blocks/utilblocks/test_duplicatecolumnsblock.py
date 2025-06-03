"""Tests for the DuplicateColumns block functionality."""

# Third Party
from datasets import Dataset
import pytest

# First Party
from sdg_hub.blocks.utilblocks import DuplicateColumns


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


def test_empty_columns_map(sample_dataset):
    """Test with empty columns map."""
    block = DuplicateColumns(block_name="test_empty", columns_map={})

    result = block.generate(sample_dataset)

    # Check that dataset remains unchanged
    assert set(result.column_names) == set(sample_dataset.column_names)
    assert result["document"] == sample_dataset["document"]
    assert result["other_col"] == sample_dataset["other_col"]


def test_nonexistent_column():
    """Test attempting to duplicate a non-existent column."""
    dataset = Dataset.from_dict({"existing_col": ["val1", "val2"]})
    block = DuplicateColumns(
        block_name="test_nonexistent", columns_map={"nonexistent_col": "new_col"}
    )

    with pytest.raises(KeyError):
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


def test_validate_column_mapping_and_values():
    """Test that columns are duplicated exactly as specified in columns_map and contain identical values."""
    # Create a dataset with consistent data types
    dataset = Dataset.from_dict(
        {
            "text": ["hello", "world", "test"],
            "number": ["1", "2", "3"],  # Using strings instead of integers
            "boolean": ["True", "False", "True"],  # Using strings instead of booleans
            "mixed": ["text", "42", "True"],  # All values as strings
        }
    )
    
    # Define the mapping
    columns_map = {
        "text": "duplicated_text",
        "number": "duplicated_number",
        "boolean": "duplicated_boolean",
        "mixed": "duplicated_mixed",
    }
    
    block = DuplicateColumns(block_name="test_mapping", columns_map=columns_map)
    
    result = block.generate(dataset)
    
    # Validate column mapping
    for original_col, new_col in columns_map.items():
        # Check that both original and new columns exist
        assert original_col in result.column_names, (
            f"Original column {original_col} not found"
        )
        assert new_col in result.column_names, f"New column {new_col} not found"
        
        # Check that values are identical
        original_values = result[original_col]
        new_values = result[new_col]
        assert original_values == new_values, (
            f"Values in {original_col} and {new_col} are not identical"
        )
        
        # Check that the mapping is exact (no extra columns)
        assert len(result.column_names) == len(dataset.column_names) + len(columns_map)
