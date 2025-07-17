"""Tests for the SetToMajorityValue block."""

# Third Party
from datasets import Dataset
import pytest

# First Party
from sdg_hub.blocks.transform import SetToMajorityValue
from sdg_hub.utils.error_handling import EmptyDatasetError, MissingColumnError


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {
            "category": ["A", "B", "A", "C", "A", "B"],
            "value": ["1", "2", "3", "4", "5", "6"],
            "mixed": ["A", "1", "A", "2", "A", "3"],
        }
    )


def test_set_to_majority_basic(sample_dataset):
    """Test basic functionality of setting column to majority value."""
    block = SetToMajorityValue(
        block_name="test_block",
        input_cols=["category"],
        output_cols=[],
    )
    result = block.generate(sample_dataset)

    # Check that all values in category column are now "A" (the majority value)
    assert all(x == "A" for x in result["category"])
    # Check that other columns remain unchanged
    assert result["value"] == sample_dataset["value"]
    assert result["mixed"] == sample_dataset["mixed"]


def test_set_to_majority_numeric(sample_dataset):
    """Test setting numeric column to majority value."""
    block = SetToMajorityValue(
        block_name="test_block",
        input_cols=["value"],
        output_cols=[],
    )
    result = block.generate(sample_dataset)

    # Since all values are unique, the first value ("1") should be the majority
    assert all(x == "1" for x in result["value"])
    # Check that other columns remain unchanged
    assert result["category"] == sample_dataset["category"]
    assert result["mixed"] == sample_dataset["mixed"]


def test_set_to_majority_mixed_types(sample_dataset):
    """Test setting mixed type column to majority value."""
    block = SetToMajorityValue(
        block_name="test_block",
        input_cols=["mixed"],
        output_cols=[],
    )
    result = block.generate(sample_dataset)

    # "A" is the majority value in mixed column
    assert all(x == "A" for x in result["mixed"])
    # Check that other columns remain unchanged
    assert result["category"] == sample_dataset["category"]
    assert result["value"] == sample_dataset["value"]


def test_set_to_majority_empty_column():
    """Test behavior with empty column."""
    dataset = Dataset.from_dict({"empty_col": []})
    block = SetToMajorityValue(
        block_name="test_block",
        input_cols=["empty_col"],
        output_cols=[],
    )

    # BaseBlock raises EmptyDatasetError for empty datasets
    with pytest.raises(EmptyDatasetError):
        block(dataset)  # Use __call__ to trigger BaseBlock validation


def test_set_to_majority_single_value():
    """Test behavior with column containing single value."""
    dataset = Dataset.from_dict({"single_col": ["A"]})
    block = SetToMajorityValue(
        block_name="test_block",
        input_cols=["single_col"],
        output_cols=[],
    )
    result = block.generate(dataset)

    assert all(x == "A" for x in result["single_col"])


def test_set_to_majority_all_unique():
    """Test behavior with column containing all unique values."""
    dataset = Dataset.from_dict({"unique_col": ["A", "B", "C"]})
    block = SetToMajorityValue(
        block_name="test_block",
        input_cols=["unique_col"],
        output_cols=[],
    )
    result = block.generate(dataset)

    # Should use the first value as majority when all values are unique
    assert all(x == "A" for x in result["unique_col"])


def test_set_to_majority_tie_handling():
    """Test behavior when there are multiple values with the same frequency."""
    dataset = Dataset.from_dict(
        {
            "tie_col": ["A", "B", "A", "B", "C", "D"],
            "other_col": ["1", "2", "3", "4", "5", "6"],
        }
    )
    block = SetToMajorityValue(
        block_name="test_block",
        input_cols=["tie_col"],
        output_cols=[],
    )
    result = block.generate(dataset)

    # When there's a tie, pandas.mode() returns the first value it encounters
    # In this case, "A" should be chosen as it appears first in the dataset
    assert all(x == "A" for x in result["tie_col"])
    assert result["other_col"] == dataset["other_col"]


def test_dataset_structure_preservation():
    """Test that the output dataset maintains the same structure as input."""
    input_dataset = Dataset.from_dict(
        {"col1": ["A", "B", "A"], "col2": ["1", "2", "3"], "col3": ["X", "Y", "Z"]}
    )

    block = SetToMajorityValue(
        block_name="test_block",
        input_cols=["col1"],
        output_cols=[],
    )
    result = block.generate(input_dataset)

    # Check column names are preserved
    assert set(result.column_names) == set(input_dataset.column_names)

    # Check number of rows is preserved
    assert len(result) == len(input_dataset)

    # Check data types are preserved
    assert result.features == input_dataset.features

    # Check target column is set to majority value
    assert all(x == "A" for x in result["col1"])

    # Check other columns remain unchanged
    assert result["col2"] == input_dataset["col2"]
    assert result["col3"] == input_dataset["col3"]


def test_set_to_majority_missing_column():
    """Test behavior with missing column."""
    dataset = Dataset.from_dict({"col1": ["A", "B", "C"]})
    block = SetToMajorityValue(
        block_name="test_block",
        input_cols=["missing_col"],
        output_cols=[],
    )

    with pytest.raises(MissingColumnError):
        block(dataset)


def test_set_to_majority_validation_errors():
    """Test Pydantic validation errors in SetToMajorityValue."""
    
    # Test empty input_cols
    with pytest.raises(ValueError, match="SetToMajorityValue requires exactly one input column"):
        SetToMajorityValue(
            block_name="test_block",
            input_cols=[],
            output_cols=[],
        )
    
    # Test multiple input_cols
    with pytest.raises(ValueError, match="SetToMajorityValue requires exactly one input column"):
        SetToMajorityValue(
            block_name="test_block",
            input_cols=["col1", "col2"],
            output_cols=[],
        )


def test_set_to_majority_with_none_values():
    """Test behavior with None values in the column."""
    dataset = Dataset.from_dict({"col_with_none": ["A", None, "A", "B", None, "A"]})
    block = SetToMajorityValue(
        block_name="test_block",
        input_cols=["col_with_none"],
        output_cols=[],
    )
    result = block.generate(dataset)

    # "A" should be the majority value (appears 3 times)
    assert all(x == "A" for x in result["col_with_none"])