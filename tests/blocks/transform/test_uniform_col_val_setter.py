"""Tests for the SetToMajorityValue block."""

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.transform import UniformColumnValueSetter
from sdg_hub.core.utils.error_handling import EmptyDatasetError, MissingColumnError
import pytest


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
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["category"],
        output_cols=[],
        reduction_strategy="mode",
    )
    result = block.generate(sample_dataset)

    # Check that all values in category column are now "A" (the majority value)
    assert all(x == "A" for x in result["category"])
    # Check that other columns remain unchanged
    assert result["value"] == sample_dataset["value"]
    assert result["mixed"] == sample_dataset["mixed"]


def test_set_to_majority_numeric(sample_dataset):
    """Test setting numeric column to majority value."""
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["value"],
        output_cols=[],
        reduction_strategy="mode",
    )
    result = block.generate(sample_dataset)

    # Since all values are unique, the first value ("1") should be the majority
    assert all(x == "1" for x in result["value"])
    # Check that other columns remain unchanged
    assert result["category"] == sample_dataset["category"]
    assert result["mixed"] == sample_dataset["mixed"]


def test_set_to_majority_mixed_types(sample_dataset):
    """Test setting mixed type column to majority value."""
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["mixed"],
        output_cols=[],
        reduction_strategy="mode",
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
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["empty_col"],
        output_cols=[],
        reduction_strategy="mode",
    )

    # BaseBlock raises EmptyDatasetError for empty datasets
    with pytest.raises(EmptyDatasetError):
        block(dataset)  # Use __call__ to trigger BaseBlock validation


def test_set_to_majority_single_value():
    """Test behavior with column containing single value."""
    dataset = Dataset.from_dict({"single_col": ["A"]})
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["single_col"],
        output_cols=[],
        reduction_strategy="mode",
    )
    result = block.generate(dataset)

    assert all(x == "A" for x in result["single_col"])


def test_set_to_majority_all_unique():
    """Test behavior with column containing all unique values."""
    dataset = Dataset.from_dict({"unique_col": ["A", "B", "C"]})
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["unique_col"],
        output_cols=[],
        reduction_strategy="mode",
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
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["tie_col"],
        output_cols=[],
        reduction_strategy="mode",
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

    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["col1"],
        output_cols=[],
        reduction_strategy="mode",
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
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["missing_col"],
        output_cols=[],
        reduction_strategy="mode",
    )

    with pytest.raises(MissingColumnError):
        block(dataset)


def test_set_to_majority_validation_errors():
    """Test Pydantic validation errors in UniformColumnValueSetter."""
    # Test empty input_cols
    with pytest.raises(
        ValueError, match="UniformColumnValueSetter requires exactly one input column"
    ):
        UniformColumnValueSetter(
            block_name="test_block",
            input_cols=[],
            output_cols=[],
            reduction_strategy="mode",
        )

    # Test multiple input_cols
    with pytest.raises(
        ValueError, match="UniformColumnValueSetter requires exactly one input column"
    ):
        UniformColumnValueSetter(
            block_name="test_block",
            input_cols=["col1", "col2"],
            output_cols=[],
            reduction_strategy="mode",
        )


def test_set_to_majority_with_none_values():
    """Test behavior with None values in the column."""
    dataset = Dataset.from_dict({"col_with_none": ["A", None, "A", "B", None, "A"]})
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["col_with_none"],
        output_cols=[],
        reduction_strategy="mode",
    )
    result = block.generate(dataset)

    # "A" should be the majority value (appears 3 times)
    assert all(x == "A" for x in result["col_with_none"])


def test_reduction_strategy_min():
    """Test min reduction strategy."""
    dataset = Dataset.from_dict(
        {"numeric_col": [5, 2, 8, 1, 9, 3], "other_col": ["A", "B", "C", "D", "E", "F"]}
    )
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["numeric_col"],
        output_cols=[],
        reduction_strategy="min",
    )
    result = block.generate(dataset)

    # All values should be set to the minimum value (1)
    assert all(x == 1 for x in result["numeric_col"])
    # Other columns should remain unchanged
    assert result["other_col"] == dataset["other_col"]


def test_reduction_strategy_max():
    """Test max reduction strategy."""
    dataset = Dataset.from_dict(
        {"numeric_col": [5, 2, 8, 1, 9, 3], "other_col": ["A", "B", "C", "D", "E", "F"]}
    )
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["numeric_col"],
        output_cols=[],
        reduction_strategy="max",
    )
    result = block.generate(dataset)

    # All values should be set to the maximum value (9)
    assert all(x == 9 for x in result["numeric_col"])
    # Other columns should remain unchanged
    assert result["other_col"] == dataset["other_col"]


def test_reduction_strategy_mean():
    """Test mean reduction strategy."""
    dataset = Dataset.from_dict(
        {"numeric_col": [1, 2, 3, 4, 5], "other_col": ["A", "B", "C", "D", "E"]}
    )
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["numeric_col"],
        output_cols=[],
        reduction_strategy="mean",
    )
    result = block.generate(dataset)

    # All values should be set to the mean value (3.0)
    assert all(x == 3.0 for x in result["numeric_col"])
    # Other columns should remain unchanged
    assert result["other_col"] == dataset["other_col"]


def test_reduction_strategy_median():
    """Test median reduction strategy."""
    dataset = Dataset.from_dict(
        {"numeric_col": [1, 3, 5, 7, 9], "other_col": ["A", "B", "C", "D", "E"]}
    )
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["numeric_col"],
        output_cols=[],
        reduction_strategy="median",
    )
    result = block.generate(dataset)

    # All values should be set to the median value (5)
    assert all(x == 5 for x in result["numeric_col"])
    # Other columns should remain unchanged
    assert result["other_col"] == dataset["other_col"]


def test_reduction_strategy_median_even_length():
    """Test median reduction strategy with even number of values."""
    dataset = Dataset.from_dict(
        {
            "numeric_col": [1, 3, 5, 7, 9, 11],
            "other_col": ["A", "B", "C", "D", "E", "F"],
        }
    )
    block = UniformColumnValueSetter(
        block_name="test_block",
        input_cols=["numeric_col"],
        output_cols=[],
        reduction_strategy="median",
    )
    result = block.generate(dataset)

    # All values should be set to the median value (6.0 - average of 5 and 7)
    assert all(x == 6.0 for x in result["numeric_col"])
    # Other columns should remain unchanged
    assert result["other_col"] == dataset["other_col"]


def test_reduction_strategy_float_values():
    """Test reduction strategies with float values."""
    dataset = Dataset.from_dict(
        {"float_col": [1.5, 2.7, 3.2, 4.1, 5.9], "other_col": ["A", "B", "C", "D", "E"]}
    )

    # Test min
    block_min = UniformColumnValueSetter(
        block_name="test_min",
        input_cols=["float_col"],
        output_cols=[],
        reduction_strategy="min",
    )
    result_min = block_min.generate(dataset)
    assert all(x == 1.5 for x in result_min["float_col"])

    # Test max
    block_max = UniformColumnValueSetter(
        block_name="test_max",
        input_cols=["float_col"],
        output_cols=[],
        reduction_strategy="max",
    )
    result_max = block_max.generate(dataset)
    assert all(x == 5.9 for x in result_max["float_col"])

    # Test mean
    block_mean = UniformColumnValueSetter(
        block_name="test_mean",
        input_cols=["float_col"],
        output_cols=[],
        reduction_strategy="mean",
    )
    result_mean = block_mean.generate(dataset)
    expected_mean = sum([1.5, 2.7, 3.2, 4.1, 5.9]) / 5
    assert all(abs(x - expected_mean) < 1e-10 for x in result_mean["float_col"])


def test_reduction_strategy_mixed_numeric_types():
    """Test reduction strategies with mixed numeric types (int and float)."""
    dataset = Dataset.from_dict(
        {"mixed_numeric": [1, 2.5, 3, 4.7, 5], "other_col": ["A", "B", "C", "D", "E"]}
    )

    # Test min
    block_min = UniformColumnValueSetter(
        block_name="test_min",
        input_cols=["mixed_numeric"],
        output_cols=[],
        reduction_strategy="min",
    )
    result_min = block_min.generate(dataset)
    assert all(x == 1 for x in result_min["mixed_numeric"])

    # Test max
    block_max = UniformColumnValueSetter(
        block_name="test_max",
        input_cols=["mixed_numeric"],
        output_cols=[],
        reduction_strategy="max",
    )
    result_max = block_max.generate(dataset)
    assert all(x == 5 for x in result_max["mixed_numeric"])

    # Test mean
    block_mean = UniformColumnValueSetter(
        block_name="test_mean",
        input_cols=["mixed_numeric"],
        output_cols=[],
        reduction_strategy="mean",
    )
    result_mean = block_mean.generate(dataset)
    expected_mean = sum([1, 2.5, 3, 4.7, 5]) / 5
    assert all(abs(x - expected_mean) < 1e-10 for x in result_mean["mixed_numeric"])
