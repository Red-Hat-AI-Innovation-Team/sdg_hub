"""Tests for the RowMultiplierBlock functionality.

This module contains tests that verify the correct behavior of the RowMultiplierBlock,
including row duplication, shuffling, and edge case handling.
"""

# Third Party
# First Party
import pandas as pd
import pytest

from sdg_hub.core.blocks.transform import RowMultiplierBlock
from sdg_hub.core.utils.error_handling import EmptyDatasetError


def test_basic_multiplication():
    """Test basic row duplication functionality."""
    data = {
        "config": ["A", "B", "C"],
        "prompt": ["p1", "p2", "p3"],
    }
    dataset = pd.DataFrame(data)

    block = RowMultiplierBlock(
        block_name="test_multiply",
        num_samples=3,
    )

    result = block(dataset)

    # 3 rows * 3 samples = 9 rows
    assert len(result) == 9
    assert "config" in result.columns
    assert "prompt" in result.columns

    # Verify each original row appears num_samples times
    assert result["config"].tolist() == ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
    assert result["prompt"].tolist() == [
        "p1",
        "p1",
        "p1",
        "p2",
        "p2",
        "p2",
        "p3",
        "p3",
        "p3",
    ]


def test_num_samples_one_returns_equivalent():
    """Test that num_samples=1 returns an equivalent dataset."""
    data = {
        "id": [1, 2, 3],
        "value": ["a", "b", "c"],
    }
    dataset = pd.DataFrame(data)

    block = RowMultiplierBlock(
        block_name="test_no_multiply",
        num_samples=1,
    )

    result = block(dataset)

    assert len(result) == 3
    assert result["id"].tolist() == [1, 2, 3]
    assert result["value"].tolist() == ["a", "b", "c"]


def test_shuffle_with_reproducible_seed():
    """Test that shuffle with random_seed produces reproducible results."""
    data = {
        "config": ["A", "B", "C"],
    }
    dataset = pd.DataFrame(data)

    block1 = RowMultiplierBlock(
        block_name="test_shuffle1",
        num_samples=3,
        shuffle=True,
        random_seed=42,
    )

    block2 = RowMultiplierBlock(
        block_name="test_shuffle2",
        num_samples=3,
        shuffle=True,
        random_seed=42,
    )

    result1 = block1(dataset)
    result2 = block2(dataset)

    # Same seed should produce same order
    assert result1["config"].tolist() == result2["config"].tolist()


def test_shuffle_changes_order():
    """Test that shuffle actually changes the row order."""
    data = {
        "config": ["A", "B", "C", "D", "E"],
    }
    dataset = pd.DataFrame(data)

    block = RowMultiplierBlock(
        block_name="test_shuffle_changes",
        num_samples=5,
        shuffle=True,
        random_seed=42,
    )

    result = block(dataset)

    # Without shuffle, order would be ["A", "A", "A", "A", "A", "B", ...]
    unshuffled_order = ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5 + ["E"] * 5

    # Shuffled order should be different
    assert result["config"].tolist() != unshuffled_order


def test_validation_error_num_samples_less_than_one():
    """Test that num_samples < 1 raises validation error."""
    with pytest.raises(ValueError):
        RowMultiplierBlock(
            block_name="test_invalid",
            num_samples=0,
        )

    with pytest.raises(ValueError):
        RowMultiplierBlock(
            block_name="test_invalid",
            num_samples=-1,
        )


def test_empty_dataset_error():
    """Test that empty dataset raises EmptyDatasetError."""
    data = {"config": [], "prompt": []}
    dataset = pd.DataFrame(data)

    block = RowMultiplierBlock(
        block_name="test_empty",
        num_samples=3,
    )

    with pytest.raises(EmptyDatasetError):
        block(dataset)


def test_data_type_preservation():
    """Test that data types are preserved after multiplication."""
    data = {
        "int_col": [1, 2],
        "float_col": [1.5, 2.5],
        "str_col": ["a", "b"],
        "bool_col": [True, False],
    }
    dataset = pd.DataFrame(data)

    block = RowMultiplierBlock(
        block_name="test_types",
        num_samples=2,
    )

    result = block(dataset)

    assert result["int_col"].dtype == dataset["int_col"].dtype
    assert result["float_col"].dtype == dataset["float_col"].dtype
    assert result["str_col"].dtype == dataset["str_col"].dtype
    assert result["bool_col"].dtype == dataset["bool_col"].dtype


def test_none_nan_value_preservation():
    """Test that None/NaN values are preserved after multiplication."""
    data = {
        "col1": [1, None, 3],
        "col2": ["a", "b", None],
    }
    dataset = pd.DataFrame(data)

    block = RowMultiplierBlock(
        block_name="test_nulls",
        num_samples=2,
    )

    result = block(dataset)

    # Check that null values are preserved
    assert result["col1"].isna().sum() == 2  # None duplicated twice
    assert result["col2"].isna().sum() == 2  # None duplicated twice


def test_single_row_multiplication():
    """Test multiplication of a single row dataset."""
    data = {
        "config": ["only_one"],
    }
    dataset = pd.DataFrame(data)

    block = RowMultiplierBlock(
        block_name="test_single",
        num_samples=5,
    )

    result = block(dataset)

    assert len(result) == 5
    assert result["config"].tolist() == ["only_one"] * 5


def test_large_num_samples():
    """Test with a larger number of samples."""
    data = {
        "id": [1],
    }
    dataset = pd.DataFrame(data)

    block = RowMultiplierBlock(
        block_name="test_large",
        num_samples=100,
    )

    result = block(dataset)

    assert len(result) == 100
    assert all(result["id"] == 1)


def test_duplicate_index_labels():
    """Test that duplication works correctly with duplicate index labels."""
    data = {"config": ["A", "B", "C"]}
    dataset = pd.DataFrame(data, index=[0, 0, 1])  # Duplicate index labels

    block = RowMultiplierBlock(
        block_name="test_dup_index",
        num_samples=2,
    )

    result = block(dataset)

    assert len(result) == 6
    assert result["config"].tolist() == ["A", "A", "B", "B", "C", "C"]
