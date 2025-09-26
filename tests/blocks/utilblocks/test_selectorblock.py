"""Tests for the SelectorBlock class."""

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.deprecated_blocks import SelectorBlock
import pytest


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {
            "response": ["Response A", "Response B", "Response C"],
            "revised_response": ["Revised A", "Revised B", "Revised C"],
            "verdict": ["Assistant A", "Assistant B", "Assistant A"],
            "other_col": ["value1", "value2", "value3"],
        }
    )


def test_selector_block_basic(sample_dataset):
    """Test basic functionality of SelectorBlock."""
    block = SelectorBlock(
        block_name="test_selector",
        choice_map={"Assistant A": "response", "Assistant B": "revised_response"},
        choice_col="verdict",
        output_col="chosen_response",
    )

    result = block.generate(sample_dataset)

    # Check that the selection worked correctly
    assert "chosen_response" in result.column_names
    assert result[0]["chosen_response"] == "Response A"  # Assistant A chose response
    assert (
        result[1]["chosen_response"] == "Revised B"
    )  # Assistant B chose revised_response
    assert result[2]["chosen_response"] == "Response C"  # Assistant A chose response

    # Check that original columns are preserved
    assert "response" in result.column_names
    assert "revised_response" in result.column_names
    assert "verdict" in result.column_names
    assert "other_col" in result.column_names


def test_selector_block_invalid_choice(sample_dataset):
    """Test SelectorBlock with invalid choice values."""
    block = SelectorBlock(
        block_name="test_selector",
        choice_map={"Assistant A": "response", "Assistant B": "revised_response"},
        choice_col="verdict",
        output_col="chosen_response",
    )

    # Create dataset with invalid choice
    invalid_dataset = Dataset.from_dict(
        {
            "response": ["Response A"],
            "revised_response": ["Revised A"],
            "verdict": ["Invalid Choice"],
            "other_col": ["value1"],
        }
    )

    with pytest.raises(KeyError):
        block.generate(invalid_dataset)


def test_selector_block_empty_dataset():
    """Test SelectorBlock with empty dataset."""
    block = SelectorBlock(
        block_name="test_selector",
        choice_map={"Assistant A": "response", "Assistant B": "revised_response"},
        choice_col="verdict",
        output_col="chosen_response",
    )

    empty_dataset = Dataset.from_dict(
        {"response": [], "revised_response": [], "verdict": [], "other_col": []}
    )

    result = block.generate(empty_dataset)
    assert len(result) == 0


def test_selector_block_custom_num_procs(sample_dataset):
    """Test SelectorBlock with custom number of processes."""
    block = SelectorBlock(
        block_name="test_selector",
        choice_map={"Assistant A": "response", "Assistant B": "revised_response"},
        choice_col="verdict",
        output_col="chosen_response",
        num_procs=2,
    )

    result = block.generate(sample_dataset)
    assert len(result) == 3
    assert "chosen_response" in result.column_names
    assert result[0]["chosen_response"] == "Response A"

    # Check that original columns are preserved
    assert "response" in result.column_names
    assert "revised_response" in result.column_names
    assert "verdict" in result.column_names
    assert "other_col" in result.column_names


def test_selector_block_empty_choice_map(sample_dataset):
    """Test SelectorBlock with an empty choice map.

    Verifies that the block raises a ValueError when trying to use an empty choice map,
    as there would be no valid mappings to look up.
    """
    with pytest.raises(ValueError, match="choice_map cannot be empty"):
        SelectorBlock(
            block_name="test_selector",
            choice_map={},  # Empty choice map
            choice_col="verdict",
            output_col="chosen_response",
        )


def test_selector_block_nonexistent_choice_map_columns(sample_dataset):
    """Test SelectorBlock with non-existent column names in choice map.

    Verifies that the block properly handles and raises MissingColumnError when
    attempting to map to columns that don't exist in the dataset.
    """
    block = SelectorBlock(
        block_name="test_selector",
        choice_map={
            "Assistant A": "non_existent_col1",
            "Assistant B": "non_existent_col2",
        },
        choice_col="verdict",
        output_col="chosen_response",
    )

    with pytest.raises(Exception):  # MissingColumnError or similar
        block.generate(sample_dataset)
