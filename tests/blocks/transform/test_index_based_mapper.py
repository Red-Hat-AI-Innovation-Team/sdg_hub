"""Tests for the IndexBasedMapperBlock class."""

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.transform import IndexBasedMapperBlock
from sdg_hub.core.utils.error_handling import MissingColumnError
import pytest


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing with multiple response and verdict columns."""
    return Dataset.from_dict(
        {
            "response_1": ["Response A1", "Response B1", "Response C1", "Response D1"],
            "response_2": ["Response A2", "Response B2", "Response C2", "Response D2"],
            "response_3": ["Response A3", "Response B3", "Response C3", "Response D3"],
            "verdict_1": ["Assistant A", "Assistant B", "Assistant C", "Assistant A"],
            "verdict_2": ["Assistant B", "Assistant C", "Assistant A", "Assistant C"],
            "other_col": ["value1", "value2", "value3", "value4"],
        }
    )


@pytest.fixture
def choice_map():
    """Standard choice mapping for tests."""
    return {
        "Assistant A": "response_1",
        "Assistant B": "response_2",
        "Assistant C": "response_3",
    }


def test_index_based_mapper_basic(sample_dataset, choice_map):
    """Test basic functionality of IndexBasedMapperBlock."""
    block = IndexBasedMapperBlock(
        block_name="test_mapper",
        input_cols=["response_1", "response_2", "response_3", "verdict_1", "verdict_2"],
        output_cols=["selected_1", "selected_2"],
        choice_map=choice_map,
        choice_cols=["verdict_1", "verdict_2"],
    )

    result = block.generate(sample_dataset)

    # Check that the selection worked correctly
    assert "selected_1" in result.column_names
    assert "selected_2" in result.column_names

    # Row 0: verdict_1=Assistant A -> response_1, verdict_2=Assistant B -> response_2
    assert result[0]["selected_1"] == "Response A1"
    assert result[0]["selected_2"] == "Response A2"

    # Row 1: verdict_1=Assistant B -> response_2, verdict_2=Assistant C -> response_3
    assert result[1]["selected_1"] == "Response B2"
    assert result[1]["selected_2"] == "Response B3"

    # Row 2: verdict_1=Assistant C -> response_3, verdict_2=Assistant A -> response_1
    assert result[2]["selected_1"] == "Response C3"
    assert result[2]["selected_2"] == "Response C1"

    # Check that original columns are preserved
    assert "response_1" in result.column_names
    assert "response_2" in result.column_names
    assert "response_3" in result.column_names
    assert "verdict_1" in result.column_names
    assert "verdict_2" in result.column_names
    assert "other_col" in result.column_names


def test_index_based_mapper_single_choice_col(choice_map):
    """Test IndexBasedMapperBlock with single choice column (backward compatibility)."""
    dataset = Dataset.from_dict(
        {
            "response_1": ["Response A", "Response B", "Response C"],
            "response_2": ["Response A2", "Response B2", "Response C2"],
            "response_3": ["Response A3", "Response B3", "Response C3"],
            "verdict": ["Assistant A", "Assistant B", "Assistant C"],
        }
    )

    block = IndexBasedMapperBlock(
        block_name="test_mapper",
        input_cols=["response_1", "response_2", "response_3", "verdict"],
        output_cols=["selected_response"],
        choice_map=choice_map,
        choice_cols=["verdict"],
    )

    result = block.generate(dataset)

    assert "selected_response" in result.column_names
    assert result[0]["selected_response"] == "Response A"
    assert result[1]["selected_response"] == "Response B2"
    assert result[2]["selected_response"] == "Response C3"


def test_index_based_mapper_invalid_choice():
    """Test IndexBasedMapperBlock with invalid choice values."""
    # Create dataset with invalid choice
    invalid_dataset = Dataset.from_dict(
        {
            "response_1": ["Response A"],
            "response_2": ["Response B"],
            "verdict": ["Invalid Choice"],
        }
    )

    block = IndexBasedMapperBlock(
        block_name="test_mapper",
        input_cols=["response_1", "response_2", "verdict"],
        output_cols=["selected_response"],
        choice_map={"Assistant A": "response_1", "Assistant B": "response_2"},
        choice_cols=["verdict"],
    )

    # Should raise ValueError for unmapped choice values during validation
    with pytest.raises(ValueError, match="Choice values.*not found in choice_map"):
        block(invalid_dataset)


def test_index_based_mapper_empty_dataset(choice_map):
    """Test IndexBasedMapperBlock with empty dataset."""
    block = IndexBasedMapperBlock(
        block_name="test_mapper",
        input_cols=["response_1", "response_2", "response_3", "verdict_1", "verdict_2"],
        output_cols=["selected_1", "selected_2"],
        choice_map=choice_map,
        choice_cols=["verdict_1", "verdict_2"],
    )

    empty_dataset = Dataset.from_dict(
        {
            "response_1": [],
            "response_2": [],
            "response_3": [],
            "verdict_1": [],
            "verdict_2": [],
        }
    )

    result = block.generate(empty_dataset)
    assert len(result) == 0


def test_index_based_mapper_missing_choice_columns():
    """Test IndexBasedMapperBlock with missing choice columns."""
    dataset = Dataset.from_dict(
        {
            "response_1": ["Response A"],
            "response_2": ["Response B"],
            # Missing verdict_1 and verdict_2 columns
        }
    )

    block = IndexBasedMapperBlock(
        block_name="test_mapper",
        input_cols=["response_1", "response_2", "verdict_1", "verdict_2"],
        output_cols=["selected_1", "selected_2"],
        choice_map={"Assistant A": "response_1", "Assistant B": "response_2"},
        choice_cols=["verdict_1", "verdict_2"],
    )

    # Should raise MissingColumnError for missing choice columns during validation
    with pytest.raises(MissingColumnError):
        block(dataset)


def test_index_based_mapper_missing_mapped_columns():
    """Test IndexBasedMapperBlock with missing mapped columns."""
    dataset = Dataset.from_dict(
        {
            "response_1": ["Response A"],
            "verdict_1": ["Assistant A"],
            "verdict_2": ["Assistant B"],
            # Missing response_2 and response_3 columns
        }
    )

    block = IndexBasedMapperBlock(
        block_name="test_mapper",
        input_cols=["response_1", "response_2", "response_3", "verdict_1", "verdict_2"],
        output_cols=["selected_1", "selected_2"],
        choice_map={
            "Assistant A": "response_1",
            "Assistant B": "response_2",
            "Assistant C": "response_3",
        },
        choice_cols=["verdict_1", "verdict_2"],
    )

    # Should raise MissingColumnError for missing mapped columns during validation
    with pytest.raises(MissingColumnError):
        block(dataset)


def test_index_based_mapper_validation_errors():
    """Test IndexBasedMapperBlock validation errors."""
    # Test empty choice_map
    with pytest.raises(ValueError, match="choice_map cannot be empty"):
        IndexBasedMapperBlock(
            block_name="test_mapper",
            input_cols=["response_1", "verdict_1"],
            output_cols=["selected_1"],
            choice_map={},
            choice_cols=["verdict_1"],
        )

    # Test empty choice_cols
    with pytest.raises(ValueError, match="choice_cols cannot be empty"):
        IndexBasedMapperBlock(
            block_name="test_mapper",
            input_cols=["response_1", "verdict_1"],
            output_cols=["selected_1"],
            choice_map={"Assistant A": "response_1"},
            choice_cols=[],
        )

    # Test mismatched choice_cols and output_cols length
    with pytest.raises(
        ValueError, match="choice_cols and output_cols must have same length"
    ):
        IndexBasedMapperBlock(
            block_name="test_mapper",
            input_cols=["response_1", "verdict_1", "verdict_2"],
            output_cols=["selected_1"],  # Only 1 output col
            choice_map={"Assistant A": "response_1"},
            choice_cols=["verdict_1", "verdict_2"],  # But 2 choice cols
        )


def test_index_based_mapper_complex_scenario():
    """Test IndexBasedMapperBlock with complex multi-column scenario."""
    dataset = Dataset.from_dict(
        {
            "model_a_response": ["Model A says X", "Model A says Y", "Model A says Z"],
            "model_b_response": ["Model B says X", "Model B says Y", "Model B says Z"],
            "model_c_response": ["Model C says X", "Model C says Y", "Model C says Z"],
            "human_response": ["Human says X", "Human says Y", "Human says Z"],
            "judge_1_choice": ["model_a", "model_b", "human"],
            "judge_2_choice": ["model_c", "human", "model_a"],
            "judge_3_choice": ["human", "model_c", "model_b"],
        }
    )

    choice_map = {
        "model_a": "model_a_response",
        "model_b": "model_b_response",
        "model_c": "model_c_response",
        "human": "human_response",
    }

    block = IndexBasedMapperBlock(
        block_name="multi_judge_mapper",
        input_cols=[
            "model_a_response",
            "model_b_response",
            "model_c_response",
            "human_response",
            "judge_1_choice",
            "judge_2_choice",
            "judge_3_choice",
        ],
        output_cols=["judge_1_selection", "judge_2_selection", "judge_3_selection"],
        choice_map=choice_map,
        choice_cols=["judge_1_choice", "judge_2_choice", "judge_3_choice"],
    )

    result = block.generate(dataset)

    # Check results
    assert result[0]["judge_1_selection"] == "Model A says X"  # judge_1_choice=model_a
    assert result[0]["judge_2_selection"] == "Model C says X"  # judge_2_choice=model_c
    assert result[0]["judge_3_selection"] == "Human says X"  # judge_3_choice=human

    assert result[1]["judge_1_selection"] == "Model B says Y"  # judge_1_choice=model_b
    assert result[1]["judge_2_selection"] == "Human says Y"  # judge_2_choice=human
    assert result[1]["judge_3_selection"] == "Model C says Y"  # judge_3_choice=model_c

    assert result[2]["judge_1_selection"] == "Human says Z"  # judge_1_choice=human
    assert result[2]["judge_2_selection"] == "Model A says Z"  # judge_2_choice=model_a
    assert result[2]["judge_3_selection"] == "Model B says Z"  # judge_3_choice=model_b


def test_index_based_mapper_choice_to_output_mapping():
    """Test that choice_to_output_map is created correctly."""
    block = IndexBasedMapperBlock(
        block_name="test_mapper",
        input_cols=["response_1", "response_2", "verdict_1", "verdict_2"],
        output_cols=["selected_1", "selected_2"],
        choice_map={"Assistant A": "response_1", "Assistant B": "response_2"},
        choice_cols=["verdict_1", "verdict_2"],
    )

    expected_mapping = {
        "verdict_1": "selected_1",
        "verdict_2": "selected_2",
    }

    assert block.choice_to_output_map == expected_mapping
