"""Tests for TextConcatBlock."""

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.transform import TextConcatBlock
import pytest


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {
            "context": ["Context 1", "Context 2", "Context 3"],
            "question": ["Question 1", "Question 2", "Question 3"],
            "other_col": ["Other 1", "Other 2", "Other 3"],
        }
    )


def test_basic_text_concat(sample_dataset):
    """Test basic text concatenation with default separator."""
    block = TextConcatBlock(
        block_name="test_concat",
        input_cols=["context", "question"],
        output_cols=["combined"],
    )

    result = block.generate(sample_dataset)

    assert "combined" in result.column_names
    assert result[0]["combined"] == "Context 1\n\nQuestion 1"
    assert result[1]["combined"] == "Context 2\n\nQuestion 2"
    assert result[2]["combined"] == "Context 3\n\nQuestion 3"


def test_custom_separator(sample_dataset):
    """Test text concatenation with custom separator."""
    block = TextConcatBlock(
        block_name="test_concat",
        input_cols=["context", "question"],
        output_cols=["combined"],
        separator=" | ",
    )

    result = block.generate(sample_dataset)

    assert result[0]["combined"] == "Context 1 | Question 1"
    assert result[1]["combined"] == "Context 2 | Question 2"
    assert result[2]["combined"] == "Context 3 | Question 3"


def test_multiple_columns(sample_dataset):
    """Test concatenating more than two columns."""
    block = TextConcatBlock(
        block_name="test_concat",
        input_cols=["context", "question", "other_col"],
        output_cols=["combined"],
    )

    result = block.generate(sample_dataset)

    assert result[0]["combined"] == "Context 1\n\nQuestion 1\n\nOther 1"
    assert result[1]["combined"] == "Context 2\n\nQuestion 2\n\nOther 2"
    assert result[2]["combined"] == "Context 3\n\nQuestion 3\n\nOther 3"


def test_empty_input_cols():
    """Test validation with empty input_cols."""
    with pytest.raises(ValueError, match="input_cols cannot be empty"):
        TextConcatBlock(
            block_name="test_concat",
            input_cols=[],
            output_cols=["combined"],
        )


def test_invalid_output_cols():
    """Test validation with invalid output_cols."""
    with pytest.raises(
        ValueError, match="TextConcatBlock requires exactly one output column"
    ):
        TextConcatBlock(
            block_name="test_concat",
            input_cols=["col1", "col2"],
            output_cols=["out1", "out2"],  # Should be exactly one
        )


def test_missing_columns():
    """Test behavior when specified columns don't exist in dataset."""
    dataset = Dataset.from_dict({"existing_col": ["Value 1", "Value 2"]})

    block = TextConcatBlock(
        block_name="test_concat",
        input_cols=["existing_col", "missing_col"],
        output_cols=["combined"],
    )

    with pytest.raises(
        ValueError, match="Input column 'missing_col' not found in sample"
    ):
        block.generate(dataset)


def test_non_string_values():
    """Test concatenating columns with non-string values."""
    dataset = Dataset.from_dict({"num_col": [1, 2, 3], "bool_col": [True, False, True]})

    block = TextConcatBlock(
        block_name="test_concat",
        input_cols=["num_col", "bool_col"],
        output_cols=["combined"],
    )

    result = block.generate(dataset)

    assert result[0]["combined"] == "1\n\nTrue"
    assert result[1]["combined"] == "2\n\nFalse"
    assert result[2]["combined"] == "3\n\nTrue"
