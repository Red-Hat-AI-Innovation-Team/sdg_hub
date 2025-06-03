"""Tests for CombineColumnsBlock."""

# Third Party
from datasets import Dataset
import pytest

# First Party
from sdg_hub.blocks.utilblocks import CombineColumnsBlock


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


def test_basic_column_combination(sample_dataset):
    """Test basic column combination with default separator."""
    block = CombineColumnsBlock(
        block_name="test_combine",
        columns=["context", "question"],
        output_col="combined",
    )

    result = block.generate(sample_dataset)

    assert "combined" in result.column_names
    assert result[0]["combined"] == "Context 1\n\nQuestion 1"
    assert result[1]["combined"] == "Context 2\n\nQuestion 2"
    assert result[2]["combined"] == "Context 3\n\nQuestion 3"


def test_custom_separator(sample_dataset):
    """Test column combination with custom separator."""
    block = CombineColumnsBlock(
        block_name="test_combine",
        columns=["context", "question"],
        output_col="combined",
        separator=" | ",
    )

    result = block.generate(sample_dataset)

    assert result[0]["combined"] == "Context 1 | Question 1"
    assert result[1]["combined"] == "Context 2 | Question 2"
    assert result[2]["combined"] == "Context 3 | Question 3"


def test_multiple_columns(sample_dataset):
    """Test combining more than two columns."""
    block = CombineColumnsBlock(
        block_name="test_combine",
        columns=["context", "question", "other_col"],
        output_col="combined",
    )

    result = block.generate(sample_dataset)

    assert result[0]["combined"] == "Context 1\n\nQuestion 1\n\nOther 1"
    assert result[1]["combined"] == "Context 2\n\nQuestion 2\n\nOther 2"
    assert result[2]["combined"] == "Context 3\n\nQuestion 3\n\nOther 3"


def test_empty_columns(sample_dataset):
    """Test combining columns with empty values."""
    dataset = Dataset.from_dict(
        {"context": ["", "Context 2", ""], "question": ["Question 1", "", "Question 3"]}
    )

    block = CombineColumnsBlock(
        block_name="test_combine",
        columns=["context", "question"],
        output_col="combined",
    )

    result = block.generate(dataset)

    assert result[0]["combined"] == "\n\nQuestion 1"
    assert result[1]["combined"] == "Context 2\n\n"
    assert result[2]["combined"] == "\n\nQuestion 3"


def test_non_string_values(sample_dataset):
    """Test combining columns with non-string values."""
    dataset = Dataset.from_dict({"num_col": [1, 2, 3], "bool_col": [True, False, True]})

    block = CombineColumnsBlock(
        block_name="test_combine",
        columns=["num_col", "bool_col"],
        output_col="combined",
    )

    result = block.generate(dataset)

    assert result[0]["combined"] == "1\n\nTrue"
    assert result[1]["combined"] == "2\n\nFalse"
    assert result[2]["combined"] == "3\n\nTrue"


def test_batch_processing(sample_dataset):
    """Test batch processing functionality."""
    block = CombineColumnsBlock(
        block_name="test_combine",
        columns=["context", "question"],
        output_col="combined",
        num_procs=2,
    )

    result = block.generate(sample_dataset)

    assert len(result) == len(sample_dataset)
    assert "combined" in result.column_names
    assert result[0]["combined"] == "Context 1\n\nQuestion 1"


def test_multiple_columns_custom_separator(sample_dataset):
    """Test combining multiple columns with custom separator."""
    block = CombineColumnsBlock(
        block_name="test_combine",
        columns=["context", "question", "other_col"],
        output_col="combined",
        separator=" | ",
    )

    result = block.generate(sample_dataset)

    assert result[0]["combined"] == "Context 1 | Question 1 | Other 1"
    assert result[1]["combined"] == "Context 2 | Question 2 | Other 2"
    assert result[2]["combined"] == "Context 3 | Question 3 | Other 3"


def test_none_values():
    """Test combining columns with None values."""
    dataset = Dataset.from_dict(
        {"col1": ["Value 1", None, "Value 3"], "col2": [None, "Value 2", "Value 3"]}
    )

    block = CombineColumnsBlock(
        block_name="test_combine", columns=["col1", "col2"], output_col="combined"
    )

    result = block.generate(dataset)

    assert result[0]["combined"] == "Value 1\n\nNone"
    assert result[1]["combined"] == "None\n\nValue 2"
    assert result[2]["combined"] == "Value 3\n\nValue 3"


def test_missing_columns():
    """Test behavior when specified columns don't exist in dataset."""
    dataset = Dataset.from_dict({"existing_col": ["Value 1", "Value 2"]})

    block = CombineColumnsBlock(
        block_name="test_combine",
        columns=["existing_col", "missing_col"],
        output_col="combined",
    )

    with pytest.raises(KeyError):
        block.generate(dataset)
