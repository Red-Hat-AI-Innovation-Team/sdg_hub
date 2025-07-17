"""Tests for the FilterByValueBlock implementation."""

# Standard
import operator
import pytest

# Third Party
from datasets import Dataset, Features, Value

# First Party
from sdg_hub.blocks import FilterByValueBlock


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {"age": ["25", "30", "35", "forty", "45"]},
        features=Features({"age": Value("string")}),
    )


@pytest.fixture
def filter_block():
    """Create a basic FilterByValueBlock instance."""
    return FilterByValueBlock(
        block_name="test_filter",
        filter_column="age",
        filter_value=30,
        operation=operator.eq,
        convert_dtype=int,
    )


@pytest.fixture
def filter_block_with_list():
    """Create a FilterByValueBlock instance that filters against a list of values."""
    return FilterByValueBlock(
        block_name="test_filter_list",
        filter_column="age",
        filter_value=[30, 35],
        operation=operator.eq,
        convert_dtype=int,
    )


def test_filter_block_initialization():
    """Test FilterByValueBlock initialization."""
    block = FilterByValueBlock(
        block_name="test_init",
        filter_column="age",
        filter_value=30,
        operation=operator.eq,
        convert_dtype=int,
    )
    assert block.column_name == "age"
    assert block.value == [30]  # Note: value is always stored as a list
    assert block.operation == operator.eq
    assert block.convert_dtype == int


def test_filter_block_with_invalid_operation():
    """Test FilterByValueBlock initialization with invalid operation."""
    with pytest.raises(ValueError, match="Operation must be from operator module"):
        FilterByValueBlock(
            block_name="test_invalid_op",
            filter_column="age",
            filter_value=30,
            operation=lambda x, y: x == y,  # Invalid operation
            convert_dtype=int,
        )


def test_filter_block_with_mixed_types(filter_block, sample_dataset, caplog):
    """Test filtering with mixed data types."""
    filtered_dataset = filter_block.generate(sample_dataset)
    assert len(filtered_dataset) == 1
    assert filtered_dataset["age"] == [30]
    assert "Error converting dtype" in caplog.text


def test_filter_block_with_list_values(filter_block_with_list, sample_dataset, caplog):
    """Test filtering with multiple values."""
    filtered_dataset = filter_block_with_list.generate(sample_dataset)
    assert len(filtered_dataset) == 2
    assert filtered_dataset["age"] == [30, 35]
    assert "Error converting dtype" in caplog.text


def test_filter_block_with_greater_than():
    """Test filtering with greater than operation."""
    block = FilterByValueBlock(
        block_name="test_gt",
        filter_column="age",
        filter_value=30,
        operation=operator.gt,
        convert_dtype=int,
    )
    dataset = Dataset.from_dict(
        {"age": ["25", "30", "35", "40", "45"]},
        features=Features({"age": Value("string")}),
    )
    filtered_dataset = block.generate(dataset)
    assert len(filtered_dataset) == 3
    assert filtered_dataset["age"] == [35, 40, 45]


def test_filter_block_with_less_than():
    """Test filtering with less than operation."""
    block = FilterByValueBlock(
        block_name="test_lt",
        filter_column="age",
        filter_value=35,
        operation=operator.lt,
        convert_dtype=int,
    )
    dataset = Dataset.from_dict(
        {"age": ["25", "30", "35", "40", "45"]},
        features=Features({"age": Value("string")}),
    )
    filtered_dataset = block.generate(dataset)
    assert len(filtered_dataset) == 2
    assert filtered_dataset["age"] == [25, 30]


def test_filter_block_with_invalid_column():
    """Test filtering with non-existent column."""
    block = FilterByValueBlock(
        block_name="test_invalid_col",
        filter_column="nonexistent",
        filter_value=30,
        operation=operator.eq,
        convert_dtype=int,
    )
    dataset = Dataset.from_dict(
        {"age": ["25", "30", "35"]},
        features=Features({"age": Value("string")}),
    )
    with pytest.raises(KeyError):
        block.generate(dataset)


def test_filter_block_with_empty_dataset():
    """Test filtering with an empty dataset."""
    block = FilterByValueBlock(
        block_name="test_empty",
        filter_column="age",
        filter_value=30,
        operation=operator.eq,
        convert_dtype=int,
    )
    dataset = Dataset.from_dict(
        {"age": []},
        features=Features({"age": Value("string")}),
    )
    filtered_dataset = block.generate(dataset)
    assert len(filtered_dataset) == 0
