"""Tests for the FilterByValueBlock implementation."""

# Standard
import operator

# Third Party
from datasets import Dataset, Features, Value
import pytest

# First Party
from sdg_hub.blocks import FilterByValueBlock
from sdg_hub.utils.error_handling import EmptyDatasetError, MissingColumnError


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
        input_cols="age",
        filter_value=30,
        operation=operator.eq,
        convert_dtype=int,
    )


@pytest.fixture
def filter_block_with_list():
    """Create a FilterByValueBlock instance that filters against a list of values."""
    return FilterByValueBlock(
        block_name="test_filter_list",
        input_cols="age",
        filter_value=[30, 35],
        operation=operator.eq,
        convert_dtype=int,
    )


def test_filter_block_initialization():
    """Test FilterByValueBlock initialization."""
    block = FilterByValueBlock(
        block_name="test_init",
        input_cols="age",
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
            input_cols="age",
            filter_value=30,
            operation=lambda x, y: x == y,  # Invalid operation
            convert_dtype=int,
        )


def test_filter_block_with_mixed_types(filter_block, sample_dataset, caplog):
    """Test filtering with mixed data types."""
    filtered_dataset = filter_block(sample_dataset)
    assert len(filtered_dataset) == 1
    assert filtered_dataset["age"] == [30]
    assert "Error converting dtype" in caplog.text


def test_filter_block_with_list_values(filter_block_with_list, sample_dataset, caplog):
    """Test filtering with multiple values."""
    filtered_dataset = filter_block_with_list(sample_dataset)
    assert len(filtered_dataset) == 2
    assert filtered_dataset["age"] == [30, 35]
    assert "Error converting dtype" in caplog.text


def test_filter_block_with_greater_than():
    """Test filtering with greater than operation."""
    block = FilterByValueBlock(
        block_name="test_gt",
        input_cols="age",
        filter_value=30,
        operation=operator.gt,
        convert_dtype=int,
    )
    dataset = Dataset.from_dict(
        {"age": ["25", "30", "35", "40", "45"]},
        features=Features({"age": Value("string")}),
    )
    filtered_dataset = block(dataset)
    assert len(filtered_dataset) == 3
    assert filtered_dataset["age"] == [35, 40, 45]


def test_filter_block_with_less_than():
    """Test filtering with less than operation."""
    block = FilterByValueBlock(
        block_name="test_lt",
        input_cols="age",
        filter_value=35,
        operation=operator.lt,
        convert_dtype=int,
    )
    dataset = Dataset.from_dict(
        {"age": ["25", "30", "35", "40", "45"]},
        features=Features({"age": Value("string")}),
    )
    filtered_dataset = block(dataset)
    assert len(filtered_dataset) == 2
    assert filtered_dataset["age"] == [25, 30]


def test_filter_block_with_invalid_column():
    """Test filtering with non-existent column."""
    block = FilterByValueBlock(
        block_name="test_invalid_col",
        input_cols="nonexistent",
        filter_value=30,
        operation=operator.eq,
        convert_dtype=int,
    )
    dataset = Dataset.from_dict(
        {"age": ["25", "30", "35"]},
        features=Features({"age": Value("string")}),
    )
    with pytest.raises(MissingColumnError):
        block(dataset)  # Use __call__ method to trigger BaseBlock validation


def test_filter_block_with_empty_dataset():
    """Test filtering with an empty dataset."""
    block = FilterByValueBlock(
        block_name="test_empty",
        input_cols="age",
        filter_value=30,
        operation=operator.eq,
        convert_dtype=int,
    )
    dataset = Dataset.from_dict(
        {"age": []},
        features=Features({"age": Value("string")}),
    )
    # BaseBlock should raise EmptyDatasetError for empty datasets
    with pytest.raises(EmptyDatasetError):
        block(dataset)


def test_filter_block_with_multiple_input_cols():
    """Test FilterByValueBlock with multiple input columns specified."""
    block = FilterByValueBlock(
        block_name="test_multi_input",
        input_cols=["age", "metadata"],  # Filter column is first
        filter_value=30,
        operation=operator.eq,
        convert_dtype=int,
    )
    assert block.input_cols == ["age", "metadata"]
    assert block.output_cols == []
    assert block.column_name == "age"  # First column is filter column
    assert block.value == [30]


def test_filter_block_single_input_col():
    """Test that input_cols works with single column specified."""
    block = FilterByValueBlock(
        block_name="test_single",
        input_cols="score",
        filter_value=2.0,
        operation=operator.ge,
        convert_dtype=float,
    )
    assert block.input_cols == ["score"]
    assert block.output_cols == []


def test_filter_block_empty_input_cols():
    """Test FilterByValueBlock raises error with empty input columns."""
    with pytest.raises(ValueError, match="requires at least one input column"):
        FilterByValueBlock(
            block_name="test_empty_input",
            input_cols=[],
            filter_value=2.0,
            operation=operator.ge,
            convert_dtype=float,
        )


def test_filter_block_with_contains():
    """Test filtering with operator.contains."""
    block = FilterByValueBlock(
        block_name="test_contains",
        input_cols="text",
        filter_value="world",
        operation=operator.contains,
    )
    dataset = Dataset.from_dict(
        {"text": ["hello world", "goodbye moon", "hello there", "world peace"]},
        features=Features({"text": Value("string")}),
    )
    filtered_dataset = block(dataset)
    assert len(filtered_dataset) == 2
    assert filtered_dataset["text"] == ["hello world", "world peace"]


def test_filter_block_with_contains_multiple_values():
    """Test filtering with operator.contains and multiple filter values."""
    block = FilterByValueBlock(
        block_name="test_contains_multi",
        input_cols="text",
        filter_value=["world", "moon"],
        operation=operator.contains,
    )
    dataset = Dataset.from_dict(
        {"text": ["hello world", "goodbye moon", "hello there", "world peace", "moon landing"]},
        features=Features({"text": Value("string")}),
    )
    filtered_dataset = block(dataset)
    assert len(filtered_dataset) == 4
    assert filtered_dataset["text"] == ["hello world", "goodbye moon", "world peace", "moon landing"]
