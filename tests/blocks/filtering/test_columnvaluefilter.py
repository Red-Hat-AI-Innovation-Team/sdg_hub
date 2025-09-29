"""Tests for the ColumnValueFilterBlock implementation."""

# Standard

# Third Party
from datasets import Dataset, Features, Value

# First Party
from sdg_hub.core.blocks import ColumnValueFilterBlock
from sdg_hub.core.utils.error_handling import EmptyDatasetError, MissingColumnError
import pytest


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {"age": ["25", "30", "35", "forty", "45"]},
        features=Features({"age": Value("string")}),
    )


@pytest.fixture
def filter_block():
    """Create a basic ColumnValueFilterBlock instance."""
    return ColumnValueFilterBlock(
        block_name="test_filter",
        input_cols="age",
        filter_value=30,
        operation="eq",
        convert_dtype="int",
    )


@pytest.fixture
def filter_block_with_list():
    """Create a ColumnValueFilterBlock instance that filters against a list of values."""
    return ColumnValueFilterBlock(
        block_name="test_filter_list",
        input_cols="age",
        filter_value=[30, 35],
        operation="eq",
        convert_dtype="int",
    )


def test_filter_block_initialization():
    """Test ColumnValueFilterBlock initialization."""
    block = ColumnValueFilterBlock(
        block_name="test_init",
        input_cols="age",
        filter_value=30,
        operation="eq",
        convert_dtype="int",
    )
    assert block.column_name == "age"
    assert block.filter_value == [30]  # Note: value is always stored as a list
    assert block.operation == "eq"
    assert block.convert_dtype == "int"


def test_filter_block_with_invalid_operation():
    """Test ColumnValueFilterBlock initialization with invalid operation."""
    with pytest.raises(ValueError, match="Unsupported operation 'invalid'"):
        ColumnValueFilterBlock(
            block_name="test_invalid_op",
            input_cols="age",
            filter_value=30,
            operation="invalid",  # Invalid operation string
            convert_dtype="int",
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
    block = ColumnValueFilterBlock(
        block_name="test_gt",
        input_cols="age",
        filter_value=30,
        operation="gt",
        convert_dtype="int",
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
    block = ColumnValueFilterBlock(
        block_name="test_lt",
        input_cols="age",
        filter_value=35,
        operation="lt",
        convert_dtype="int",
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
    block = ColumnValueFilterBlock(
        block_name="test_invalid_col",
        input_cols="nonexistent",
        filter_value=30,
        operation="eq",
        convert_dtype="int",
    )
    dataset = Dataset.from_dict(
        {"age": ["25", "30", "35"]},
        features=Features({"age": Value("string")}),
    )
    with pytest.raises(MissingColumnError):
        block(dataset)  # Use __call__ method to trigger BaseBlock validation


def test_filter_block_with_empty_dataset():
    """Test filtering with an empty dataset."""
    block = ColumnValueFilterBlock(
        block_name="test_empty",
        input_cols="age",
        filter_value=30,
        operation="eq",
        convert_dtype="int",
    )
    dataset = Dataset.from_dict(
        {"age": []},
        features=Features({"age": Value("string")}),
    )
    # BaseBlock should raise EmptyDatasetError for empty datasets
    with pytest.raises(EmptyDatasetError):
        block(dataset)


def test_filter_block_with_multiple_input_cols():
    """Test ColumnValueFilterBlock with multiple input columns specified."""
    block = ColumnValueFilterBlock(
        block_name="test_multi_input",
        input_cols=["age", "metadata"],  # Filter column is first
        filter_value=30,
        operation="eq",
        convert_dtype="int",
    )
    assert block.input_cols == ["age", "metadata"]
    assert block.output_cols == []
    assert block.column_name == "age"  # First column is filter column
    assert block.filter_value == [30]


def test_filter_block_single_input_col():
    """Test that input_cols works with single column specified."""
    block = ColumnValueFilterBlock(
        block_name="test_single",
        input_cols="score",
        filter_value=2.0,
        operation="ge",
        convert_dtype="float",
    )
    assert block.input_cols == ["score"]
    assert block.output_cols == []


def test_filter_block_empty_input_cols():
    """Test ColumnValueFilterBlock raises error with empty input columns."""
    with pytest.raises(ValueError, match="requires at least one input column"):
        ColumnValueFilterBlock(
            block_name="test_empty_input",
            input_cols=[],
            filter_value=2.0,
            operation="ge",
            convert_dtype="float",
        )


def test_filter_block_with_contains():
    """Test filtering with contains operation."""
    block = ColumnValueFilterBlock(
        block_name="test_contains",
        input_cols="text",
        filter_value="world",
        operation="contains",
    )
    dataset = Dataset.from_dict(
        {"text": ["hello world", "goodbye moon", "hello there", "world peace"]},
        features=Features({"text": Value("string")}),
    )
    filtered_dataset = block(dataset)
    assert len(filtered_dataset) == 2
    assert filtered_dataset["text"] == ["hello world", "world peace"]


def test_filter_block_with_contains_multiple_values():
    """Test filtering with contains operation and multiple filter values."""
    block = ColumnValueFilterBlock(
        block_name="test_contains_multi",
        input_cols="text",
        filter_value=["world", "moon"],
        operation="contains",
    )
    dataset = Dataset.from_dict(
        {
            "text": [
                "hello world",
                "goodbye moon",
                "hello there",
                "world peace",
                "moon landing",
            ]
        },
        features=Features({"text": Value("string")}),
    )
    filtered_dataset = block(dataset)
    assert len(filtered_dataset) == 4
    assert filtered_dataset["text"] == [
        "hello world",
        "goodbye moon",
        "world peace",
        "moon landing",
    ]


def test_filter_block_with_all_operations():
    """Test all supported operations work correctly."""
    dataset = Dataset.from_dict(
        {"score": ["10", "20", "30", "40", "50"]},
        features=Features({"score": Value("string")}),
    )

    # Test eq
    block_eq = ColumnValueFilterBlock(
        block_name="test_eq",
        input_cols="score",
        filter_value=30,
        operation="eq",
        convert_dtype="int",
    )
    result = block_eq(dataset)
    assert result["score"] == [30]

    # Test ne
    block_ne = ColumnValueFilterBlock(
        block_name="test_ne",
        input_cols="score",
        filter_value=30,
        operation="ne",
        convert_dtype="int",
    )
    result = block_ne(dataset)
    assert result["score"] == [10, 20, 40, 50]

    # Test le
    block_le = ColumnValueFilterBlock(
        block_name="test_le",
        input_cols="score",
        filter_value=30,
        operation="le",
        convert_dtype="int",
    )
    result = block_le(dataset)
    assert result["score"] == [10, 20, 30]

    # Test ge
    block_ge = ColumnValueFilterBlock(
        block_name="test_ge",
        input_cols="score",
        filter_value=30,
        operation="ge",
        convert_dtype="int",
    )
    result = block_ge(dataset)
    assert result["score"] == [30, 40, 50]


def test_filter_block_with_in_operation():
    """Test filtering with 'in' operation."""
    block = ColumnValueFilterBlock(
        block_name="test_in",
        input_cols="category",
        filter_value=["science", "history"],
        operation="in",
    )
    dataset = Dataset.from_dict(
        {"category": ["science", "math", "history", "art", "science"]},
        features=Features({"category": Value("string")}),
    )
    filtered_dataset = block(dataset)
    assert len(filtered_dataset) == 3
    assert filtered_dataset["category"] == ["science", "history", "science"]


def test_filter_block_without_conversion():
    """Test filtering without data type conversion."""
    block = ColumnValueFilterBlock(
        block_name="test_no_convert",
        input_cols="name",
        filter_value="Alice",
        operation="eq",
    )
    dataset = Dataset.from_dict(
        {"name": ["Alice", "Bob", "Charlie", "Alice"]},
        features=Features({"name": Value("string")}),
    )
    filtered_dataset = block(dataset)
    assert len(filtered_dataset) == 2
    assert filtered_dataset["name"] == ["Alice", "Alice"]


def test_filter_block_with_invalid_dtype():
    """Test ColumnValueFilterBlock initialization with invalid dtype."""
    with pytest.raises(ValueError, match="Unsupported dtype 'invalid'"):
        ColumnValueFilterBlock(
            block_name="test_invalid_dtype",
            input_cols="score",
            filter_value=30,
            operation="eq",
            convert_dtype="invalid",
        )


def test_filter_block_float_conversion():
    """Test filtering with float conversion."""
    block = ColumnValueFilterBlock(
        block_name="test_float",
        input_cols="price",
        filter_value=19.99,
        operation="gt",
        convert_dtype="float",
    )
    dataset = Dataset.from_dict(
        {"price": ["9.99", "19.99", "29.99", "invalid", "39.99"]},
        features=Features({"price": Value("string")}),
    )
    filtered_dataset = block(dataset)
    assert len(filtered_dataset) == 2
    assert filtered_dataset["price"] == [29.99, 39.99]
