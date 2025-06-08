"""Tests for the deprecated Pipeline class."""

# Standard
from unittest.mock import MagicMock

# Third Party
from datasets import Dataset
from datasets.data_files import EmptyDatasetError
import pytest

# First Party
from sdg_hub.pipeline import Pipeline


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {"text": ["sample text 1", "sample text 2"], "label": [0, 1]}
    )


def test_pipeline_deprecation_warning():
    """Test that Pipeline initialization raises deprecation warning."""
    mock_block = MagicMock()
    chained_blocks = [
        {
            "block_type": mock_block,
            "block_config": {"block_name": "test_block"},
            "drop_columns": [],
            "gen_kwargs": {},
            "drop_duplicates": False,
        }
    ]

    with pytest.warns(DeprecationWarning, match="Pipeline class is deprecated"):
        Pipeline(chained_blocks)


def test_drop_duplicates():
    """Test _drop_duplicates method functionality."""
    pipeline = Pipeline([])  # Empty blocks list is fine for this test
    dataset = Dataset.from_dict(
        {"text": ["duplicate", "duplicate", "unique"], "label": [0, 0, 1]}
    )

    # Test dropping duplicates on single column
    result = pipeline._drop_duplicates(dataset, ["text"])
    assert len(result) == 2

    # Test dropping duplicates on multiple columns
    result = pipeline._drop_duplicates(dataset, ["text", "label"])
    assert len(result) == 2


def test_generate_with_empty_dataset(sample_dataset):
    """Test generate method handles empty dataset correctly."""
    # Create a mock block class that returns empty dataset
    mock_block_class = MagicMock()
    mock_block_instance = MagicMock()
    mock_block_instance.generate.return_value = Dataset.from_dict({})
    mock_block_class.return_value = mock_block_instance

    chained_blocks = [
        {
            "block_type": mock_block_class,
            "block_config": {"block_name": "test_block"},
            "drop_columns": [],
            "gen_kwargs": {},
            "drop_duplicates": False,
        }
    ]

    pipeline = Pipeline(chained_blocks)
    with pytest.raises(EmptyDatasetError, match="Empty dataset after running block"):
        pipeline.generate(sample_dataset)


def test_generate_with_drop_columns(sample_dataset):
    """Test generate method handles column dropping correctly."""
    # Create a mock block class that returns our mock instance
    mock_block_class = MagicMock()
    mock_block_instance = MagicMock()
    mock_block_instance.generate.return_value = Dataset.from_dict(
        {"text": ["new text"], "label": [1], "extra": ["extra"]}
    )
    mock_block_class.return_value = mock_block_instance

    chained_blocks = [
        {
            "block_type": mock_block_class,
            "block_config": {"block_name": "test_block"},
            "drop_columns": ["extra"],
            "gen_kwargs": {},
            "drop_duplicates": False,
        }
    ]

    pipeline = Pipeline(chained_blocks)
    result = pipeline.generate(sample_dataset)
    assert "extra" not in result.column_names
    assert "text" in result.column_names
    assert "label" in result.column_names


def test_generate_with_multiple_blocks(sample_dataset):
    """Test generate method with multiple blocks in sequence."""
    # Create mock block classes that return our mock instances
    mock_block1_class = MagicMock()
    mock_block1_instance = MagicMock()
    mock_block1_instance.generate.return_value = Dataset.from_dict(
        {"text": ["processed text"], "label": [1]}
    )
    mock_block1_class.return_value = mock_block1_instance

    mock_block2_class = MagicMock()
    mock_block2_instance = MagicMock()
    mock_block2_instance.generate.return_value = Dataset.from_dict(
        {"text": ["further processed"], "label": [1], "new_col": ["value"]}
    )
    mock_block2_class.return_value = mock_block2_instance

    chained_blocks = [
        {
            "block_type": mock_block1_class,
            "block_config": {"block_name": "block1"},
            "drop_columns": [],
            "gen_kwargs": {},
            "drop_duplicates": False,
        },
        {
            "block_type": mock_block2_class,
            "block_config": {"block_name": "block2"},
            "drop_columns": [],
            "gen_kwargs": {},
            "drop_duplicates": False,
        },
    ]

    pipeline = Pipeline(chained_blocks)
    result = pipeline.generate(sample_dataset)

    # Verify both blocks were called
    mock_block1_instance.generate.assert_called_once()
    mock_block2_instance.generate.assert_called_once()

    # Verify final dataset structure
    assert "text" in result.column_names
    assert "label" in result.column_names
    assert "new_col" in result.column_names


def test_generate_with_drop_duplicates(sample_dataset):
    """Test generate method handles duplicate dropping correctly."""
    # Create a mock block class that returns duplicate data
    mock_block_class = MagicMock()
    mock_block_instance = MagicMock()
    mock_block_instance.generate.return_value = Dataset.from_dict(
        {"text": ["duplicate", "duplicate", "unique"], "label": [0, 0, 1]}
    )
    mock_block_class.return_value = mock_block_instance

    chained_blocks = [
        {
            "block_type": mock_block_class,
            "block_config": {"block_name": "test_block"},
            "drop_columns": [],
            "gen_kwargs": {},
            "drop_duplicates": ["text", "label"],
        }
    ]

    pipeline = Pipeline(chained_blocks)
    result = pipeline.generate(sample_dataset)
    assert len(result) == 2  # Should have removed one duplicate


def test_generate_with_nonexistent_drop_columns(sample_dataset):
    """Test generate method handles nonexistent columns gracefully."""
    # Create a mock block class that returns our mock instance
    mock_block_class = MagicMock()
    mock_block_instance = MagicMock()
    mock_block_instance.generate.return_value = Dataset.from_dict(
        {"text": ["new text"], "label": [1]}
    )
    mock_block_class.return_value = mock_block_instance

    chained_blocks = [
        {
            "block_type": mock_block_class,
            "block_config": {"block_name": "test_block"},
            "drop_columns": ["nonexistent"],  # Column doesn't exist in dataset
            "gen_kwargs": {},
            "drop_duplicates": False,
        }
    ]

    pipeline = Pipeline(chained_blocks)
    result = pipeline.generate(sample_dataset)
    assert "text" in result.column_names
    assert "label" in result.column_names
