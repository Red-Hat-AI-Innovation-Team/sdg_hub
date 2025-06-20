"""Tests for the Flow class functionality."""

# Standard
from unittest.mock import MagicMock, patch
import pytest

# Third Party
from datasets import Dataset
from datasets.data_files import EmptyDatasetError

# First Party
from sdg_hub.flow import Flow

from io import StringIO
from rich.console import Console


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    return MagicMock()


@pytest.fixture
def flow(mock_llm_client):
    """Create a Flow instance with mock LLM client."""
    return Flow(mock_llm_client)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {"text": ["sample text 1", "sample text 2"], "label": [0, 1]}
    )


def test_flow_initialization(mock_llm_client):
    """Test Flow initialization with different parameters."""
    # Test basic initialization
    flow = Flow(mock_llm_client)
    assert flow.llm_client == mock_llm_client
    assert flow.chained_blocks is None
    assert flow.num_samples_to_generate is None

    # Test initialization with num_samples
    flow = Flow(mock_llm_client, num_samples_to_generate=100)
    assert flow.num_samples_to_generate == 100


def test_generate_without_blocks(flow, sample_dataset):
    """Test generate method raises error when no blocks are initialized."""
    with pytest.raises(ValueError, match="Flow has not been initialized with blocks"):
        flow.generate(sample_dataset)


def test_drop_duplicates(flow):
    """Test _drop_duplicates method functionality."""
    dataset = Dataset.from_dict(
        {"text": ["duplicate", "duplicate", "unique"], "label": [0, 0, 1]}
    )
    
    # Test dropping duplicates on single column
    result = flow._drop_duplicates(dataset, ["text"])
    assert len(result) == 2
    
    # Test dropping duplicates on multiple columns
    result = flow._drop_duplicates(dataset, ["text", "label"])
    assert len(result) == 2


def test_generate_with_empty_dataset(flow, sample_dataset):
    """Test generate method handles empty dataset correctly."""
    # Mock a block that returns empty dataset
    mock_block = MagicMock()
    mock_block.generate.return_value = Dataset.from_dict({})
    
    flow.chained_blocks = [
        {
            "block_type": mock_block,
            "block_config": {"block_name": "test_block"},
            "drop_columns": [],
            "gen_kwargs": {},
            "drop_duplicates": False,
        }
    ]

    with pytest.raises(EmptyDatasetError, match="Empty dataset after running block"):
        flow.generate(sample_dataset)


def test_generate_with_drop_columns(flow, sample_dataset):
    """Test generate method handles column dropping correctly."""
    # Create a mock block class that returns our mock instance
    mock_block_class = MagicMock()
    mock_block_instance = MagicMock()
    mock_block_instance.generate.return_value = Dataset.from_dict(
        {"text": ["new text"], "label": [1], "extra": ["extra"]}
    )
    mock_block_class.return_value = mock_block_instance
    
    flow.chained_blocks = [
        {
            "block_type": mock_block_class,
            "block_config": {"block_name": "test_block"},
            "drop_columns": ["extra"],
            "gen_kwargs": {},
            "drop_duplicates": False,
        }
    ]

    result = flow.generate(sample_dataset)
    assert "extra" not in result.column_names
    assert "text" in result.column_names
    assert "label" in result.column_names


def test_generate_with_multiple_blocks(flow, sample_dataset):
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
    
    flow.chained_blocks = [
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

    result = flow.generate(sample_dataset)
    
    # Verify both blocks were called
    mock_block1_instance.generate.assert_called_once()
    mock_block2_instance.generate.assert_called_once()
    
    # Verify final dataset structure
    assert "text" in result.column_names
    assert "label" in result.column_names
    assert "new_col" in result.column_names


def test_get_flow_from_file_with_llm_block(flow):
    """Test get_flow_from_file with LLM block configuration."""
    mock_yaml_content = [
        {
            "block_type": "LLMBlock",
            "block_config": {
                "block_name": "test_llm",
                "model_id": "test_model",
                "client": None,  # Will be set by Flow
            },
        }
    ]

    with (
        patch("yaml.safe_load", return_value=mock_yaml_content),
        patch("builtins.open", MagicMock()),
        patch("os.path.isfile", return_value=True),
        patch("os.path.dirname", return_value="test_dir"),
        patch(
            "sdg_hub.flow.BlockRegistry.get_registry",
            return_value={"LLMBlock": MagicMock},
        ),
        patch(
            "sdg_hub.flow.PromptRegistry.get_registry",
            return_value={"test_model": "test_prompt"},
        ),
    ):
        result = flow.get_flow_from_file("test.yaml")

        # Verify LLM client was set
        assert result.chained_blocks[0]["block_config"]["client"] == flow.llm_client
        # Verify model_id was set correctly
        assert result.chained_blocks[0]["block_config"]["model_id"] == "test_model"


def test_get_flow_from_file_with_num_samples(mock_llm_client):
    """Test get_flow_from_file with num_samples configuration."""
    flow = Flow(mock_llm_client, num_samples_to_generate=100)
    mock_yaml_content = [
        {
            "block_type": "LLMBlock",
            "block_config": {"block_name": "test_llm", "model_id": "test_model"},
        }
    ]

    with (
        patch("yaml.safe_load", return_value=mock_yaml_content),
        patch("builtins.open", MagicMock()),
        patch("os.path.isfile", return_value=True),
        patch("os.path.dirname", return_value="test_dir"),
        patch(
            "sdg_hub.flow.BlockRegistry.get_registry",
            return_value={"LLMBlock": MagicMock},
        ),
        patch(
            "sdg_hub.flow.PromptRegistry.get_registry",
            return_value={"test_model": "test_prompt"},
        ),
    ):
        result = flow.get_flow_from_file("test.yaml")
        assert result.chained_blocks[0]["num_samples"] == 100

def test_generate_verbose_logs_shows_rich_table(flow, sample_dataset):
    """Test that verbose log level produces rich table output."""
    flow.log_level = "verbose"

    # Override the console with a custom StringIO to capture output
    console_output = StringIO()
    flow.console = Console(file=console_output, force_terminal=False, width=100)

    mock_block = MagicMock()
    mock_block.generate.return_value = sample_dataset
    flow.chained_blocks = [
        {
            "block_type": lambda **kwargs: mock_block,
            "block_config": {"block_name": "test_block"},
            "drop_columns": [],
            "gen_kwargs": {},
            "drop_duplicates": False,
        }
    ]

    flow.generate(sample_dataset)

    out = console_output.getvalue()
    assert "Rows" in out
    assert "Columns" in out
    assert "test_block" in out

def test_log_level_from_env(monkeypatch, mock_llm_client):
    """Test log level is set from environment variable."""
    monkeypatch.setenv("SDG_HUB_LOG_LEVEL", "debug")
    flow = Flow(mock_llm_client)
    assert flow.log_level == "debug"