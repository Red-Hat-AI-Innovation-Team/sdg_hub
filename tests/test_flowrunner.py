"""Tests for the flow_runner module."""

# Standard
from unittest.mock import MagicMock, patch
import json

# Third Party
from click.testing import CliRunner
from datasets import Dataset
import pytest

# First Party
from sdg_hub.flow_runner import main, run_flow


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    # Create a larger dataset for debug mode testing
    return Dataset.from_dict(
        {
            "text": ["sample text " + str(i) for i in range(50)],
            "metadata": [{"id": i} for i in range(50)]
        }
    )


@pytest.fixture
def mock_flow_config(tmp_path):
    """Create a mock flow configuration file."""
    flow_config = {"name": "test_flow", "steps": [{"type": "test_step", "config": {}}]}
    flow_path = tmp_path / "test_flow.json"
    flow_path.write_text(json.dumps(flow_config))
    return str(flow_path)


@pytest.fixture
def mock_output_path(tmp_path):
    """Create a mock output path."""
    return str(tmp_path / "output.json")


@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    """Create a mock checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return str(checkpoint_dir)


@pytest.fixture
def mock_input_dataset(tmp_path):
    """Create a mock input dataset file."""
    dataset = {
        "text": ["sample text " + str(i) for i in range(50)],
        "metadata": [{"id": i} for i in range(50)],
    }
    dataset_path = tmp_path / "input.json"
    dataset_path.write_text(json.dumps(dataset))
    return str(dataset_path)


@patch("sdg_hub.flow_runner.load_dataset")
@patch("sdg_hub.flow_runner.OpenAI")
@patch("sdg_hub.flow_runner.Flow")
@patch("sdg_hub.flow_runner.SDG")
def test_run_flow_success(
    mock_sdg,
    mock_flow,
    mock_openai,
    mock_load_dataset,
    mock_dataset,
    mock_flow_config,
    mock_output_path,
    mock_checkpoint_dir,
    mock_input_dataset,
):
    """Test successful execution of run_flow."""
    # Setup mocks
    mock_load_dataset.return_value = mock_dataset
    mock_openai_instance = MagicMock()
    mock_openai.return_value = mock_openai_instance
    mock_flow_instance = MagicMock()
    mock_flow.return_value = mock_flow_instance
    mock_flow_instance.get_flow_from_file.return_value = {"test": "config"}
    mock_sdg_instance = MagicMock()
    mock_sdg.return_value = mock_sdg_instance
    mock_sdg_instance.generate.return_value = mock_dataset

    # Run the function
    run_flow(
        ds_path=mock_input_dataset,
        batch_size=8,
        num_workers=4,
        save_path=mock_output_path,
        endpoint="http://test.endpoint",
        flow_path=mock_flow_config,
        checkpoint_dir=mock_checkpoint_dir,
        save_freq=2,
        debug=False,
        dataset_start_index=0,
        dataset_end_index=None,
    )

    # Verify mocks were called correctly
    mock_load_dataset.assert_called_once_with(
        "json", data_files=mock_input_dataset, split="train"
    )
    mock_openai.assert_called_once()
    mock_flow_instance.get_flow_from_file.assert_called_once_with(mock_flow_config)
    mock_sdg.assert_called_once()
    mock_sdg_instance.generate.assert_called_once_with(
        mock_dataset, checkpoint_dir=mock_checkpoint_dir
    )


@patch("sdg_hub.flow_runner.load_dataset")
@patch("sdg_hub.flow_runner.OpenAI")
@patch("sdg_hub.flow_runner.Flow")
@patch("sdg_hub.flow_runner.SDG")
def test_run_flow_debug_mode(
    mock_sdg,
    mock_flow,
    mock_openai,
    mock_load_dataset,
    mock_dataset,
    mock_flow_config,
    mock_output_path,
    mock_checkpoint_dir,
    mock_input_dataset,
):
    """Test run_flow in debug mode."""
    # Setup mocks
    mock_load_dataset.return_value = mock_dataset
    mock_openai_instance = MagicMock()
    mock_openai.return_value = mock_openai_instance
    mock_flow_instance = MagicMock()
    mock_flow.return_value = mock_flow_instance
    mock_flow_instance.get_flow_from_file.return_value = {"test": "config"}
    mock_sdg_instance = MagicMock()
    mock_sdg.return_value = mock_sdg_instance
    mock_sdg_instance.generate.return_value = mock_dataset

    # Mock the shuffle and select methods
    mock_dataset.shuffle = MagicMock(return_value=mock_dataset)
    mock_dataset.select = MagicMock(return_value=mock_dataset)

    # Run the function in debug mode
    run_flow(
        ds_path=mock_input_dataset,
        batch_size=8,
        num_workers=4,
        save_path=mock_output_path,
        endpoint="http://test.endpoint",
        flow_path=mock_flow_config,
        checkpoint_dir=mock_checkpoint_dir,
        save_freq=2,
        debug=True,
        dataset_start_index=0,
        dataset_end_index=None,
    )

    # Verify debug mode behavior
    mock_dataset.shuffle.assert_called_once_with(seed=42)
    mock_dataset.select.assert_called_once_with(range(30))


def test_run_flow_missing_flow_file(
    mock_output_path,
    mock_checkpoint_dir,
    mock_input_dataset,
):
    """Test run_flow with non-existent flow file."""
    with pytest.raises(FileNotFoundError):
        run_flow(
            ds_path=mock_input_dataset,
            batch_size=8,
            num_workers=4,
            save_path=mock_output_path,
            endpoint="http://test.endpoint",
            flow_path="nonexistent.json",
            checkpoint_dir=mock_checkpoint_dir,
            save_freq=2,
            dataset_start_index=0,
            dataset_end_index=None,
        )


@patch("sdg_hub.flow_runner.run_flow")
def test_cli_main_success(
    mock_run_flow,
    mock_flow_config,
    mock_output_path,
    mock_checkpoint_dir,
    mock_input_dataset,
):
    """Test CLI main function success case."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--ds_path",
            mock_input_dataset,
            "--bs",
            "8",
            "--num_workers",
            "4",
            "--save_path",
            mock_output_path,
            "--endpoint",
            "http://test.endpoint",
            "--flow",
            mock_flow_config,
            "--checkpoint_dir",
            mock_checkpoint_dir,
            "--save_freq",
            "2",
        ],
    )
    assert result.exit_code == 0
    mock_run_flow.assert_called_once()


def test_cli_main_missing_required_args():
    """Test CLI main function with missing required arguments."""
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_cli_main_invalid_dataset_path():
    """Test CLI main function with invalid dataset path."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--ds_path",
            "nonexistent.json",
            "--bs",
            "8",
            "--num_workers",
            "4",
            "--save_path",
            "output.json",
            "--endpoint",
            "http://test.endpoint",
            "--flow",
            "flow.json",
            "--checkpoint_dir",
            "checkpoints",
            "--save_freq",
            "2",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value" in result.output


@patch("sdg_hub.flow_runner.load_dataset")
@patch("sdg_hub.flow_runner.OpenAI")
@patch("sdg_hub.flow_runner.Flow")
@patch("sdg_hub.flow_runner.SDG")
def test_run_flow_with_dataset_indices(
    mock_sdg,
    mock_flow,
    mock_openai,
    mock_load_dataset,
    mock_dataset,
    mock_flow_config,
    mock_output_path,
    mock_checkpoint_dir,
    mock_input_dataset,
):
    """Test run_flow with dataset start and end indices."""
    # Setup mocks
    mock_load_dataset.return_value = mock_dataset
    mock_openai_instance = MagicMock()
    mock_openai.return_value = mock_openai_instance
    mock_flow_instance = MagicMock()
    mock_flow.return_value = mock_flow_instance
    mock_flow_instance.get_flow_from_file.return_value = {"test": "config"}
    mock_sdg_instance = MagicMock()
    mock_sdg.return_value = mock_sdg_instance
    mock_sdg_instance.generate.return_value = mock_dataset

    # Mock the select method
    mock_dataset.select = MagicMock(return_value=mock_dataset)

    # Run the function with dataset indices
    run_flow(
        ds_path=mock_input_dataset,
        batch_size=8,
        num_workers=4,
        save_path=mock_output_path,
        endpoint="http://test.endpoint",
        flow_path=mock_flow_config,
        checkpoint_dir=mock_checkpoint_dir,
        save_freq=2,
        debug=False,
        dataset_start_index=10,
        dataset_end_index=20,
    )

    # Verify dataset slicing was called
    mock_dataset.select.assert_called_once_with(range(10, 20))


@patch("sdg_hub.flow_runner.load_dataset")
@patch("sdg_hub.flow_runner.OpenAI")
@patch("sdg_hub.flow_runner.Flow")
@patch("sdg_hub.flow_runner.SDG")
def test_run_flow_with_modified_save_path(
    mock_sdg,
    mock_flow,
    mock_openai,
    mock_load_dataset,
    mock_dataset,
    mock_flow_config,
    mock_checkpoint_dir,
    mock_input_dataset,
    tmp_path,
):
    """Test run_flow modifies save path when dataset indices are provided."""
    # Setup mocks
    mock_load_dataset.return_value = mock_dataset
    mock_openai_instance = MagicMock()
    mock_openai.return_value = mock_openai_instance
    mock_flow_instance = MagicMock()
    mock_flow.return_value = mock_flow_instance
    mock_flow_instance.get_flow_from_file.return_value = {"test": "config"}
    mock_sdg_instance = MagicMock()
    mock_sdg.return_value = mock_sdg_instance
    mock_sdg_instance.generate.return_value = mock_dataset

    # Mock the select method
    mock_dataset.select = MagicMock(return_value=mock_dataset)
    # Mock to_json to capture the save path
    mock_dataset.to_json = MagicMock()

    save_path = str(tmp_path / "output.jsonl")

    # Run the function with dataset indices
    run_flow(
        ds_path=mock_input_dataset,
        batch_size=8,
        num_workers=4,
        save_path=save_path,
        endpoint="http://test.endpoint",
        flow_path=mock_flow_config,
        checkpoint_dir=mock_checkpoint_dir,
        save_freq=2,
        debug=False,
        dataset_start_index=5,
        dataset_end_index=15,
    )

    # Verify the save path was modified
    expected_save_path = str(tmp_path / "output_5_15.jsonl")
    mock_dataset.to_json.assert_called_once_with(
        expected_save_path, orient="records", lines=True
    )


@patch("sdg_hub.flow_runner.load_dataset")
@patch("sdg_hub.flow_runner.OpenAI")
@patch("sdg_hub.flow_runner.Flow")
@patch("sdg_hub.flow_runner.SDG")
def test_run_flow_without_dataset_indices(
    mock_sdg,
    mock_flow,
    mock_openai,
    mock_load_dataset,
    mock_dataset,
    mock_flow_config,
    mock_output_path,
    mock_checkpoint_dir,
    mock_input_dataset,
):
    """Test run_flow without dataset indices doesn't slice dataset."""
    # Setup mocks
    mock_load_dataset.return_value = mock_dataset
    mock_openai_instance = MagicMock()
    mock_openai.return_value = mock_openai_instance
    mock_flow_instance = MagicMock()
    mock_flow.return_value = mock_flow_instance
    mock_flow_instance.get_flow_from_file.return_value = {"test": "config"}
    mock_sdg_instance = MagicMock()
    mock_sdg.return_value = mock_sdg_instance
    mock_sdg_instance.generate.return_value = mock_dataset

    # Mock the select method
    mock_dataset.select = MagicMock(return_value=mock_dataset)

    # Run the function without dataset indices (using defaults)
    run_flow(
        ds_path=mock_input_dataset,
        batch_size=8,
        num_workers=4,
        save_path=mock_output_path,
        endpoint="http://test.endpoint",
        flow_path=mock_flow_config,
        checkpoint_dir=mock_checkpoint_dir,
        save_freq=2,
        debug=False,
        dataset_start_index=0,
        dataset_end_index=None,
    )

    # Verify dataset slicing was NOT called
    mock_dataset.select.assert_not_called()


@patch("sdg_hub.flow_runner.run_flow")
def test_cli_main_with_dataset_indices(
    mock_run_flow,
    mock_flow_config,
    mock_output_path,
    mock_checkpoint_dir,
    mock_input_dataset,
):
    """Test CLI main function with dataset indices."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--ds_path",
            mock_input_dataset,
            "--bs",
            "8",
            "--num_workers",
            "4",
            "--save_path",
            mock_output_path,
            "--endpoint",
            "http://test.endpoint",
            "--flow",
            mock_flow_config,
            "--checkpoint_dir",
            mock_checkpoint_dir,
            "--save_freq",
            "2",
            "--dataset_start_index",
            "10",
            "--dataset_end_index",
            "50",
        ],
    )
    assert result.exit_code == 0
    
    # Verify run_flow was called with the correct parameters
    mock_run_flow.assert_called_once()
    call_args = mock_run_flow.call_args
    assert call_args.kwargs["dataset_start_index"] == 10
    assert call_args.kwargs["dataset_end_index"] == 50
