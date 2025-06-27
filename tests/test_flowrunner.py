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
from sdg_hub.utils.error_handling import (
    APIConnectionError,
    DatasetLoadError,
    FlowConfigurationError,
)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    # Create a larger dataset for debug mode testing
    return Dataset.from_dict(
        {
            "text": ["sample text " + str(i) for i in range(50)],
            "metadata": [{"id": i} for i in range(50)],
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
    mock_openai_instance.models.list.return_value = MagicMock(data=[])
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
    mock_openai_instance.models.list.return_value = MagicMock(data=[])
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


@patch("sdg_hub.flow_runner.load_dataset")
@patch("sdg_hub.flow_runner.OpenAI")
def test_run_flow_missing_flow_file(
    mock_openai,
    mock_load_dataset,
    mock_output_path,
    mock_checkpoint_dir,
    mock_input_dataset,
    mock_dataset,
):
    """Test run_flow with non-existent flow file."""
    # Setup mocks
    mock_load_dataset.return_value = mock_dataset
    mock_openai_instance = MagicMock()
    mock_openai_instance.models.list.return_value = MagicMock(data=[])
    mock_openai.return_value = mock_openai_instance
    
    with pytest.raises(FlowConfigurationError) as exc_info:
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
    
    assert "Flow configuration file not found" in str(exc_info.value)


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
    mock_openai_instance.models.list.return_value = MagicMock(data=[])
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
    mock_openai_instance.models.list.return_value = MagicMock(data=[])
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
    mock_openai_instance.models.list.return_value = MagicMock(data=[])
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


class TestFlowRunnerErrorHandling:
    """Test error handling scenarios in flow_runner."""

    def test_dataset_load_error_invalid_file(
        self, mock_flow_config, mock_output_path, mock_checkpoint_dir
    ):
        """Test DatasetLoadError when dataset file is invalid."""
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.side_effect = Exception("Invalid JSON format")

            with pytest.raises(DatasetLoadError) as exc_info:
                run_flow(
                    ds_path="invalid.json",
                    save_path=mock_output_path,
                    endpoint="http://test.endpoint",
                    flow_path=mock_flow_config,
                    checkpoint_dir=mock_checkpoint_dir,
                )

            assert "Failed to load dataset from 'invalid.json'" in str(exc_info.value)
            assert "Invalid JSON format" in exc_info.value.details

    def test_api_connection_error_empty_endpoint(
        self,
        mock_dataset,
        mock_flow_config,
        mock_output_path,
        mock_checkpoint_dir,
        mock_input_dataset,
    ):
        """Test APIConnectionError with empty endpoint."""
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset

            with pytest.raises(APIConnectionError) as exc_info:
                run_flow(
                    ds_path=mock_input_dataset,
                    save_path=mock_output_path,
                    endpoint="",  # Empty endpoint
                    flow_path=mock_flow_config,
                    checkpoint_dir=mock_checkpoint_dir,
                )

            assert "API endpoint cannot be empty" in str(exc_info.value)

    def test_flow_configuration_error_missing_file(
        self, mock_dataset, mock_output_path, mock_checkpoint_dir, mock_input_dataset
    ):
        """Test FlowConfigurationError with missing flow file."""
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset

            with patch("sdg_hub.flow_runner.OpenAI") as mock_openai:
                mock_openai_instance = MagicMock()
                mock_openai_instance.models.list.return_value = MagicMock(data=[])
                mock_openai.return_value = mock_openai_instance
                with pytest.raises(FlowConfigurationError) as exc_info:
                    run_flow(
                        ds_path=mock_input_dataset,
                        save_path=mock_output_path,
                        endpoint="http://test.endpoint",
                        flow_path="nonexistent.yaml",
                        checkpoint_dir=mock_checkpoint_dir,
                    )

                assert "Flow configuration file not found" in str(exc_info.value)

    def test_dataset_load_error_bounds_checking(
        self, mock_dataset, mock_flow_config, mock_output_path, mock_checkpoint_dir, mock_input_dataset
    ):
        """Test DatasetLoadError with out of bounds indices."""
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset  # 50 rows
            
            with pytest.raises(DatasetLoadError) as exc_info:
                run_flow(
                    ds_path=mock_input_dataset,
                    save_path=mock_output_path,
                    endpoint="http://test.endpoint",
                    flow_path=mock_flow_config,
                    checkpoint_dir=mock_checkpoint_dir,
                    dataset_start_index=60,  # Out of bounds
                    dataset_end_index=70,
                )
            
            assert "out of bounds for dataset with 50 rows" in str(exc_info.value)

    def test_dataset_load_error_invalid_indices(
        self, mock_dataset, mock_flow_config, mock_output_path, mock_checkpoint_dir, mock_input_dataset
    ):
        """Test DatasetLoadError when start >= end index."""
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset
            
            with pytest.raises(DatasetLoadError) as exc_info:
                run_flow(
                    ds_path=mock_input_dataset,
                    save_path=mock_output_path,
                    endpoint="http://test.endpoint",
                    flow_path=mock_flow_config,
                    checkpoint_dir=mock_checkpoint_dir,
                    dataset_start_index=30,
                    dataset_end_index=20,  # Less than start
                )
            
            assert "Start index (30) must be less than end index (20)" in str(exc_info.value)

    def test_dataset_slicing_exception_handling(
        self, mock_flow_config, mock_output_path, mock_checkpoint_dir, mock_input_dataset
    ):
        """Test DatasetLoadError when dataset slicing fails."""
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_ds = MagicMock()
            mock_ds.__len__ = MagicMock(return_value=50)
            mock_ds.select.side_effect = Exception("Dataset selection failed")
            mock_load_dataset.return_value = mock_ds
            
            with pytest.raises(DatasetLoadError) as exc_info:
                run_flow(
                    ds_path=mock_input_dataset,
                    save_path=mock_output_path,
                    endpoint="http://test.endpoint",
                    flow_path=mock_flow_config,
                    checkpoint_dir=mock_checkpoint_dir,
                    dataset_start_index=5,
                    dataset_end_index=15,
                )
            
            assert "Failed to process dataset slicing or debug mode" in str(exc_info.value)
            assert "Dataset selection failed" in exc_info.value.details

    def test_api_connection_error_openai_client_failure(
        self, mock_dataset, mock_flow_config, mock_output_path, mock_checkpoint_dir, mock_input_dataset
    ):
        """Test APIConnectionError when OpenAI client initialization fails."""
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset
            
            with patch("sdg_hub.flow_runner.OpenAI") as mock_openai:
                mock_openai.side_effect = Exception("Connection timeout")
                
                with pytest.raises(APIConnectionError) as exc_info:
                    run_flow(
                        ds_path=mock_input_dataset,
                        save_path=mock_output_path,
                        endpoint="http://test.endpoint",
                        flow_path=mock_flow_config,
                        checkpoint_dir=mock_checkpoint_dir,
                    )
                
                assert "Failed to initialize OpenAI client" in str(exc_info.value)
                assert "Connection timeout" in exc_info.value.details

    def test_flow_configuration_error_invalid_yaml(
        self, mock_dataset, mock_output_path, mock_checkpoint_dir, mock_input_dataset, tmp_path
    ):
        """Test FlowConfigurationError with invalid YAML."""
        # Create invalid YAML file
        invalid_yaml_path = tmp_path / "invalid.yaml"
        invalid_yaml_path.write_text("invalid: yaml: content: [")
        
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset
            
            with patch("sdg_hub.flow_runner.OpenAI") as mock_openai:
                mock_openai_instance = MagicMock()
                mock_openai_instance.models.list.return_value = MagicMock(data=[])
                mock_openai.return_value = mock_openai_instance
                with pytest.raises(FlowConfigurationError) as exc_info:
                    run_flow(
                        ds_path=mock_input_dataset,
                        save_path=mock_output_path,
                        endpoint="http://test.endpoint",
                        flow_path=str(invalid_yaml_path),
                        checkpoint_dir=mock_checkpoint_dir,
                    )
                
                assert "contains invalid YAML" in str(exc_info.value)

    def test_flow_configuration_error_empty_file(
        self, mock_dataset, mock_output_path, mock_checkpoint_dir, mock_input_dataset, tmp_path
    ):
        """Test FlowConfigurationError with empty flow file."""
        empty_yaml_path = tmp_path / "empty.yaml"
        empty_yaml_path.write_text("")
        
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset
            
            with patch("sdg_hub.flow_runner.OpenAI") as mock_openai:
                mock_openai_instance = MagicMock()
                mock_openai_instance.models.list.return_value = MagicMock(data=[])
                mock_openai.return_value = mock_openai_instance
                with pytest.raises(FlowConfigurationError) as exc_info:
                    run_flow(
                        ds_path=mock_input_dataset,
                        save_path=mock_output_path,
                        endpoint="http://test.endpoint",
                        flow_path=str(empty_yaml_path),
                        checkpoint_dir=mock_checkpoint_dir,
                    )
                
                assert "Flow configuration file is empty" in str(exc_info.value)

    def test_flow_configuration_error_flow_creation_failure(
        self, mock_dataset, mock_flow_config, mock_output_path, mock_checkpoint_dir, mock_input_dataset
    ):
        """Test FlowConfigurationError when Flow creation fails."""
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset
            
            with patch("sdg_hub.flow_runner.OpenAI") as mock_openai:
                mock_openai_instance = MagicMock()
                mock_openai_instance.models.list.return_value = MagicMock(data=[])
                mock_openai.return_value = mock_openai_instance
                with patch("sdg_hub.flow_runner.Flow") as mock_flow:
                    mock_flow.side_effect = Exception("Invalid block configuration")
                    
                    with pytest.raises(FlowConfigurationError) as exc_info:
                        run_flow(
                            ds_path=mock_input_dataset,
                            save_path=mock_output_path,
                            endpoint="http://test.endpoint",
                            flow_path=mock_flow_config,
                            checkpoint_dir=mock_checkpoint_dir,
                        )
                    
                    assert "Failed to create flow from configuration file" in str(exc_info.value)
                    assert "Invalid block configuration" in exc_info.value.details

    def test_data_generation_error_no_data_generated(
        self, mock_dataset, mock_flow_config, mock_output_path, mock_checkpoint_dir, mock_input_dataset
    ):
        """Test DataGenerationError when no data is generated."""
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset
            
            with patch("sdg_hub.flow_runner.OpenAI") as mock_openai:
                mock_openai_instance = MagicMock()
                mock_openai_instance.models.list.return_value = MagicMock(data=[])
                mock_openai.return_value = mock_openai_instance
                with patch("sdg_hub.flow_runner.Flow") as mock_flow:
                    mock_flow_instance = MagicMock()
                    mock_flow.return_value = mock_flow_instance
                    mock_flow_instance.get_flow_from_file.return_value = {"test": "config"}
                    
                    with patch("sdg_hub.flow_runner.SDG") as mock_sdg:
                        mock_sdg_instance = MagicMock()
                        mock_sdg.return_value = mock_sdg_instance
                        mock_sdg_instance.generate.return_value = None  # No data generated
                        
                        from sdg_hub.utils.error_handling import DataGenerationError
                        with pytest.raises(DataGenerationError) as exc_info:
                            run_flow(
                                ds_path=mock_input_dataset,
                                save_path=mock_output_path,
                                endpoint="http://test.endpoint",
                                flow_path=mock_flow_config,
                                checkpoint_dir=mock_checkpoint_dir,
                            )
                        
                        assert "no data was generated" in str(exc_info.value)

    def test_data_generation_error_sdg_failure(
        self, mock_dataset, mock_flow_config, mock_output_path, mock_checkpoint_dir, mock_input_dataset
    ):
        """Test DataGenerationError when SDG.generate fails."""
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset
            
            with patch("sdg_hub.flow_runner.OpenAI") as mock_openai:
                mock_openai_instance = MagicMock()
                mock_openai_instance.models.list.return_value = MagicMock(data=[])
                mock_openai.return_value = mock_openai_instance
                with patch("sdg_hub.flow_runner.Flow") as mock_flow:
                    mock_flow_instance = MagicMock()
                    mock_flow.return_value = mock_flow_instance
                    mock_flow_instance.get_flow_from_file.return_value = {"test": "config"}
                    
                    with patch("sdg_hub.flow_runner.SDG") as mock_sdg:
                        mock_sdg_instance = MagicMock()
                        mock_sdg.return_value = mock_sdg_instance
                        mock_sdg_instance.generate.side_effect = Exception("API rate limit exceeded")
                        
                        from sdg_hub.utils.error_handling import DataGenerationError
                        with pytest.raises(DataGenerationError) as exc_info:
                            run_flow(
                                ds_path=mock_input_dataset,
                                save_path=mock_output_path,
                                endpoint="http://test.endpoint",
                                flow_path=mock_flow_config,
                                checkpoint_dir=mock_checkpoint_dir,
                            )
                        
                        assert "Data generation failed during processing" in str(exc_info.value)
                        assert "API rate limit exceeded" in exc_info.value.details

    def test_data_save_error(
        self, mock_dataset, mock_flow_config, mock_checkpoint_dir, mock_input_dataset, tmp_path
    ):
        """Test DataSaveError when save fails."""
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset
            
            with patch("sdg_hub.flow_runner.OpenAI") as mock_openai:
                mock_openai_instance = MagicMock()
                mock_openai_instance.models.list.return_value = MagicMock(data=[])
                mock_openai.return_value = mock_openai_instance
                with patch("sdg_hub.flow_runner.Flow") as mock_flow:
                    mock_flow_instance = MagicMock()
                    mock_flow.return_value = mock_flow_instance
                    mock_flow_instance.get_flow_from_file.return_value = {"test": "config"}
                    
                    with patch("sdg_hub.flow_runner.SDG") as mock_sdg:
                        mock_sdg_instance = MagicMock()
                        mock_sdg.return_value = mock_sdg_instance
                        generated_data = MagicMock()
                        generated_data.__len__ = MagicMock(return_value=10)
                        generated_data.to_json.side_effect = PermissionError("Permission denied")
                        mock_sdg_instance.generate.return_value = generated_data
                        
                        save_path = str(tmp_path / "output.jsonl")
                        
                        from sdg_hub.utils.error_handling import DataSaveError
                        with pytest.raises(DataSaveError) as exc_info:
                            run_flow(
                                ds_path=mock_input_dataset,
                                save_path=save_path,
                                endpoint="http://test.endpoint",
                                flow_path=mock_flow_config,
                                checkpoint_dir=mock_checkpoint_dir,
                            )
                        
                        assert "Failed to save generated data" in str(exc_info.value)
                        assert "Permission denied" in exc_info.value.details

    def test_flow_runner_error_unexpected_exception(
        self, mock_dataset, mock_flow_config, mock_output_path, mock_checkpoint_dir, mock_input_dataset
    ):
        """Test FlowRunnerError for unexpected exceptions."""
        with patch("sdg_hub.flow_runner.load_dataset") as mock_load_dataset:
            mock_load_dataset.side_effect = Exception("User interrupted")
            
            from sdg_hub.utils.error_handling import FlowRunnerError
            with pytest.raises(FlowRunnerError) as exc_info:
                run_flow(
                    ds_path=mock_input_dataset,
                    save_path=mock_output_path,
                    endpoint="http://test.endpoint",
                    flow_path=mock_flow_config,
                    checkpoint_dir=mock_checkpoint_dir,
                )
            
            assert "Failed to load dataset from" in str(exc_info.value)
            assert "User interrupted" in exc_info.value.details


class TestFlowRunnerCLIErrorHandling:
    """Test CLI error handling scenarios."""

    def test_cli_dataset_load_error(self, mock_flow_config, mock_checkpoint_dir, mock_input_dataset):
        """Test CLI handling of DatasetLoadError."""
        with patch("sdg_hub.flow_runner.run_flow") as mock_run_flow:
            mock_run_flow.side_effect = DatasetLoadError("Dataset load failed", "File not found")
            
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--ds_path", mock_input_dataset,
                    "--save_path", "output.json",
                    "--endpoint", "http://test.endpoint",
                    "--flow", mock_flow_config,
                    "--checkpoint_dir", mock_checkpoint_dir,
                ],
            )
            
            assert result.exit_code == 1
            assert "Error: Dataset load failed" in result.output

    def test_cli_keyboard_interrupt(self, mock_flow_config, mock_checkpoint_dir, mock_input_dataset):
        """Test CLI handling of KeyboardInterrupt."""
        with patch("sdg_hub.flow_runner.run_flow") as mock_run_flow:
            mock_run_flow.side_effect = KeyboardInterrupt()
            
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--ds_path", mock_input_dataset,
                    "--save_path", "output.json",
                    "--endpoint", "http://test.endpoint",
                    "--flow", mock_flow_config,
                    "--checkpoint_dir", mock_checkpoint_dir,
                ],
            )
            
            assert result.exit_code == 130  # Standard SIGINT exit code
            assert "Flow execution interrupted by user" in result.output

    def test_cli_unexpected_error(self, mock_flow_config, mock_checkpoint_dir, mock_input_dataset):
        """Test CLI handling of unexpected errors."""
        with patch("sdg_hub.flow_runner.run_flow") as mock_run_flow:
            mock_run_flow.side_effect = RuntimeError("Unexpected runtime error")
            
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--ds_path", mock_input_dataset,
                    "--save_path", "output.json",
                    "--endpoint", "http://test.endpoint",
                    "--flow", mock_flow_config,
                    "--checkpoint_dir", mock_checkpoint_dir,
                ],
            )
            
            assert result.exit_code == 1
            assert "Unexpected error occurred" in result.output
