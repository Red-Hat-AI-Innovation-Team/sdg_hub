"""Tests for the SDG (Synthetic Data Generator) class."""

# Standard
from unittest.mock import MagicMock, patch
import tempfile

# Third Party
from datasets import Dataset
import pytest

# First Party
from sdg_hub.sdg import SDG
from sdg_hub.flow import Flow


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {
            "instruction": ["Generate a question", "Create a task", "Write code"],
            "input": ["topic 1", "topic 2", "topic 3"],
            "output": ["question 1", "task 1", "code 1"],
        }
    )


@pytest.fixture
def mock_flow():
    """Create a mock Flow object for testing."""
    flow = MagicMock(spec=Flow)
    flow.generate.return_value = Dataset.from_dict(
        {
            "instruction": ["Generated question"],
            "input": ["Generated input"],
            "output": ["Generated output"],
        }
    )
    return flow


@pytest.fixture
def mock_multiple_flows():
    """Create multiple mock Flow objects for testing."""
    flow1 = MagicMock(spec=Flow)
    flow1.generate.return_value = Dataset.from_dict(
        {
            "instruction": ["Flow1 question"],
            "input": ["Flow1 input"],
            "output": ["Flow1 output"],
        }
    )

    flow2 = MagicMock(spec=Flow)
    flow2.generate.return_value = Dataset.from_dict(
        {
            "instruction": ["Flow2 question"],
            "input": ["Flow2 input"],
            "output": ["Flow2 output"],
        }
    )

    return [flow1, flow2]


class TestSDGInitialization:
    """Test SDG initialization and basic properties."""

    def test_init_with_single_flow(self, mock_flow):
        """Test initialization with a single flow."""
        sdg = SDG(flows=[mock_flow])
        assert len(sdg.flows) == 1
        assert sdg.flows[0] == mock_flow
        assert sdg.num_workers == 1
        assert sdg.batch_size is None
        assert sdg.save_freq is None

    def test_init_with_multiple_flows(self, mock_multiple_flows):
        """Test initialization with multiple flows."""
        sdg = SDG(flows=mock_multiple_flows)
        assert len(sdg.flows) == 2
        assert sdg.flows == mock_multiple_flows

    def test_init_with_custom_parameters(self, mock_flow):
        """Test initialization with custom parameters."""
        sdg = SDG(flows=[mock_flow], num_workers=4, batch_size=10, save_freq=5)
        assert sdg.num_workers == 4
        assert sdg.batch_size == 10
        assert sdg.save_freq == 5

    def test_init_empty_flows_list(self):
        """Test initialization with empty flows list."""
        sdg = SDG(flows=[])
        assert len(sdg.flows) == 0


class TestSDGDatasetSplitting:
    """Test dataset splitting functionality."""

    def test_split_dataset_basic(self, mock_flow, sample_dataset):
        """Test basic dataset splitting."""
        sdg = SDG(flows=[mock_flow])
        splits = sdg._split_dataset(sample_dataset, batch_size=2)

        expected = [(0, 2), (2, 3)]
        assert splits == expected

    def test_split_dataset_exact_division(self, mock_flow):
        """Test dataset splitting with exact division."""
        dataset = Dataset.from_dict({"data": list(range(10))})
        sdg = SDG(flows=[mock_flow])
        splits = sdg._split_dataset(dataset, batch_size=5)

        expected = [(0, 5), (5, 10)]
        assert splits == expected

    def test_split_dataset_single_batch(self, mock_flow, sample_dataset):
        """Test dataset splitting with batch size larger than dataset."""
        sdg = SDG(flows=[mock_flow])
        splits = sdg._split_dataset(sample_dataset, batch_size=10)

        expected = [(0, 3)]
        assert splits == expected

    def test_split_dataset_single_item_batches(self, mock_flow, sample_dataset):
        """Test dataset splitting with batch size of 1."""
        sdg = SDG(flows=[mock_flow])
        splits = sdg._split_dataset(sample_dataset, batch_size=1)

        expected = [(0, 1), (1, 2), (2, 3)]
        assert splits == expected


class TestSDGDataGeneration:
    """Test data generation functionality."""

    def test_generate_data_static_method_success(self, mock_flow, sample_dataset):
        """Test _generate_data static method with successful generation."""
        result = SDG._generate_data([mock_flow], (0, 2), sample_dataset, 0)
        assert result is not None
        mock_flow.generate.assert_called_once()

    def test_generate_data_static_method_with_multiple_flows(
        self, mock_multiple_flows, sample_dataset
    ):
        """Test _generate_data static method with multiple flows."""
        result = SDG._generate_data(mock_multiple_flows, (0, 2), sample_dataset, 0)
        assert result is not None

        # Both flows should be called in sequence
        mock_multiple_flows[0].generate.assert_called_once()
        mock_multiple_flows[1].generate.assert_called_once()

    def test_generate_data_static_method_with_exception(self, sample_dataset):
        """Test _generate_data static method handles exceptions."""
        failing_flow = MagicMock(spec=Flow)
        failing_flow.generate.side_effect = Exception("Flow failed")

        with patch("sdg_hub.sdg.logger") as mock_logger:
            result = SDG._generate_data([failing_flow], (0, 2), sample_dataset, 0)
            assert result is None
            mock_logger.error.assert_called_once()

    def test_generate_without_batch_size_single_flow(self, mock_flow, sample_dataset):
        """Test generate method without batch size using single flow."""
        sdg = SDG(flows=[mock_flow])

        with patch.object(sdg, "_split_dataset") as mock_split:
            with patch("sdg_hub.sdg.Checkpointer") as mock_checkpointer_class:
                mock_checkpointer = MagicMock()
                mock_checkpointer.load_existing_data.return_value = (
                    sample_dataset,
                    None,
                )
                mock_checkpointer_class.return_value = mock_checkpointer

                result = sdg.generate(sample_dataset)

                # Should not call split_dataset when batch_size is None
                mock_split.assert_not_called()
                mock_flow.generate.assert_called_once_with(sample_dataset)

    def test_generate_without_batch_size_multiple_flows(
        self, mock_multiple_flows, sample_dataset
    ):
        """Test generate method without batch size using multiple flows."""
        sdg = SDG(flows=mock_multiple_flows)

        with patch("sdg_hub.sdg.Checkpointer") as mock_checkpointer_class:
            mock_checkpointer = MagicMock()
            mock_checkpointer.load_existing_data.return_value = (sample_dataset, None)
            mock_checkpointer_class.return_value = mock_checkpointer

            result = sdg.generate(sample_dataset)

            # Both flows should be called in sequence
            mock_multiple_flows[0].generate.assert_called_once_with(sample_dataset)
            mock_multiple_flows[1].generate.assert_called_once()

    @patch("sdg_hub.sdg.safe_concatenate_datasets")
    def test_generate_with_batch_size(
        self, mock_concatenate, mock_flow, sample_dataset
    ):
        """Test generate method with batch size."""
        mock_concatenate.return_value = sample_dataset
        sdg = SDG(flows=[mock_flow], batch_size=2, num_workers=1)

        with patch("sdg_hub.sdg.Checkpointer") as mock_checkpointer_class:
            mock_checkpointer = MagicMock()
            mock_checkpointer.load_existing_data.return_value = (sample_dataset, None)
            mock_checkpointer.should_save_checkpoint.return_value = False
            mock_checkpointer_class.return_value = mock_checkpointer

            with patch.object(sdg, "_generate_data") as mock_generate_data:
                mock_generate_data.return_value = Dataset.from_dict({"test": ["data"]})

                result = sdg.generate(sample_dataset)

                # Should call _generate_data for each split
                assert mock_generate_data.call_count > 0
                mock_concatenate.assert_called_once()

    def test_generate_with_pre_generated_data(self, mock_flow, sample_dataset):
        """Test generate method with pre-generated data from checkpoints."""
        pre_generated = Dataset.from_dict({"pre": ["generated"]})
        empty_dataset = Dataset.from_dict({})

        with patch("sdg_hub.sdg.Checkpointer") as mock_checkpointer_class:
            mock_checkpointer = MagicMock()
            mock_checkpointer.load_existing_data.return_value = (
                empty_dataset,
                pre_generated,
            )
            mock_checkpointer_class.return_value = mock_checkpointer

            sdg = SDG(flows=[mock_flow])
            result = sdg.generate(sample_dataset)

            assert result == pre_generated
            mock_flow.generate.assert_not_called()


class TestSDGMultithreading:
    """Test multithreading functionality."""

    @patch("sdg_hub.sdg.ThreadPoolExecutor")
    @patch("sdg_hub.sdg.safe_concatenate_datasets")
    def test_generate_with_multiple_workers(
        self, mock_concatenate, mock_executor_class, mock_flow, sample_dataset
    ):
        """Test generate method with multiple workers."""
        mock_concatenate.return_value = sample_dataset
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        # Mock futures
        mock_future = MagicMock()
        mock_future.result.return_value = Dataset.from_dict({"test": ["data"]})
        mock_executor.submit.return_value = mock_future

        with patch("sdg_hub.sdg.as_completed") as mock_as_completed:
            mock_as_completed.return_value = [mock_future]

            sdg = SDG(flows=[mock_flow], batch_size=2, num_workers=4)

            with patch("sdg_hub.sdg.Checkpointer") as mock_checkpointer_class:
                mock_checkpointer = MagicMock()
                mock_checkpointer.load_existing_data.return_value = (
                    sample_dataset,
                    None,
                )
                mock_checkpointer.should_save_checkpoint.return_value = False
                mock_checkpointer_class.return_value = mock_checkpointer

                result = sdg.generate(sample_dataset)

                # Verify ThreadPoolExecutor was called with correct max_workers
                mock_executor_class.assert_called_once_with(max_workers=4)


class TestSDGCheckpointing:
    """Test checkpointing functionality."""

    def test_generate_with_checkpointing(self, mock_flow, sample_dataset):
        """Test generate method with checkpointing enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sdg = SDG(flows=[mock_flow], batch_size=2, save_freq=1)

            with patch("sdg_hub.sdg.Checkpointer") as mock_checkpointer_class:
                mock_checkpointer = MagicMock()
                mock_checkpointer.load_existing_data.return_value = (
                    sample_dataset,
                    None,
                )
                mock_checkpointer.should_save_checkpoint.return_value = True
                mock_checkpointer_class.return_value = mock_checkpointer

                with patch("sdg_hub.sdg.safe_concatenate_datasets") as mock_concatenate:
                    mock_concatenate.return_value = sample_dataset

                    result = sdg.generate(sample_dataset, checkpoint_dir=temp_dir)

                    # Verify checkpointer was initialized with correct parameters
                    mock_checkpointer_class.assert_called_once_with(temp_dir, 1)
                    mock_checkpointer.load_existing_data.assert_called_once_with(
                        sample_dataset
                    )

    def test_generate_without_checkpointing(self, mock_flow, sample_dataset):
        """Test generate method without checkpointing."""
        sdg = SDG(flows=[mock_flow])

        with patch("sdg_hub.sdg.Checkpointer") as mock_checkpointer_class:
            mock_checkpointer = MagicMock()
            mock_checkpointer.load_existing_data.return_value = (sample_dataset, None)
            mock_checkpointer_class.return_value = mock_checkpointer

            result = sdg.generate(sample_dataset)

            # Verify checkpointer was initialized with None checkpoint_dir
            mock_checkpointer_class.assert_called_once_with(None, None)


class TestSDGErrorHandling:
    """Test error handling in various scenarios."""

    def test_generate_with_failing_flow(self, sample_dataset):
        """Test generate method handles flow failures gracefully."""
        failing_flow = MagicMock(spec=Flow)
        failing_flow.generate.side_effect = Exception("Flow failed")

        sdg = SDG(flows=[failing_flow])

        with patch("sdg_hub.sdg.Checkpointer") as mock_checkpointer_class:
            mock_checkpointer = MagicMock()
            mock_checkpointer.load_existing_data.return_value = (sample_dataset, None)
            mock_checkpointer_class.return_value = mock_checkpointer

            with pytest.raises(Exception, match="Flow failed"):
                sdg.generate(sample_dataset)

    def test_generate_data_with_none_result(self, mock_flow, sample_dataset):
        """Test generate method handles None results from _generate_data."""
        sdg = SDG(flows=[mock_flow], batch_size=2)

        with patch("sdg_hub.sdg.Checkpointer") as mock_checkpointer_class:
            mock_checkpointer = MagicMock()
            mock_checkpointer.load_existing_data.return_value = (sample_dataset, None)
            mock_checkpointer.should_save_checkpoint.return_value = False
            mock_checkpointer_class.return_value = mock_checkpointer

            with patch.object(sdg, "_generate_data") as mock_generate_data:
                mock_generate_data.return_value = None  # Simulate failure

                with patch("sdg_hub.sdg.safe_concatenate_datasets") as mock_concatenate:
                    mock_concatenate.return_value = Dataset.from_dict({"empty": []})

                    result = sdg.generate(sample_dataset)

                    # Should handle None results gracefully
                    assert result is not None


class TestSDGIntegration:
    """Integration tests for SDG functionality."""

    def test_end_to_end_single_flow(self, sample_dataset):
        """Test end-to-end generation with a single flow."""
        # Create a real Flow-like mock that maintains call order
        flow = MagicMock(spec=Flow)
        expected_output = Dataset.from_dict(
            {
                "instruction": ["Generated instruction"],
                "input": ["Generated input"],
                "output": ["Generated output"],
            }
        )
        flow.generate.return_value = expected_output

        sdg = SDG(flows=[flow])

        with patch("sdg_hub.sdg.Checkpointer") as mock_checkpointer_class:
            mock_checkpointer = MagicMock()
            mock_checkpointer.load_existing_data.return_value = (sample_dataset, None)
            mock_checkpointer_class.return_value = mock_checkpointer

            result = sdg.generate(sample_dataset)

            flow.generate.assert_called_once_with(sample_dataset)
            assert result == expected_output

    def test_end_to_end_batched_generation(self, sample_dataset):
        """Test end-to-end batched generation."""
        flow = MagicMock(spec=Flow)
        flow.generate.return_value = Dataset.from_dict({"result": ["batch_result"]})

        sdg = SDG(flows=[flow], batch_size=1, num_workers=1)

        with patch("sdg_hub.sdg.Checkpointer") as mock_checkpointer_class:
            mock_checkpointer = MagicMock()
            mock_checkpointer.load_existing_data.return_value = (sample_dataset, None)
            mock_checkpointer.should_save_checkpoint.return_value = False
            mock_checkpointer_class.return_value = mock_checkpointer

            with patch("sdg_hub.sdg.safe_concatenate_datasets") as mock_concatenate:
                expected_result = Dataset.from_dict({"final": ["result"]})
                mock_concatenate.return_value = expected_result

                result = sdg.generate(sample_dataset)

                # Should process each item in the dataset
                assert flow.generate.call_count == len(sample_dataset)
                assert result == expected_result
