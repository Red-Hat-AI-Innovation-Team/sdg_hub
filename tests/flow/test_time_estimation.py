# SPDX-License-Identifier: Apache-2.0
"""Tests for time estimation functionality in Flow class."""

# Standard
from unittest.mock import patch
import tempfile

# Third Party
from datasets import Dataset
import pytest

# First Party
from sdg_hub import FlowMetadata
from sdg_hub.core.flow.base import Flow
from sdg_hub.core.flow.metadata import RecommendedModels
from sdg_hub.core.utils.error_handling import EmptyDatasetError, FlowValidationError
from sdg_hub.core.utils.time_estimator import (
    calculate_block_throughput,
    calculate_time_with_pipeline,
    estimate_execution_time,
    is_llm_using_block,
)
from tests.flow.conftest import MockBlock


class TestTimeEstimation:
    """Test time estimation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        self.test_metadata = FlowMetadata(
            name="Test Flow",
            description="A test flow for time estimation",
            version="1.0.0",
            author="Test Author",
            recommended_models=RecommendedModels(
                default="test-model", compatible=["alt-model"], experimental=[]
            ),
            tags=["test"],
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        # Standard
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_mock_block(self, name="test_block", input_cols=None, output_cols=None):
        """Create a mock block for testing."""
        return MockBlock(
            block_name=name,
            input_cols=input_cols or ["input"],
            output_cols=output_cols or ["output"],
        )

    def create_mock_llm_block(self, name="llm_block", async_mode=False):
        """Create a mock LLM block with async capabilities."""
        block = MockBlock(block_name=name, input_cols=["input"], output_cols=["output"])
        # Add LLM attributes
        block.model = "test-model"
        block.api_base = "http://localhost:8000/v1"
        block.api_key = "EMPTY"
        block.async_mode = async_mode
        return block

    def test_estimate_total_time_without_cached_results(self):
        """Test estimate_total_time when no cached dry_run results exist."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"] * 10})

        # Mock dry_run to return predictable results
        mock_dry_run_results = {
            "flow_name": "Test Flow",
            "sample_size": 2,
            "original_dataset_size": 10,
            "execution_time_seconds": 2.0,
            "blocks_executed": [
                {
                    "block_name": "test_block",
                    "block_type": "MockBlock",
                    "execution_time_seconds": 2.0,
                    "input_rows": 2,
                    "output_rows": 2,
                    "parameters_used": {},
                }
            ],
            "execution_successful": True,
        }

        with patch(
            "sdg_hub.core.flow.base.Flow.dry_run", return_value=mock_dry_run_results
        ):
            result = flow.estimate_total_time(dataset, sample_size=2)

        assert "estimated_time_seconds" in result
        assert result["estimated_time_seconds"] > 0
        assert "total_estimated_requests" in result

    def test_estimate_total_time_with_cached_results(self):
        """Test estimate_total_time when cached dry_run results exist."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"] * 10})

        # Set up cached results
        flow._cached_dry_run_results = {
            "flow_name": "Test Flow",
            "sample_size": 2,
            "original_dataset_size": 10,
            "execution_time_seconds": 2.0,
            "blocks_executed": [
                {
                    "block_name": "test_block",
                    "block_type": "MockBlock",
                    "execution_time_seconds": 2.0,
                    "input_rows": 2,
                    "output_rows": 2,
                    "parameters_used": {},
                }
            ],
            "execution_successful": True,
        }

        # Should use cached results, not call dry_run
        with patch("sdg_hub.core.flow.base.Flow.dry_run") as mock_dry_run:
            result = flow.estimate_total_time(dataset, sample_size=2)

        # dry_run should not be called when cached results match sample_size
        mock_dry_run.assert_not_called()

        assert "estimated_time_seconds" in result
        assert result["estimated_time_seconds"] > 0

    def test_estimate_total_time_with_async_blocks(self):
        """Test estimate_total_time with async blocks requiring two dry runs."""
        # Create flow with async block
        async_block = self.create_mock_llm_block("async_llm_block", async_mode=True)
        flow = Flow(blocks=[async_block], metadata=self.test_metadata)
        flow._model_config_set = True

        dataset = Dataset.from_dict({"input": ["test"] * 100})

        # Mock dry_run results for 1 sample
        dry_run_1_sample = {
            "flow_name": "Test Flow",
            "sample_size": 1,
            "original_dataset_size": 100,
            "execution_time_seconds": 1.0,
            "blocks_executed": [
                {
                    "block_name": "async_llm_block",
                    "block_type": "MockBlock",
                    "execution_time_seconds": 1.0,
                    "input_rows": 1,
                    "output_rows": 1,
                    "parameters_used": {"model": "test-model"},
                }
            ],
            "execution_successful": True,
        }

        # Mock dry_run results for 5 samples
        dry_run_5_samples = {
            "flow_name": "Test Flow",
            "sample_size": 5,
            "original_dataset_size": 100,
            "execution_time_seconds": 2.0,
            "blocks_executed": [
                {
                    "block_name": "async_llm_block",
                    "block_type": "MockBlock",
                    "execution_time_seconds": 2.0,
                    "input_rows": 5,
                    "output_rows": 5,
                    "parameters_used": {"model": "test-model"},
                }
            ],
            "execution_successful": True,
        }

        call_count = [0]

        def mock_dry_run_side_effect(_dataset, sample_size, _runtime_params=None):
            call_count[0] += 1
            if sample_size == 1:
                return dry_run_1_sample
            else:
                return dry_run_5_samples

        with patch(
            "sdg_hub.core.flow.base.Flow.dry_run", side_effect=mock_dry_run_side_effect
        ):
            result = flow.estimate_total_time(dataset, sample_size=5)

        # Should call dry_run twice for async blocks
        assert call_count[0] == 2

        assert "estimated_time_seconds" in result
        assert result["estimated_time_seconds"] > 0
        assert "total_estimated_requests" in result
        assert result["total_estimated_requests"] > 0

    def test_estimate_total_time_no_async_blocks(self):
        """Test estimate_total_time with only sequential blocks."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"] * 100})

        # Mock dry_run results for sequential block
        mock_dry_run = {
            "flow_name": "Test Flow",
            "sample_size": 2,
            "original_dataset_size": 100,
            "execution_time_seconds": 2.0,
            "blocks_executed": [
                {
                    "block_name": "test_block",
                    "block_type": "MockBlock",
                    "execution_time_seconds": 2.0,
                    "input_rows": 2,
                    "output_rows": 2,
                    "parameters_used": {},
                }
            ],
            "execution_successful": True,
        }

        call_count = [0]

        def mock_dry_run_side_effect(_dataset, sample_size, _runtime_params=None):
            call_count[0] += 1
            return mock_dry_run

        with patch(
            "sdg_hub.core.flow.base.Flow.dry_run", side_effect=mock_dry_run_side_effect
        ):
            result = flow.estimate_total_time(dataset, sample_size=2)

        # Should only call dry_run once for sequential blocks
        assert call_count[0] == 1
        assert "estimated_time_seconds" in result
        assert result["estimated_time_seconds"] > 0

    def test_estimate_total_time_with_max_concurrency(self):
        """Test estimate_total_time with max_concurrency parameter."""
        async_block = self.create_mock_llm_block("async_llm_block", async_mode=True)
        flow = Flow(blocks=[async_block], metadata=self.test_metadata)
        flow._model_config_set = True

        dataset = Dataset.from_dict({"input": ["test"] * 100})

        # Mock dry_run results for 1 sample
        mock_dry_run_1 = {
            "flow_name": "Test Flow",
            "sample_size": 1,
            "original_dataset_size": 100,
            "execution_time_seconds": 1.0,
            "blocks_executed": [
                {
                    "block_name": "async_llm_block",
                    "block_type": "MockBlock",
                    "execution_time_seconds": 1.0,
                    "input_rows": 1,
                    "output_rows": 1,
                    "parameters_used": {"model": "test-model"},
                }
            ],
            "execution_successful": True,
        }
        
        # Mock dry_run results for 5 samples (should scale)
        mock_dry_run_5 = {
            "flow_name": "Test Flow",
            "sample_size": 5,
            "original_dataset_size": 100,
            "execution_time_seconds": 2.0,
            "blocks_executed": [
                {
                    "block_name": "async_llm_block",
                    "block_type": "MockBlock",
                    "execution_time_seconds": 2.0,
                    "input_rows": 5,
                    "output_rows": 5,
                    "parameters_used": {"model": "test-model"},
                }
            ],
            "execution_successful": True,
        }

        def mock_dry_run_side_effect(_dataset, sample_size, _runtime_params=None):
            if sample_size == 1:
                return mock_dry_run_1
            else:
                return mock_dry_run_5

        with patch("sdg_hub.core.flow.base.Flow.dry_run", side_effect=mock_dry_run_side_effect):
            # Test with low concurrency
            result_low = flow.estimate_total_time(
                dataset, sample_size=5, max_concurrency=10
            )

            # Test with high concurrency
            result_high = flow.estimate_total_time(
                dataset, sample_size=5, max_concurrency=100
            )

        # Both should have valid results
        assert result_low["estimated_time_seconds"] > 0
        assert result_high["estimated_time_seconds"] > 0

        # Higher concurrency should generally be faster (or equal)
        assert (
            result_high["estimated_time_seconds"]
            <= result_low["estimated_time_seconds"]
        )

    def test_estimate_total_time_caching_behavior(self):
        """Test that dry_run properly caches results for estimate_total_time."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        # Use unique values to avoid duplicate validation error
        dataset = Dataset.from_dict({"input": [f"test{i}" for i in range(10)]})

        # Initially, no cached results
        assert getattr(flow, "_cached_dry_run_results", None) is None

        # Run dry_run
        dry_run_result = flow.dry_run(dataset, sample_size=2)

        # Results should be cached
        assert getattr(flow, "_cached_dry_run_results", None) is not None
        cached = getattr(flow, "_cached_dry_run_results", None)
        assert cached == dry_run_result
        assert cached["sample_size"] == 2

        # estimate_total_time should use cached results
        with patch("sdg_hub.core.flow.base.Flow.dry_run") as mock_dry_run:
            flow.estimate_total_time(dataset, sample_size=2)
            # Should not call dry_run again
            mock_dry_run.assert_not_called()

    def test_estimate_total_time_different_sample_sizes(self):
        """Test estimate_total_time with different sample sizes than cached."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"] * 10})

        # Cache results for sample_size=2
        object.__setattr__(
            flow,
            "_cached_dry_run_results",
            {
                "flow_name": "Test Flow",
                "sample_size": 2,
                "original_dataset_size": 10,
                "execution_time_seconds": 2.0,
                "blocks_executed": [],
                "execution_successful": True,
            },
        )

        # Request estimate with different sample size
        new_dry_run_results = {
            "flow_name": "Test Flow",
            "sample_size": 5,
            "original_dataset_size": 10,
            "execution_time_seconds": 5.0,
            "blocks_executed": [],
            "execution_successful": True,
        }

        with patch(
            "sdg_hub.core.flow.base.Flow.dry_run", return_value=new_dry_run_results
        ) as mock_dry_run:
            flow.estimate_total_time(dataset, sample_size=5)
            # Should call dry_run with new sample size
            mock_dry_run.assert_called_once()

    def test_estimate_total_time_with_runtime_params(self):
        """Test estimate_total_time passes runtime_params to dry_run."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"] * 10})

        runtime_params = {"test_block": {"temperature": 0.5, "max_tokens": 100}}

        mock_dry_run_results = {
            "flow_name": "Test Flow",
            "sample_size": 2,
            "original_dataset_size": 10,
            "execution_time_seconds": 2.0,
            "blocks_executed": [],
            "execution_successful": True,
        }

        with patch(
            "sdg_hub.core.flow.base.Flow.dry_run", return_value=mock_dry_run_results
        ) as mock_dry_run:
            flow.estimate_total_time(dataset, runtime_params=runtime_params)

            # Verify runtime_params were passed to dry_run
            mock_dry_run.assert_called_once()
            # dry_run is called with (self, dataset, sample_size, runtime_params)
            # Check positional args - runtime_params should be the 3rd argument (index 2)
            call_args, call_kwargs = mock_dry_run.call_args
            # dry_run is called as self.dry_run(dataset, sample_size, runtime_params)
            # So runtime_params should be at position 2 in call_args
            if len(call_args) > 2:
                passed_runtime_params = call_args[2]
            else:
                passed_runtime_params = call_kwargs.get("runtime_params")
            # Check if runtime_params was passed and equals expected value
            assert passed_runtime_params == runtime_params

    def test_estimate_total_time_empty_flow(self):
        """Test estimate_total_time with empty flow raises error."""
        flow = Flow(blocks=[], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"] * 10})

        with pytest.raises(FlowValidationError) as exc_info:
            flow.estimate_total_time(dataset)

        assert "empty flow" in str(exc_info.value).lower()

    def test_estimate_total_time_empty_dataset(self):
        """Test estimate_total_time with empty dataset raises error."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        empty_dataset = Dataset.from_dict({"input": []})

        with pytest.raises(EmptyDatasetError):
            flow.estimate_total_time(empty_dataset)

    def test_estimate_total_time_max_concurrency_validation(self):
        """Test estimate_total_time validates max_concurrency parameter."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2"]})

        # Test with zero value
        with pytest.raises(FlowValidationError) as exc_info:
            flow.estimate_total_time(dataset, max_concurrency=0)
        assert "must be greater than 0" in str(exc_info.value)

        # Test with negative value
        with pytest.raises(FlowValidationError) as exc_info:
            flow.estimate_total_time(dataset, max_concurrency=-1)
        assert "must be greater than 0" in str(exc_info.value)

        # Test with boolean value
        with pytest.raises(FlowValidationError) as exc_info:
            flow.estimate_total_time(dataset, max_concurrency=True)
        assert "must be an int" in str(exc_info.value)

        # Test with string value
        with pytest.raises(FlowValidationError) as exc_info:
            flow.estimate_total_time(dataset, max_concurrency="10")
        assert "must be an int" in str(exc_info.value)

        # Test with float value
        with pytest.raises(FlowValidationError) as exc_info:
            flow.estimate_total_time(dataset, max_concurrency=10.5)
        assert "must be an int" in str(exc_info.value)

        # Test with None (should not raise)
        result = flow.estimate_total_time(dataset, max_concurrency=None)
        assert result is not None

        # Test with valid positive integer
        result = flow.estimate_total_time(dataset, max_concurrency=10)
        assert result is not None

    def test_estimate_total_time_wraps_estimator_errors(self):
        """Test that errors from estimate_execution_time are wrapped as FlowValidationError."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2"]})

        # Mock estimate_execution_time to raise an exception
        with patch(
            "sdg_hub.core.flow.base.estimate_execution_time"
        ) as mock_estimate:
            mock_estimate.side_effect = ValueError("Invalid time measurements")

            # Should wrap the ValueError as FlowValidationError
            with pytest.raises(FlowValidationError) as exc_info:
                flow.estimate_total_time(dataset)

            assert "Error estimating execution time" in str(exc_info.value)
            assert "Invalid time measurements" in str(exc_info.value)
            # Check exception chaining
            assert isinstance(exc_info.value.__cause__, ValueError)

        # Also test with async blocks to cover the second error handler
        async_block = self.create_mock_block("async_block")
        async_block.async_mode = True
        flow_with_async = Flow(blocks=[async_block], metadata=self.test_metadata)

        with patch(
            "sdg_hub.core.flow.base.estimate_execution_time"
        ) as mock_estimate:
            mock_estimate.side_effect = RuntimeError("Throughput calculation failed")

            with pytest.raises(FlowValidationError) as exc_info:
                flow_with_async.estimate_total_time(dataset)

            assert "Error estimating execution time" in str(exc_info.value)
            assert "Throughput calculation failed" in str(exc_info.value)
            assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_dry_run_caches_results(self):
        """Test that dry_run properly sets _cached_dry_run_results."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2", "test3"]})

        # Initially no cached results
        assert getattr(flow, "_cached_dry_run_results", None) is None

        # Run dry_run
        result = flow.dry_run(dataset, sample_size=2)

        # Check results are cached
        cached = getattr(flow, "_cached_dry_run_results", None)
        assert cached is not None
        assert cached == result
        assert cached["sample_size"] == 2
        assert cached["original_dataset_size"] == 3
        assert cached["execution_successful"] is True

    def test_dry_run_updates_cache_on_each_run(self):
        """Test that each dry_run updates the cached results."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        # Use unique values to avoid duplicate validation error
        dataset = Dataset.from_dict(
            {"input": ["test1", "test2", "test3", "test4", "test5"]}
        )

        # First dry_run with sample_size=1
        flow.dry_run(dataset, sample_size=1)
        cached1 = getattr(flow, "_cached_dry_run_results", None)
        assert cached1["sample_size"] == 1

        # Second dry_run with sample_size=3
        result2 = flow.dry_run(dataset, sample_size=3)
        cached2 = getattr(flow, "_cached_dry_run_results", None)
        assert cached2["sample_size"] == 3
        assert cached2 == result2


class TestTimeEstimatorIntegration:
    """Test integration with time_estimator module."""

    def test_time_estimator_module_functions(self):
        """Test that time_estimator module functions are called correctly."""

        # Test is_llm_using_block
        llm_block_info = {
            "block_type": "LLMChatBlock",
            "parameters_used": {"model": "test-model"},
        }
        assert is_llm_using_block(llm_block_info) is True

        non_llm_block_info = {"block_type": "TextConcat", "parameters_used": {}}
        assert is_llm_using_block(non_llm_block_info) is False

        # Test calculate_block_throughput
        block_1 = {
            "execution_time_seconds": 1.0,
            "input_rows": 1,
            "block_name": "test_block",
        }
        block_2 = {
            "execution_time_seconds": 2.0,
            "input_rows": 5,
            "block_name": "test_block",
        }

        throughput_result = calculate_block_throughput(block_1, block_2, 1, 5)
        assert "throughput" in throughput_result
        assert "amplification" in throughput_result
        assert "startup_overhead" in throughput_result
        assert throughput_result["throughput"] > 0

        # Test calculate_time_with_pipeline
        time_result = calculate_time_with_pipeline(
            num_requests=100, throughput=10.0, startup_overhead=0.5, max_concurrent=50
        )
        assert time_result > 0

        # Test estimate_execution_time with single dry run
        dry_run_1 = {
            "sample_size": 2,
            "execution_time_seconds": 2.0,
            "blocks_executed": [],
        }

        single_result = estimate_execution_time(
            dry_run_1=dry_run_1, dry_run_2=None, total_dataset_size=100
        )
        assert "estimated_time_seconds" in single_result
        assert single_result["estimated_time_seconds"] > 0

        # Test estimate_execution_time with two dry runs
        dry_run_2 = {
            "sample_size": 5,
            "execution_time_seconds": 4.0,
            "blocks_executed": [
                {
                    "block_name": "llm_block",
                    "block_type": "LLMChatBlock",
                    "execution_time_seconds": 4.0,
                    "input_rows": 5,
                    "output_rows": 5,
                    "parameters_used": {"model": "test"},
                }
            ],
        }

        dry_run_1_with_blocks = {
            "sample_size": 1,
            "execution_time_seconds": 1.0,
            "blocks_executed": [
                {
                    "block_name": "llm_block",
                    "block_type": "LLMChatBlock",
                    "execution_time_seconds": 1.0,
                    "input_rows": 1,
                    "output_rows": 1,
                    "parameters_used": {"model": "test"},
                }
            ],
        }

        dual_result = estimate_execution_time(
            dry_run_1=dry_run_1_with_blocks,
            dry_run_2=dry_run_2,
            total_dataset_size=1000,
            max_concurrency=100,
        )
        assert "estimated_time_seconds" in dual_result
        assert "block_estimates" in dual_result
        assert "total_estimated_requests" in dual_result

    def test_time_estimator_edge_cases(self):
        """Test edge cases in time estimator functions."""

        # Test with zero execution time
        block_zero_time = {
            "execution_time_seconds": 0,
            "input_rows": 10,
            "block_name": "test",
        }

        with pytest.raises(ValueError) as exc_info:
            calculate_block_throughput(block_zero_time, block_zero_time, 10, 10)
        assert "Cannot calculate throughput" in str(exc_info.value)

        # Test with zero requests
        time_zero = calculate_time_with_pipeline(
            num_requests=0, throughput=10.0, startup_overhead=0.5, max_concurrent=100
        )
        assert time_zero == 0

        # Test with very low concurrency
        time_low_concurrent = calculate_time_with_pipeline(
            num_requests=1000, throughput=100.0, startup_overhead=0.1, max_concurrent=1
        )
        assert time_low_concurrent > 0

        # Test with very high concurrency
        time_high_concurrent = calculate_time_with_pipeline(
            num_requests=1000,
            throughput=100.0,
            startup_overhead=0.1,
            max_concurrent=1000,
        )
        assert time_high_concurrent > 0
        assert time_high_concurrent < time_low_concurrent

        # Test with invalid max_concurrent values (should be clamped to 1)
        time_zero_concurrent = calculate_time_with_pipeline(
            num_requests=100, throughput=10.0, startup_overhead=1.0, max_concurrent=0
        )
        assert time_zero_concurrent > 0  # Should not crash or return invalid result

        time_negative_concurrent = calculate_time_with_pipeline(
            num_requests=100, throughput=10.0, startup_overhead=1.0, max_concurrent=-5
        )
        assert time_negative_concurrent > 0  # Should not crash

        # Both should produce the same result (clamped to 1)
        time_one_concurrent = calculate_time_with_pipeline(
            num_requests=100, throughput=10.0, startup_overhead=1.0, max_concurrent=1
        )
        assert time_zero_concurrent == time_one_concurrent
        assert time_negative_concurrent == time_one_concurrent

        # Test that high throughput values are preserved (not capped at 0.1)
        # This test would catch the min/max bug
        dry_run_high_throughput = {
            "sample_size": 100,
            "execution_time_seconds": 0.1,  # 100 rows in 0.1 seconds
            "blocks_executed": [
                {
                    "block_name": "high_throughput_block",
                    "execution_time_seconds": 0.1,
                    "input_rows": 100,
                    "block_type": "LLMChatBlock",
                    "parameters_used": {"model": "gpt-4"},
                }
            ]
        }
        
        # Calculate with very high throughput (1000 req/sec based on 100 rows in 0.1 second)
        result = estimate_execution_time(
            dry_run_1=dry_run_high_throughput,
            dry_run_2={
                "sample_size": 200,
                "execution_time_seconds": 0.2,
                "blocks_executed": [
                    {
                        "block_name": "high_throughput_block",
                        "execution_time_seconds": 0.2,
                        "input_rows": 200,
                        "block_type": "LLMChatBlock",
                        "parameters_used": {"model": "gpt-4"},
                    }
                ]
            },
            total_dataset_size=10000,
            max_concurrency=100
        )
        
        # With correct max() function: time = 10000/1000 = 10 seconds
        # With incorrect min() function: time = 10000/0.1 = 100000 seconds
        # So if estimated time is < 1000 seconds, we're using max() correctly
        assert result["estimated_time_seconds"] < 1000, (
            f"Estimated time {result['estimated_time_seconds']} is too high, "
            "suggesting min() is being used instead of max() for throughput flooring"
        )
