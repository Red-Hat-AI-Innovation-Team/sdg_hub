# SPDX-License-Identifier: Apache-2.0
"""Tests for time estimation functionality in Flow class."""

# ruff: noqa: I001
# Standard
import tempfile

# Third Party
import pandas as pd
import pytest

# First Party
from sdg_hub import FlowMetadata
from sdg_hub.core.flow.base import Flow
from sdg_hub.core.flow.metadata import RecommendedModels
from sdg_hub.core.utils.time_estimator import (
    calculate_block_throughput,
    calculate_time_with_pipeline,
    estimate_execution_time,
    is_llm_using_block,
)
from tests.flow.conftest import MockBlock


class TestTimeEstimation:
    """Test time estimation functionality via dry_run."""

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
        block.model = "test-model"
        block.api_base = "http://localhost:8000/v1"
        block.api_key = "EMPTY"
        block.async_mode = async_mode
        return block

    def test_dry_run_without_estimation(self):
        """Test dry_run without enable_time_estimation flag."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = pd.DataFrame({"input": [f"test{i}" for i in range(10)]})

        result = flow.dry_run(dataset, sample_size=2)

        assert "sample_size" in result
        assert "blocks_executed" in result
        # Time estimation is not returned in results (only displayed in table)
        assert "time_estimation" not in result

    def test_dry_run_with_estimation_sync_blocks(self):
        """Test dry_run with enable_time_estimation for synchronous blocks."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = pd.DataFrame({"input": [f"test{i}" for i in range(10)]})

        # Time estimation is displayed in table but not returned in results
        result = flow.dry_run(dataset, sample_size=2, enable_time_estimation=True)

        assert "sample_size" in result
        assert "blocks_executed" in result
        assert result["execution_successful"] is True

    def test_dry_run_with_estimation_async_blocks(self):
        """Test dry_run with enable_time_estimation for async blocks."""
        async_block = self.create_mock_llm_block("async_block", async_mode=True)
        flow = Flow(blocks=[async_block], metadata=self.test_metadata)
        flow._model_config_set = True
        dataset = pd.DataFrame({"input": [f"test{i}" for i in range(10)]})

        # Time estimation is displayed in table but not returned in results
        result = flow.dry_run(
            dataset, sample_size=5, enable_time_estimation=True, max_concurrency=100
        )

        assert "sample_size" in result
        assert "blocks_executed" in result
        assert result["execution_successful"] is True

    def test_dry_run_estimation_different_concurrency_levels(self):
        """Test that dry_run completes successfully with different max_concurrency values."""
        async_block = self.create_mock_llm_block("async_block", async_mode=True)
        flow = Flow(blocks=[async_block], metadata=self.test_metadata)
        flow._model_config_set = True
        dataset = pd.DataFrame({"input": [f"test{i}" for i in range(10)]})

        # Both should complete successfully (estimation displayed but not returned)
        result_low = flow.dry_run(
            dataset, sample_size=5, enable_time_estimation=True, max_concurrency=10
        )
        result_high = flow.dry_run(
            dataset, sample_size=5, enable_time_estimation=True, max_concurrency=100
        )

        assert result_low["execution_successful"] is True
        assert result_high["execution_successful"] is True

    def test_dry_run_estimation_sample_size_1(self):
        """Test dry_run estimation with sample_size=1 triggers second run with 5."""
        async_block = self.create_mock_llm_block("async_block", async_mode=True)
        flow = Flow(blocks=[async_block], metadata=self.test_metadata)
        flow._model_config_set = True
        dataset = pd.DataFrame({"input": [f"test{i}" for i in range(10)]})

        result = flow.dry_run(
            dataset, sample_size=1, enable_time_estimation=True, max_concurrency=100
        )

        assert result["sample_size"] == 1
        assert result["execution_successful"] is True

    def test_dry_run_estimation_sample_size_other(self):
        """Test dry_run estimation with sample_size=3 runs both 1 and 5."""
        async_block = self.create_mock_llm_block("async_block", async_mode=True)
        flow = Flow(blocks=[async_block], metadata=self.test_metadata)
        flow._model_config_set = True
        dataset = pd.DataFrame({"input": [f"test{i}" for i in range(10)]})

        result = flow.dry_run(
            dataset, sample_size=3, enable_time_estimation=True, max_concurrency=100
        )

        assert result["sample_size"] == 3
        assert result["execution_successful"] is True

    def test_estimate_sample_size_1_triggers_run_with_5(self):
        """Test sample_size=1 with estimation triggers second run with 5."""
        async_block = self.create_mock_llm_block("async_block", async_mode=True)
        flow = Flow(blocks=[async_block], metadata=self.test_metadata)
        flow._model_config_set = True
        dataset = pd.DataFrame({"input": [f"test{i}" for i in range(20)]})

        result = flow.dry_run(
            dataset, sample_size=1, enable_time_estimation=True, max_concurrency=100
        )

        assert result["sample_size"] == 1
        assert result["execution_successful"] is True

    def test_estimate_sample_size_5_triggers_run_with_1(self):
        """Test sample_size=5 with estimation triggers second run with 1."""
        async_block = self.create_mock_llm_block("async_block", async_mode=True)
        flow = Flow(blocks=[async_block], metadata=self.test_metadata)
        flow._model_config_set = True
        dataset = pd.DataFrame({"input": [f"test{i}" for i in range(20)]})

        result = flow.dry_run(
            dataset, sample_size=5, enable_time_estimation=True, max_concurrency=100
        )

        assert result["sample_size"] == 5
        assert result["execution_successful"] is True

    def test_estimate_sample_size_3_runs_canonical_pair(self):
        """Test sample_size=3 with estimation runs both 1 and 5."""
        async_block = self.create_mock_llm_block("async_block", async_mode=True)
        flow = Flow(blocks=[async_block], metadata=self.test_metadata)
        flow._model_config_set = True
        dataset = pd.DataFrame({"input": [f"test{i}" for i in range(20)]})

        result = flow.dry_run(
            dataset, sample_size=3, enable_time_estimation=True, max_concurrency=100
        )

        assert result["sample_size"] == 3
        assert result["execution_successful"] is True

    def test_estimate_sample_size_10_runs_canonical_pair(self):
        """Test sample_size=10 with estimation runs canonical (1, 5) pair."""
        async_block = self.create_mock_llm_block("async_block", async_mode=True)
        flow = Flow(blocks=[async_block], metadata=self.test_metadata)
        flow._model_config_set = True
        dataset = pd.DataFrame({"input": [f"test{i}" for i in range(20)]})

        result = flow.dry_run(
            dataset, sample_size=10, enable_time_estimation=True, max_concurrency=100
        )

        assert result["sample_size"] == 10
        assert result["execution_successful"] is True


class TestTimeEstimatorIntegration:
    """Test integration with time_estimator module."""

    def test_time_estimator_module_functions(self):
        """Test that time_estimator module functions are called correctly."""

        # Test is_llm_using_block - block type detection
        llm_block_info = {
            "block_type": "LLMChatBlock",
            "parameters_used": {"model": "test-model"},
        }
        assert is_llm_using_block(llm_block_info) is True

        # Test is_llm_using_block - parameter-based detection
        # Block with model parameter but non-LLM block type
        llm_by_params_model = {
            "block_type": "CustomBlock",
            "parameters_used": {"model": "test-model"},
        }
        assert is_llm_using_block(llm_by_params_model) is True

        llm_by_params_api_base = {
            "block_type": "CustomBlock",
            "parameters_used": {"api_base": "http://localhost:8000"},
        }
        assert is_llm_using_block(llm_by_params_api_base) is True

        llm_by_params_api_key = {
            "block_type": "CustomBlock",
            "parameters_used": {"api_key": "secret"},
        }
        assert is_llm_using_block(llm_by_params_api_key) is True

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
            ],
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
                ],
            },
            total_dataset_size=10000,
            max_concurrency=100,
        )

        # With correct max() function: time = 10000/1000 = 10 seconds
        # With incorrect min() function: time = 10000/0.1 = 100000 seconds
        # So if estimated time is < 1000 seconds, we're using max() correctly
        assert result["estimated_time_seconds"] < 1000, (
            f"Estimated time {result['estimated_time_seconds']} is too high, "
            "suggesting min() is being used instead of max() for throughput flooring"
        )

    def test_estimate_without_total_dataset_size(self):
        """Test estimate_execution_time without total_dataset_size parameter (coverage for line 274)."""
        # This tests the fallback: total_dataset_size = dry_run_1.get("original_dataset_size", ...)
        dry_run = {
            "sample_size": 5,
            "original_dataset_size": 100,
            "execution_time_seconds": 10.0,
            "blocks_executed": [],
        }

        # Call without total_dataset_size - should use original_dataset_size from dry_run
        result = estimate_execution_time(
            dry_run_1=dry_run, dry_run_2=None, total_dataset_size=None
        )

        # Should scale to original_dataset_size (100)
        assert result["estimated_time_seconds"] > 0
        # Verify it used original_dataset_size: (10/5)*100 = 200, then *1.2 buffer = 240
        assert 230 < result["estimated_time_seconds"] < 250

    def test_estimate_with_mismatched_block_counts(self):
        """Test estimate_execution_time with mismatched dry_run block counts (coverage for line 301)."""
        # dry_run_1 has 2 blocks, dry_run_2 has only 1 block
        dry_run_1 = {
            "sample_size": 1,
            "execution_time_seconds": 2.0,
            "blocks_executed": [
                {
                    "block_name": "block1",
                    "block_type": "LLMChatBlock",
                    "execution_time_seconds": 1.0,
                    "input_rows": 1,
                    "parameters_used": {"model": "test"},
                },
                {
                    "block_name": "block2",
                    "block_type": "LLMChatBlock",
                    "execution_time_seconds": 1.0,
                    "input_rows": 1,
                    "parameters_used": {"model": "test"},
                },
            ],
        }

        dry_run_2 = {
            "sample_size": 5,
            "execution_time_seconds": 5.0,
            "blocks_executed": [
                {
                    "block_name": "block1",
                    "block_type": "LLMChatBlock",
                    "execution_time_seconds": 5.0,
                    "input_rows": 5,
                    "parameters_used": {"model": "test"},
                }
                # block2 is missing!
            ],
        }

        # Should handle gracefully - only process block1, skip block2
        result = estimate_execution_time(
            dry_run_1=dry_run_1, dry_run_2=dry_run_2, total_dataset_size=100
        )

        assert result["estimated_time_seconds"] > 0
        # Should only have 1 block estimate (not 2)
        assert len(result["block_estimates"]) == 1
        assert result["block_estimates"][0]["block"] == "block1"

    def test_estimate_with_non_llm_blocks(self):
        """Test estimate_execution_time skips non-LLM blocks (coverage for line 307)."""
        dry_run_1 = {
            "sample_size": 1,
            "execution_time_seconds": 2.0,
            "blocks_executed": [
                {
                    "block_name": "transform_block",
                    "block_type": "TransformBlock",  # Not an LLM block
                    "execution_time_seconds": 1.0,
                    "input_rows": 1,
                    "parameters_used": {},
                },
                {
                    "block_name": "llm_block",
                    "block_type": "LLMChatBlock",  # LLM block
                    "execution_time_seconds": 1.0,
                    "input_rows": 1,
                    "parameters_used": {"model": "test"},
                },
            ],
        }

        dry_run_2 = {
            "sample_size": 5,
            "execution_time_seconds": 6.0,
            "blocks_executed": [
                {
                    "block_name": "transform_block",
                    "block_type": "TransformBlock",
                    "execution_time_seconds": 1.0,
                    "input_rows": 5,
                    "parameters_used": {},
                },
                {
                    "block_name": "llm_block",
                    "block_type": "LLMChatBlock",
                    "execution_time_seconds": 5.0,
                    "input_rows": 5,
                    "parameters_used": {"model": "test"},
                },
            ],
        }

        # Should only process llm_block, skip transform_block
        result = estimate_execution_time(
            dry_run_1=dry_run_1, dry_run_2=dry_run_2, total_dataset_size=100
        )

        assert result["estimated_time_seconds"] > 0
        # Should only have 1 block estimate (LLM block only)
        assert len(result["block_estimates"]) == 1
        assert result["block_estimates"][0]["block"] == "llm_block"
        assert result["total_estimated_requests"] > 0  # LLM blocks count requests
