# Standard
from unittest.mock import MagicMock, patch

# Third Party
# First Party
from sdg_hub.core.utils.flow_metrics import (
    aggregate_block_metrics,
    display_metrics_summary,
    display_time_estimation_summary,
    save_metrics_to_json,
)
import pandas as pd


class TestAggregateBlockMetrics:
    """Tests for aggregate_block_metrics function."""

    def test_aggregate_single_block(self):
        """Test aggregating metrics for a single block."""
        entries = [
            {
                "block_name": "test_block",
                "block_type": "TestType",
                "execution_time": 1.5,
                "input_rows": 10,
                "output_rows": 10,
                "added_cols": ["col1"],
                "removed_cols": [],
                "status": "success",
            }
        ]

        result = aggregate_block_metrics(entries)

        assert len(result) == 1
        assert result[0]["block_name"] == "test_block"
        assert result[0]["execution_time"] == 1.5
        assert sorted(result[0]["added_cols"]) == ["col1"]

    def test_aggregate_multiple_runs_same_block(self):
        """Test aggregating multiple runs of the same block (chunked execution)."""
        entries = [
            {
                "block_name": "test_block",
                "block_type": "TestType",
                "execution_time": 1.0,
                "input_rows": 5,
                "output_rows": 5,
                "added_cols": ["col1"],
                "removed_cols": [],
                "status": "success",
            },
            {
                "block_name": "test_block",
                "block_type": "TestType",
                "execution_time": 2.0,
                "input_rows": 5,
                "output_rows": 5,
                "added_cols": ["col2"],
                "removed_cols": [],
                "status": "success",
            },
        ]

        result = aggregate_block_metrics(entries)

        assert len(result) == 1
        assert result[0]["execution_time"] == 3.0
        assert result[0]["input_rows"] == 10
        assert result[0]["output_rows"] == 10
        assert sorted(result[0]["added_cols"]) == ["col1", "col2"]

    def test_aggregate_failed_block(self):
        """Test aggregating metrics with failed status."""
        entries = [
            {
                "block_name": "test_block",
                "block_type": "TestType",
                "execution_time": 1.0,
                "input_rows": 10,
                "output_rows": 0,
                "added_cols": [],
                "removed_cols": [],
                "status": "failed",
                "error": "Test error",
            }
        ]

        result = aggregate_block_metrics(entries)

        assert len(result) == 1
        assert result[0]["status"] == "failed"
        assert result[0]["error"] == "Test error"


class TestDisplayMetricsSummary:
    """Tests for display_metrics_summary function."""

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_with_no_metrics(self, mock_console):
        """Test display when no metrics are provided."""
        display_metrics_summary([], "Test Flow")

        # Console should not be instantiated if no metrics
        mock_console.assert_not_called()

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_with_successful_blocks(self, mock_console):
        """Test display with successful block execution."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        metrics = [
            {
                "block_name": "test_block",
                "block_type": "TestType",
                "execution_time": 1.5,
                "input_rows": 10,
                "output_rows": 10,
                "added_cols": ["col1"],
                "removed_cols": [],
                "status": "success",
            }
        ]

        dataset = pd.DataFrame({"col1": list(range(10))})
        display_metrics_summary(metrics, "Test Flow", dataset)

        # Verify console.print was called
        assert mock_console_instance.print.call_count >= 2

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_with_failed_blocks(self, mock_console):
        """Test display with failed block execution."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        metrics = [
            {
                "block_name": "test_block",
                "block_type": "TestType",
                "execution_time": 1.0,
                "input_rows": 10,
                "output_rows": 0,
                "added_cols": [],
                "removed_cols": [],
                "status": "failed",
                "error": "Test error",
            }
        ]

        display_metrics_summary(metrics, "Test Flow", None)

        # Verify console.print was called
        assert mock_console_instance.print.call_count >= 2

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_with_added_and_removed_columns(self, mock_console):
        """Test display with blocks that both add and remove columns."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        metrics = [
            {
                "block_name": "test_block",
                "block_type": "TestType",
                "execution_time": 1.5,
                "input_rows": 10,
                "output_rows": 10,
                "added_cols": ["col1", "col2"],
                "removed_cols": ["old_col"],
                "status": "success",
            }
        ]

        dataset = pd.DataFrame({"col1": list(range(10)), "col2": list(range(10))})
        display_metrics_summary(metrics, "Test Flow", dataset)

        # Verify console.print was called
        assert mock_console_instance.print.call_count >= 2

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_with_partial_completion(self, mock_console):
        """Test display with some blocks succeeded and some failed."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        metrics = [
            {
                "block_name": "block1",
                "block_type": "TestType",
                "execution_time": 1.0,
                "input_rows": 10,
                "output_rows": 10,
                "added_cols": ["col1"],
                "removed_cols": [],
                "status": "success",
            },
            {
                "block_name": "block2",
                "block_type": "TestType",
                "execution_time": 0.5,
                "input_rows": 10,
                "output_rows": 0,
                "added_cols": [],
                "removed_cols": [],
                "status": "failed",
                "error": "Test error",
            },
        ]

        # Partial completion - dataset exists but some blocks failed
        dataset = pd.DataFrame({"col1": list(range(10))})
        display_metrics_summary(metrics, "Test Flow", dataset)

        # Verify console.print was called
        assert mock_console_instance.print.call_count >= 2


class TestDisplayTimeEstimationSummary:
    """Tests for display_time_estimation_summary function."""

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_time_under_60_seconds(self, mock_console):
        """Test display when estimated time is under 60 seconds."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        time_estimation = {
            "estimated_time_seconds": 45.5,
            "total_estimated_requests": 0,
        }

        display_time_estimation_summary(time_estimation, 100)

        # Verify console was used
        assert mock_console_instance.print.call_count >= 2

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_time_between_60_and_3600_seconds(self, mock_console):
        """Test display when estimated time is between 1 minute and 1 hour."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        time_estimation = {
            "estimated_time_seconds": 1800,  # 30 minutes
            "total_estimated_requests": 0,
        }

        display_time_estimation_summary(time_estimation, 100)

        # Verify console was used
        assert mock_console_instance.print.call_count >= 2

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_time_over_3600_seconds(self, mock_console):
        """Test display when estimated time is over 1 hour."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        time_estimation = {
            "estimated_time_seconds": 7200,  # 2 hours
            "total_estimated_requests": 0,
        }

        display_time_estimation_summary(time_estimation, 100)

        # Verify console was used
        assert mock_console_instance.print.call_count >= 2

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_with_requests_per_sample(self, mock_console):
        """Test display includes requests per sample when requests > 0."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        time_estimation = {
            "estimated_time_seconds": 300,
            "total_estimated_requests": 1000,
        }

        display_time_estimation_summary(time_estimation, 100)

        # Verify console was used
        assert mock_console_instance.print.call_count >= 2

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_without_requests(self, mock_console):
        """Test display when total_estimated_requests is 0."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        time_estimation = {
            "estimated_time_seconds": 300,
            "total_estimated_requests": 0,
        }

        display_time_estimation_summary(time_estimation, 100)

        # Verify console was used
        assert mock_console_instance.print.call_count >= 2

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_with_max_concurrency(self, mock_console):
        """Test display includes max concurrency when provided."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        time_estimation = {
            "estimated_time_seconds": 300,
            "total_estimated_requests": 1000,
        }

        display_time_estimation_summary(time_estimation, 100, max_concurrency=50)

        # Verify console was used
        assert mock_console_instance.print.call_count >= 2

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_with_block_estimates_under_60_seconds(self, mock_console):
        """Test display with per-block estimates (time < 60 seconds)."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        time_estimation = {
            "estimated_time_seconds": 300,
            "total_estimated_requests": 1000,
            "block_estimates": [
                {
                    "block": "block1",
                    "estimated_time": 45.5,
                    "estimated_requests": 500,
                    "throughput": 11.0,
                    "amplification": 1.0,
                }
            ],
        }

        display_time_estimation_summary(time_estimation, 100)

        # Verify console was used, should include block table
        assert mock_console_instance.print.call_count >= 3

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_with_block_estimates_over_60_seconds(self, mock_console):
        """Test display with per-block estimates (time > 60 seconds)."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        time_estimation = {
            "estimated_time_seconds": 300,
            "total_estimated_requests": 1000,
            "block_estimates": [
                {
                    "block": "block1",
                    "estimated_time": 120.0,
                    "estimated_requests": 500,
                    "throughput": 4.17,
                    "amplification": 1.0,
                },
                {
                    "block": "block2",
                    "estimated_time": 180.0,
                    "estimated_requests": 500,
                    "throughput": 2.78,
                    "amplification": 1.0,
                },
            ],
        }

        display_time_estimation_summary(time_estimation, 100)

        # Verify console was used, should include block table
        assert mock_console_instance.print.call_count >= 3

    @patch("sdg_hub.core.utils.flow_metrics.Console")
    def test_display_without_block_estimates(self, mock_console):
        """Test display without per-block estimates."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        time_estimation = {
            "estimated_time_seconds": 300,
            "total_estimated_requests": 1000,
            "block_estimates": [],
        }

        display_time_estimation_summary(time_estimation, 100)

        # Verify console was used, but no extra prints for block table
        assert mock_console_instance.print.call_count >= 2


class TestSaveMetricsToJson:
    """Tests for save_metrics_to_json function."""

    @patch("sdg_hub.core.utils.flow_metrics.Path")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("sdg_hub.core.utils.flow_metrics.time.perf_counter")
    def test_save_metrics_success(self, mock_perf_counter, mock_open, mock_path):
        """Test successful metrics save to JSON."""
        mock_perf_counter.return_value = 100.0
        mock_logger = MagicMock()

        # Mock Path operations
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__ = MagicMock(return_value=mock_path_instance)
        mock_path_instance.parent = MagicMock()

        metrics = [
            {
                "block_name": "test_block",
                "block_type": "TestType",
                "execution_time": 1.5,
                "input_rows": 10,
                "output_rows": 10,
                "added_cols": ["col1"],
                "removed_cols": [],
                "status": "success",
            }
        ]

        save_metrics_to_json(
            metrics,
            "Test Flow",
            "1.0.0",
            True,
            90.0,
            "/tmp/logs",
            logger=mock_logger,
        )

        # Verify logger was called with success message
        assert mock_logger.info.called

    @patch("sdg_hub.core.utils.flow_metrics.Path")
    @patch("builtins.open", side_effect=Exception("Write error"))
    @patch("sdg_hub.core.utils.flow_metrics.time.perf_counter")
    def test_save_metrics_failure(self, mock_perf_counter, mock_open, mock_path):
        """Test metrics save handles exceptions gracefully."""
        mock_perf_counter.return_value = 100.0
        mock_logger = MagicMock()

        # Mock Path operations
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__ = MagicMock(return_value=mock_path_instance)
        mock_path_instance.parent = MagicMock()

        metrics = [
            {
                "block_name": "test_block",
                "block_type": "TestType",
                "execution_time": 1.5,
                "input_rows": 10,
                "output_rows": 10,
                "added_cols": [],
                "removed_cols": [],
                "status": "success",
            }
        ]

        # Should not raise exception even when file write fails
        save_metrics_to_json(
            metrics,
            "Test Flow",
            "1.0.0",
            True,
            90.0,
            "/tmp/logs",
            logger=mock_logger,
        )

        # Verify logger was called with warning
        assert mock_logger.warning.called
