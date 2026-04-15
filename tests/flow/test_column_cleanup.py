# SPDX-License-Identifier: Apache-2.0
"""Tests for column cleanup feature."""

# First Party
# Third Party
import pandas as pd
import pytest

from sdg_hub.core.blocks.transform.duplicate_columns import DuplicateColumnsBlock
from sdg_hub.core.blocks.transform.text_concat import TextConcatBlock
from sdg_hub.core.flow.base import Flow
from sdg_hub.core.flow.column_tracker import ColumnDependencyTracker
from sdg_hub.core.flow.metadata import FlowMetadata


class TestFlowMetadataColumnCleanup:
    """Tests for FlowMetadata output_columns field."""

    def test_output_columns_default_none(self):
        """Test that output_columns defaults to None."""
        metadata = FlowMetadata(name="test")
        assert metadata.output_columns is None

    def test_output_columns_set(self):
        """Test setting output_columns."""
        metadata = FlowMetadata(
            name="test",
            output_columns=["col1", "col2"],
        )
        assert metadata.output_columns == ["col1", "col2"]

    def test_output_columns_strips_whitespace(self):
        """Test that output_columns strips whitespace."""
        metadata = FlowMetadata(
            name="test",
            output_columns=["  col1  ", "col2  "],
        )
        assert metadata.output_columns == ["col1", "col2"]

    def test_output_columns_rejects_duplicates(self):
        """Test that duplicate column names are rejected."""
        with pytest.raises(ValueError, match="duplicate"):
            FlowMetadata(
                name="test",
                output_columns=["col1", "col1"],
            )

    def test_output_columns_empty_list_rejected(self):
        """Test that empty list is rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            FlowMetadata(
                name="test",
                output_columns=[],
            )


class TestColumnDependencyTracker:
    """Tests for ColumnDependencyTracker."""

    def test_build_dependency_graph_simple(self):
        """Test building dependency graph with simple blocks."""
        block1 = TextConcatBlock(
            block_name="b1", input_cols=["a", "b"], output_cols="ab"
        )
        block2 = TextConcatBlock(
            block_name="b2", input_cols=["ab", "c"], output_cols="abc"
        )

        tracker = ColumnDependencyTracker(
            blocks=[block1, block2],
            columns_to_keep={"abc"},
            original_columns={"a", "b", "c"},
        )

        assert tracker._last_consumer["a"] == 0
        assert tracker._last_consumer["b"] == 0
        assert tracker._last_consumer["ab"] == 1
        assert tracker._last_consumer["c"] == 1

    def test_extract_input_columns_string(self):
        """Test extracting input columns from string."""
        result = ColumnDependencyTracker._extract_input_columns("col1")
        assert result == ["col1"]

    def test_extract_input_columns_list(self):
        """Test extracting input columns from list."""
        result = ColumnDependencyTracker._extract_input_columns(["col1", "col2"])
        assert result == ["col1", "col2"]

    def test_extract_input_columns_dict(self):
        """Test extracting input columns from dict (keys are source cols)."""
        result = ColumnDependencyTracker._extract_input_columns({"src": "dst"})
        assert result == ["src"]

    def test_extract_input_columns_none(self):
        """Test extracting input columns from None."""
        result = ColumnDependencyTracker._extract_input_columns(None)
        assert result == []

    def test_get_droppable_columns_preserves_final(self):
        """Test that final output columns are never dropped."""
        block1 = TextConcatBlock(block_name="b1", input_cols=["a"], output_cols="final")

        tracker = ColumnDependencyTracker(
            blocks=[block1],
            columns_to_keep={"final"},
            original_columns={"a"},
        )

        droppable = tracker.get_droppable_columns(0, {"a", "final"})
        assert "final" not in droppable
        assert "a" not in droppable  # original preserved

    def test_get_droppable_columns_preserves_original(self):
        """Test that original columns are never dropped."""
        block1 = TextConcatBlock(block_name="b1", input_cols=["a"], output_cols="temp")
        # block2 consumes temp, so temp has a known last consumer (block2)
        block2 = DuplicateColumnsBlock(block_name="b2", input_cols={"temp": "output"})

        tracker = ColumnDependencyTracker(
            blocks=[block1, block2],
            columns_to_keep={"output"},
            original_columns={"a"},
        )

        # After block2 (index 1), temp's last consumer is satisfied
        droppable = tracker.get_droppable_columns(1, {"a", "temp", "output"})
        assert "a" not in droppable
        assert "temp" in droppable

    def test_get_droppable_columns_after_last_consumer(self):
        """Test dropping columns after their last consumer."""
        block1 = TextConcatBlock(
            block_name="b1", input_cols=["a", "b"], output_cols="ab"
        )
        block2 = TextConcatBlock(
            block_name="b2", input_cols=["ab"], output_cols="final"
        )

        tracker = ColumnDependencyTracker(
            blocks=[block1, block2],
            columns_to_keep={"final"},
            original_columns={"a", "b"},
        )

        # After block 0: 'ab' still needed by block 1
        droppable0 = tracker.get_droppable_columns(0, {"a", "b", "ab"})
        assert "ab" not in droppable0

        # After block 1: 'ab' no longer needed
        droppable1 = tracker.get_droppable_columns(1, {"a", "b", "ab", "final"})
        assert "ab" in droppable1


class TestColumnDependencyTrackerEdgeCases:
    """Edge case tests for ColumnDependencyTracker."""

    def test_empty_blocks_list(self):
        """Test tracker with empty blocks list."""
        tracker = ColumnDependencyTracker(
            blocks=[],
            columns_to_keep={"output"},
            original_columns={"input"},
        )
        assert tracker._last_consumer == {}
        droppable = tracker.get_droppable_columns(0, {"input", "output", "temp"})
        # 'temp' has no known consumer, so it's deferred to final cleanup
        assert "temp" not in droppable
        assert "input" not in droppable
        assert "output" not in droppable

    def test_get_droppable_columns_never_consumed(self):
        """Test that columns never consumed are deferred to final cleanup."""
        block1 = TextConcatBlock(
            block_name="b1", input_cols=["a"], output_cols="unused"
        )
        block2 = TextConcatBlock(block_name="b2", input_cols=["a"], output_cols="final")

        tracker = ColumnDependencyTracker(
            blocks=[block1, block2],
            columns_to_keep={"final"},
            original_columns={"a"},
        )

        # 'unused' is never consumed by any block — conservative approach
        # defers it to _cleanup_final_columns rather than dropping early
        droppable = tracker.get_droppable_columns(0, {"a", "unused"})
        assert "unused" not in droppable


class TestFlowColumnCleanup:
    """Tests for Flow column cleanup during execution."""

    def test_flow_without_output_columns_keeps_all(self):
        """Test that without output_columns, all columns are kept."""
        flow = Flow(
            metadata=FlowMetadata(name="test"),
            blocks=[
                TextConcatBlock(
                    block_name="concat",
                    input_cols=["a", "b"],
                    output_cols="ab",
                ),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        result = flow.generate(dataset)

        assert "a" in result.columns
        assert "b" in result.columns
        assert "ab" in result.columns

    def test_flow_with_output_columns_drops_intermediate(self):
        """Test that intermediate columns are dropped."""
        flow = Flow(
            metadata=FlowMetadata(
                name="test",
                output_columns=["final"],
            ),
            blocks=[
                TextConcatBlock(
                    block_name="step1",
                    input_cols=["a", "b"],
                    output_cols="intermediate",
                ),
                DuplicateColumnsBlock(
                    block_name="step2",
                    input_cols={"intermediate": "final"},
                ),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        result = flow.generate(dataset)

        # Original columns preserved
        assert "a" in result.columns
        assert "b" in result.columns
        # Final output kept
        assert "final" in result.columns
        # Intermediate dropped
        assert "intermediate" not in result.columns

    def test_flow_drops_columns_early_during_execution(self):
        """Test that columns are dropped early as they become unused."""
        flow = Flow(
            metadata=FlowMetadata(
                name="test",
                output_columns=["final"],
            ),
            blocks=[
                TextConcatBlock(
                    block_name="step1",
                    input_cols=["a", "b"],
                    output_cols="temp1",
                ),
                TextConcatBlock(
                    block_name="step2",
                    input_cols=["temp1", "c"],
                    output_cols="temp2",
                ),
                DuplicateColumnsBlock(
                    block_name="step3",
                    input_cols={"temp2": "final"},
                ),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"], "c": ["3"]})
        result = flow.generate(dataset)

        # Original columns preserved
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns
        # Final output kept
        assert "final" in result.columns
        # All intermediates dropped
        assert "temp1" not in result.columns
        assert "temp2" not in result.columns

    def test_flow_missing_output_columns_raises(self):
        """Test error when output_columns specifies missing columns."""
        from sdg_hub.core.utils.error_handling import FlowValidationError

        flow = Flow(
            metadata=FlowMetadata(
                name="test",
                output_columns=["nonexistent", "output"],
            ),
            blocks=[
                TextConcatBlock(
                    block_name="concat",
                    input_cols=["a", "b"],
                    output_cols="output",
                ),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        with pytest.raises(FlowValidationError, match="output_columns not found"):
            flow.generate(dataset)

    def test_flow_early_drop_verified_mid_execution(self):
        """Test that columns are actually dropped mid-execution, not just at the end.

        Uses a block that records which columns exist when it runs, proving
        that early dropping happened before the final cleanup.
        """
        from sdg_hub.core.blocks.base import BaseBlock
        from sdg_hub.core.blocks.registry import BlockRegistry

        captured_columns: list[set[str]] = []

        @BlockRegistry.register("ColumnCapture", "test", "Captures columns for testing")
        class ColumnCaptureBlock(BaseBlock):
            """Block that records available columns during execution."""

            def generate(self, samples, **kwargs):
                captured_columns.append(set(samples.columns))
                return samples

        flow = Flow(
            metadata=FlowMetadata(
                name="test",
                output_columns=["final"],
            ),
            blocks=[
                TextConcatBlock(
                    block_name="step1",
                    input_cols=["a", "b"],
                    output_cols="temp",
                ),
                # temp's last consumer is step2 (index 1), so it should be
                # dropped after step2 completes, before step3 runs
                DuplicateColumnsBlock(
                    block_name="step2",
                    input_cols={"temp": "final"},
                ),
                ColumnCaptureBlock(block_name="capture"),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        flow.generate(dataset)

        # The capture block should see that 'temp' was already dropped
        assert len(captured_columns) == 1
        assert "temp" not in captured_columns[0]
        assert "final" in captured_columns[0]


class TestCleanupFinalColumns:
    """Unit tests for _cleanup_final_columns function."""

    def test_drops_intermediate_columns(self):
        """Test that intermediate columns are dropped."""
        from sdg_hub.core.flow.execution import _cleanup_final_columns

        dataset = pd.DataFrame({"a": ["1"], "intermediate": ["2"], "output": ["3"]})
        result = _cleanup_final_columns(
            dataset, {"output"}, {"a"}, __import__("logging").getLogger()
        )
        assert list(result.columns) == ["a", "output"]

    def test_preserves_original_and_output(self):
        """Test that original and output columns are all preserved."""
        from sdg_hub.core.flow.execution import _cleanup_final_columns

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"], "output": ["3"]})
        result = _cleanup_final_columns(
            dataset, {"output"}, {"a", "b"}, __import__("logging").getLogger()
        )
        assert set(result.columns) == {"a", "b", "output"}

    def test_nothing_to_drop(self):
        """Test when all columns should be kept."""
        from sdg_hub.core.flow.execution import _cleanup_final_columns

        dataset = pd.DataFrame({"a": ["1"], "output": ["2"]})
        result = _cleanup_final_columns(
            dataset, {"output"}, {"a"}, __import__("logging").getLogger()
        )
        assert set(result.columns) == {"a", "output"}

    def test_missing_output_columns_raises(self):
        """Test that missing output columns raise FlowValidationError."""
        from sdg_hub.core.flow.execution import _cleanup_final_columns
        from sdg_hub.core.utils.error_handling import FlowValidationError

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        with pytest.raises(FlowValidationError, match="output_columns not found"):
            _cleanup_final_columns(
                dataset, {"missing_col"}, {"a"}, __import__("logging").getLogger()
            )


class TestCheckpointColumnCleanup:
    """Tests for column cleanup on checkpoint early-return path."""

    def test_checkpoint_early_return_drops_intermediate_columns(self, tmp_path):
        """Test that column cleanup applies when all samples are already checkpointed."""
        from sdg_hub.core.flow.checkpointer import FlowCheckpointer

        flow = Flow(
            metadata=FlowMetadata(
                name="test",
                output_columns=["output"],
            ),
            blocks=[
                TextConcatBlock(
                    block_name="concat",
                    input_cols=["a", "b"],
                    output_cols="output",
                ),
            ],
        )

        # Simulate a prior run that checkpointed all samples with extra columns
        checkpoint_dir = str(tmp_path / "checkpoints")
        checkpointer = FlowCheckpointer(
            checkpoint_dir=checkpoint_dir, flow_id=flow.metadata.id
        )
        completed = pd.DataFrame(
            {"a": ["1"], "b": ["2"], "output": ["12"], "intermediate": ["extra"]}
        )
        checkpointer.add_completed_samples(completed)
        checkpointer.save_final_checkpoint()

        # Now run with same input — should early-return and still apply cleanup
        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        result = flow.generate(dataset, checkpoint_dir=checkpoint_dir)

        # Original columns preserved, output kept, intermediate dropped
        assert "a" in result.columns
        assert "b" in result.columns
        assert "output" in result.columns
        assert "intermediate" not in result.columns
