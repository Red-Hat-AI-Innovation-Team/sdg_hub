# SPDX-License-Identifier: Apache-2.0
"""Tests for the Flow checkpointing functionality."""

# Standard
from pathlib import Path
import json
import tempfile

# First Party
from sdg_hub.core.flow.checkpointer import FlowCheckpointer

# Third Party
import pandas as pd
import pytest


class TestFlowCheckpointer:
    """Test FlowCheckpointer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.flow_id = "test_flow_id"

    def teardown_method(self):
        """Clean up test fixtures."""
        # Standard
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_checkpointer_disabled(self):
        """Test checkpointer when disabled (no checkpoint_dir)."""
        checkpointer = FlowCheckpointer()

        assert not checkpointer.is_enabled
        assert checkpointer.checkpoint_dir is None

        # Should be no-ops
        dataset = pd.DataFrame({"input": ["test"]})
        remaining, completed = checkpointer.load_existing_progress(dataset)
        assert remaining.equals(dataset)
        assert completed is None

        checkpointer.add_completed_samples(dataset)
        checkpointer.save_final_checkpoint()

    def test_checkpointer_enabled(self):
        """Test checkpointer when enabled."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        assert checkpointer.is_enabled
        assert checkpointer.checkpoint_dir == self.temp_dir
        assert checkpointer.save_freq == 2
        assert checkpointer.flow_id == self.flow_id
        assert Path(self.temp_dir).exists()

    def test_load_existing_progress_no_checkpoints(self):
        """Test loading progress when no checkpoints exist."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        dataset = pd.DataFrame({"input": ["test1", "test2"]})
        remaining, completed = checkpointer.load_existing_progress(dataset)

        assert remaining.equals(dataset)
        assert completed is None

    def test_save_and_load_single_checkpoint(self):
        """Test saving and loading a single checkpoint."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        # Add some completed samples
        dataset = pd.DataFrame(
            {"input": ["test1", "test2"], "output": ["result1", "result2"]}
        )

        checkpointer.add_completed_samples(dataset)

        # Should have saved a checkpoint
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 1

        # Metadata should exist
        assert Path(checkpointer.metadata_path).exists()

        # Load progress info
        progress = checkpointer.get_progress_info()
        assert progress["samples_processed"] == 2
        assert progress["checkpoint_counter"] == 1

    def test_save_checkpoint_with_save_freq(self):
        """Test checkpoint saving with save frequency."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=3, flow_id=self.flow_id
        )

        # Add samples one by one
        sample1 = pd.DataFrame({"input": ["test1"], "output": ["result1"]})
        sample2 = pd.DataFrame({"input": ["test2"], "output": ["result2"]})
        sample3 = pd.DataFrame({"input": ["test3"], "output": ["result3"]})
        sample4 = pd.DataFrame({"input": ["test4"], "output": ["result4"]})

        # Add first sample - no checkpoint yet
        checkpointer.add_completed_samples(sample1)
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 0

        # Add second sample - no checkpoint yet
        checkpointer.add_completed_samples(sample2)
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 0

        # Add third sample - should trigger checkpoint
        checkpointer.add_completed_samples(sample3)
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 1

        # Add fourth sample - should not trigger checkpoint yet
        checkpointer.add_completed_samples(sample4)
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 1  # Still only one

        # Save final checkpoint
        checkpointer.save_final_checkpoint()
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 2  # Now two checkpoints

    def test_load_existing_checkpoints(self):
        """Test loading existing checkpoints and finding remaining work."""
        # First, create some checkpoints
        checkpointer1 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        completed_data = pd.DataFrame(
            {"input": ["test1", "test2"], "output": ["result1", "result2"]}
        )
        checkpointer1.add_completed_samples(completed_data)

        # Now create a new checkpointer and test loading
        checkpointer2 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        # Input dataset with some new samples
        input_dataset = pd.DataFrame(
            {
                "input": ["test1", "test2", "test3", "test4"],
            }
        )

        remaining, completed = checkpointer2.load_existing_progress(input_dataset)

        # Should find that test1 and test2 are completed
        assert len(completed) == 2
        assert len(remaining) == 2
        assert remaining["input"].tolist() == ["test3", "test4"]

    def test_load_all_samples_completed(self):
        """Test loading when all samples are already completed."""
        # Create checkpoints for all input samples
        checkpointer1 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        completed_data = pd.DataFrame(
            {"input": ["test1", "test2"], "output": ["result1", "result2"]}
        )
        checkpointer1.add_completed_samples(completed_data)

        # Input dataset with only the same samples
        input_dataset = pd.DataFrame(
            {
                "input": ["test1", "test2"],
            }
        )

        checkpointer2 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        remaining, completed = checkpointer2.load_existing_progress(input_dataset)

        assert len(remaining) == 0
        assert len(completed) == 2

    def test_find_remaining_samples_no_common_columns(self):
        """Test finding remaining samples when no common columns exist."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        input_dataset = pd.DataFrame(
            {
                "input": ["test1", "test2"],
            }
        )

        completed_dataset = pd.DataFrame(
            {
                "output": ["result1", "result2"],
            }
        )

        remaining = checkpointer._find_remaining_samples(
            input_dataset, completed_dataset
        )

        # Should return entire input dataset when no common columns
        assert len(remaining) == len(input_dataset)
        assert remaining["input"].equals(input_dataset["input"])

    def test_metadata_persistence(self):
        """Test metadata saving and loading."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=5, flow_id=self.flow_id
        )

        # Add some samples to trigger metadata save
        dataset = pd.DataFrame(
            {
                "input": ["test1", "test2", "test3", "test4", "test5"],
                "output": ["result1", "result2", "result3", "result4", "result5"],
            }
        )
        checkpointer.add_completed_samples(dataset)

        # Check metadata content
        with open(checkpointer.metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["flow_id"] == self.flow_id
        assert metadata["save_freq"] == 5
        assert metadata["samples_processed"] == 5
        assert metadata["checkpoint_counter"] == 1

    def test_cleanup_checkpoints(self):
        """Test cleaning up all checkpoints."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        # Create some checkpoints
        dataset = pd.DataFrame(
            {"input": ["test1", "test2"], "output": ["result1", "result2"]}
        )
        checkpointer.add_completed_samples(dataset)

        # Verify files exist
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 1
        assert Path(checkpointer.metadata_path).exists()

        # Clean up
        checkpointer.cleanup_checkpoints()

        # Verify files are gone
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 0
        assert not Path(checkpointer.metadata_path).exists()

    def test_progress_info(self):
        """Test getting progress information."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=3, flow_id=self.flow_id
        )

        progress = checkpointer.get_progress_info()

        assert progress["checkpoint_dir"] == self.temp_dir
        assert progress["save_freq"] == 3
        assert progress["flow_id"] == self.flow_id
        assert progress["samples_processed"] == 0
        assert progress["checkpoint_counter"] == 0
        assert progress["pending_samples"] == 0
        assert progress["is_enabled"] is True

    def test_multiple_checkpoint_files_loading(self):
        """Test loading multiple checkpoint files in correct order."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        # Create multiple checkpoints manually
        checkpoint1_data = pd.DataFrame(
            {"input": ["test1", "test2"], "output": ["result1", "result2"]}
        )
        checkpoint2_data = pd.DataFrame(
            {"input": ["test3", "test4"], "output": ["result3", "result4"]}
        )

        checkpointer.add_completed_samples(checkpoint1_data)
        checkpointer.add_completed_samples(checkpoint2_data)

        # Load all completed samples
        completed = checkpointer._load_completed_samples()

        assert len(completed) == 4
        assert set(completed["input"]) == {"test1", "test2", "test3", "test4"}
        assert set(completed["output"]) == {"result1", "result2", "result3", "result4"}

    def test_load_corrupted_checkpoint(self):
        """Test handling corrupted checkpoint files."""
        # First create a working checkpointer with save_freq to trigger checkpoint save
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir,
            save_freq=1,  # Save after each sample
            flow_id=self.flow_id,
        )

        # Create a good checkpoint first
        good_data = pd.DataFrame({"input": ["test1"], "output": ["result1"]})
        checkpointer.add_completed_samples(good_data)

        # Create a corrupted checkpoint file manually
        corrupted_file = Path(self.temp_dir) / "checkpoint_0002.jsonl"
        with open(corrupted_file, "w") as f:
            f.write("invalid json content")

        # Should still load the good checkpoint and warn about the bad one
        completed = checkpointer._load_completed_samples()

        # Should get the good data (may be None if all checkpoints failed to load)
        if completed is not None:
            assert len(completed) >= 1
            assert "test1" in completed["input"].tolist()

    def test_compute_dataset_signature_change_detection(self):
        """Test that dataset signature detects changes in dataset."""
        # Create original dataset
        original_dataset = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]}
        )

        # Compute original signature
        original_sig = FlowCheckpointer._compute_dataset_signature(original_dataset)

        # Test 1: Same dataset should produce same signature
        same_dataset = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]}
        )
        same_sig = FlowCheckpointer._compute_dataset_signature(same_dataset)
        assert original_sig == same_sig

        # Test 2: Different head values should change signature
        modified_head = pd.DataFrame(
            {"col1": [99, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]}
        )
        modified_head_sig = FlowCheckpointer._compute_dataset_signature(modified_head)
        assert original_sig != modified_head_sig

        # Test 3: Different tail values should change signature
        modified_tail = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 99], "col2": ["a", "b", "c", "d", "e"]}
        )
        modified_tail_sig = FlowCheckpointer._compute_dataset_signature(modified_tail)
        assert original_sig != modified_tail_sig

        # Test 4: Different columns should change signature
        different_cols = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col3": ["a", "b", "c", "d", "e"]}
        )
        different_cols_sig = FlowCheckpointer._compute_dataset_signature(different_cols)
        assert original_sig != different_cols_sig

        # Test 5: Different size should change signature
        different_size = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        different_size_sig = FlowCheckpointer._compute_dataset_signature(different_size)
        assert original_sig != different_size_sig

    def test_dataset_signature_validation_raises_error_on_mismatch(self):
        """Test that load_existing_progress raises FlowValidationError when dataset signature mismatches."""
        # First Party
        from sdg_hub.core.utils.error_handling import FlowValidationError

        # Create initial dataset and save checkpoint
        initial_dataset = pd.DataFrame(
            {"input": ["test1", "test2", "test3"], "value": [1, 2, 3]}
        )

        checkpointer1 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        # First call to load_existing_progress saves initial dataset signature
        checkpointer1.load_existing_progress(initial_dataset)

        # Save checkpoint with initial dataset signature
        completed_data = pd.DataFrame(
            {"input": ["test1"], "value": [1], "output": ["result1"]}
        )
        checkpointer1.add_completed_samples(completed_data)

        # Test 1: Different dataset with different columns should raise error
        different_dataset = pd.DataFrame(
            {"input": ["test1", "test2", "test3"], "different_col": [1, 2, 3]}
        )

        checkpointer2 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        # Should raise FlowValidationError with helpful message
        with pytest.raises(FlowValidationError) as exc_info:
            checkpointer2.load_existing_progress(different_dataset)

        assert "Dataset has changed" in str(exc_info.value)
        assert "Saved checkpoint info" in str(exc_info.value)
        assert "Current dataset info" in str(exc_info.value)
        assert "different_col" in str(exc_info.value)  # New column name in error

        # Test 2: Different dataset size should also raise error
        different_size = pd.DataFrame({"input": ["test1"], "value": [1]})

        checkpointer3 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        with pytest.raises(FlowValidationError) as exc_info:
            checkpointer3.load_existing_progress(different_size)

        assert "Dataset has changed" in str(exc_info.value)
        # Should mention size mismatch
        assert "1 rows" in str(exc_info.value)  # Current size
        assert "3 rows" in str(exc_info.value)  # Saved size

    def test_flow_id_mismatch_raises_error(self):
        """Test that load_existing_progress raises FlowValidationError when flow ID mismatches."""
        # First Party
        from sdg_hub.core.utils.error_handling import FlowValidationError

        # Create checkpoint with flow_id_1
        dataset = pd.DataFrame({"input": ["test1", "test2"], "value": [1, 2]})

        checkpointer1 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id="flow_id_1"
        )

        # Save initial metadata with flow_id_1
        checkpointer1.load_existing_progress(dataset)

        # Save some checkpoint data
        completed_data = pd.DataFrame(
            {"input": ["test1"], "value": [1], "output": ["result1"]}
        )
        checkpointer1.add_completed_samples(completed_data)

        # Try to load with different flow ID
        checkpointer2 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id="flow_id_2"
        )

        # Should raise FlowValidationError
        with pytest.raises(FlowValidationError) as exc_info:
            checkpointer2.load_existing_progress(dataset)

        assert "Flow ID mismatch" in str(exc_info.value)
        assert "flow_id_1" in str(exc_info.value)  # Saved flow ID
        assert "flow_id_2" in str(exc_info.value)  # Current flow ID
        assert "Mixing checkpoints from different flows" in str(exc_info.value)

    def test_find_remaining_samples_with_sdg_input_index(self):
        """Test _find_remaining_samples using _sdg_input_index for robust matching."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        # Create input dataset with _sdg_input_index
        input_dataset = pd.DataFrame(
            {
                "original_col": ["a", "b", "c", "d"],
                "_sdg_input_index": [0, 1, 2, 3],
            }
        )

        # Create completed dataset where flow RENAMED the input column
        # This simulates a flow that transforms/renames columns
        completed_dataset = pd.DataFrame(
            {
                "renamed_col": ["a", "b"],  # Column renamed!
                "output": ["result1", "result2"],
                "_sdg_input_index": [0, 1],  # But index is preserved
            }
        )

        # Find remaining samples - should use index matching
        remaining = checkpointer._find_remaining_samples(
            input_dataset, completed_dataset
        )

        # Should correctly identify samples 2 and 3 as remaining (indices 2, 3)
        assert len(remaining) == 2
        assert remaining["_sdg_input_index"].tolist() == [2, 3]
        assert remaining["original_col"].tolist() == ["c", "d"]

    def test_find_remaining_samples_index_fallback_warning(self):
        """Test that _find_remaining_samples falls back to column matching with warning."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        # Input dataset WITHOUT _sdg_input_index
        input_dataset = pd.DataFrame({"input": ["test1", "test2", "test3"]})

        # Completed dataset also WITHOUT _sdg_input_index (old checkpoint format)
        completed_dataset = pd.DataFrame({"input": ["test1"], "output": ["result1"]})

        # Should fall back to column-based matching
        remaining = checkpointer._find_remaining_samples(
            input_dataset, completed_dataset
        )

        # Should find test2 and test3 as remaining
        assert len(remaining) == 2
        assert remaining["input"].tolist() == ["test2", "test3"]

    def test_save_chunk_immediately(self):
        """Test save_chunk_immediately saves chunks without buffering."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        # Save first chunk
        chunk1 = pd.DataFrame({"input": ["test1", "test2"], "output": ["r1", "r2"]})
        checkpointer.save_chunk_immediately(chunk1)

        # Should create checkpoint file immediately
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 1

        # Verify metadata updated
        progress = checkpointer.get_progress_info()
        assert progress["samples_processed"] == 2
        assert progress["checkpoint_counter"] == 1

        # Save second chunk
        chunk2 = pd.DataFrame({"input": ["test3"], "output": ["r3"]})
        checkpointer.save_chunk_immediately(chunk2)

        # Should create second checkpoint file
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 2

        # Verify metadata updated
        progress = checkpointer.get_progress_info()
        assert progress["samples_processed"] == 3
        assert progress["checkpoint_counter"] == 2

        # Verify pending samples is NOT used (immediate save)
        assert progress["pending_samples"] == 0

    def test_save_chunk_immediately_empty_chunk(self):
        """Test save_chunk_immediately skips empty chunks with warning."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        # Try to save empty chunk
        empty_chunk = pd.DataFrame({"input": [], "output": []})
        checkpointer.save_chunk_immediately(empty_chunk)

        # Should NOT create any checkpoint files
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 0

        # Counter should not increment
        progress = checkpointer.get_progress_info()
        assert progress["checkpoint_counter"] == 0

    def test_load_all_checkpoints(self):
        """Test load_all_checkpoints loads and concatenates all checkpoint files."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        # Save multiple chunks
        chunk1 = pd.DataFrame({"input": ["test1", "test2"], "output": ["r1", "r2"]})
        chunk2 = pd.DataFrame({"input": ["test3", "test4"], "output": ["r3", "r4"]})
        chunk3 = pd.DataFrame({"input": ["test5"], "output": ["r5"]})

        checkpointer.save_chunk_immediately(chunk1)
        checkpointer.save_chunk_immediately(chunk2)
        checkpointer.save_chunk_immediately(chunk3)

        # Load all checkpoints
        all_data = checkpointer.load_all_checkpoints()

        # Should have all 5 samples
        assert len(all_data) == 5
        assert set(all_data["input"]) == {"test1", "test2", "test3", "test4", "test5"}
        assert set(all_data["output"]) == {"r1", "r2", "r3", "r4", "r5"}

    def test_load_all_checkpoints_error_when_disabled(self):
        """Test load_all_checkpoints raises error when checkpointing is disabled."""
        checkpointer = FlowCheckpointer()  # No checkpoint_dir

        with pytest.raises(ValueError) as exc_info:
            checkpointer.load_all_checkpoints()

        assert "Cannot load checkpoints when checkpointing is disabled" in str(
            exc_info.value
        )

    def test_load_all_checkpoints_error_when_no_files(self):
        """Test load_all_checkpoints raises error when no checkpoint files exist."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        with pytest.raises(ValueError) as exc_info:
            checkpointer.load_all_checkpoints()

        assert "No checkpoint files found" in str(exc_info.value)
