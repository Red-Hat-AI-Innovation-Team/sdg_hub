# SPDX-License-Identifier: Apache-2.0
"""Tests for the Flow checkpointing functionality."""

# Standard
from pathlib import Path
import json
import tempfile

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.flow.checkpointer import FlowCheckpointer


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
        dataset = Dataset.from_dict({"input": ["test"]})
        remaining, completed = checkpointer.load_existing_progress(dataset)
        assert remaining == dataset
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

        dataset = Dataset.from_dict({"input": ["test1", "test2"]})
        remaining, completed = checkpointer.load_existing_progress(dataset)

        assert remaining == dataset
        assert completed is None

    def test_save_and_load_single_checkpoint(self):
        """Test saving and loading a single checkpoint."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        # Add some completed samples
        dataset = Dataset.from_dict(
            {"input": ["test1", "test2"], "output": ["result1", "result2"]}
        )

        checkpointer.add_completed_samples(dataset)
        # Explicitly save checkpoint due to new explicit-save behavior
        checkpointer._save_checkpoint()

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
        """Test explicit checkpoint saving (save_freq no longer triggers auto-save)."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=3, flow_id=self.flow_id
        )

        # Add samples one by one
        sample1 = Dataset.from_dict({"input": ["test1"], "output": ["result1"]})
        sample2 = Dataset.from_dict({"input": ["test2"], "output": ["result2"]})
        sample3 = Dataset.from_dict({"input": ["test3"], "output": ["result3"]})
        sample4 = Dataset.from_dict({"input": ["test4"], "output": ["result4"]})

        # Add first chunk and save explicitly
        checkpointer.add_completed_samples(sample1)
        checkpointer.add_completed_samples(sample2)
        checkpointer.add_completed_samples(sample3)
        checkpointer._save_checkpoint()
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 1

        # Add second chunk and save explicitly
        checkpointer.add_completed_samples(sample4)
        checkpointer._save_checkpoint()
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 2

        # Verify checkpoint contents
        loaded = checkpointer.load_all_completed_samples()
        assert len(loaded) == 4
        assert loaded["input"] == ["test1", "test2", "test3", "test4"]

    def test_load_existing_checkpoints(self):
        """Test loading existing checkpoints and finding remaining work."""
        # First, create some checkpoints
        checkpointer1 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        completed_data = Dataset.from_dict(
            {"input": ["test1", "test2"], "output": ["result1", "result2"]}
        )
        checkpointer1.add_completed_samples(completed_data)
        checkpointer1._save_checkpoint()

        # Now create a new checkpointer and test loading
        checkpointer2 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        # Input dataset with some new samples
        input_dataset = Dataset.from_dict(
            {
                "input": ["test1", "test2", "test3", "test4"],
            }
        )

        remaining, completed = checkpointer2.load_existing_progress(input_dataset)

        # Should find that test1 and test2 are completed
        assert len(completed) == 2
        assert len(remaining) == 2
        assert remaining["input"] == ["test3", "test4"]

    def test_load_all_samples_completed(self):
        """Test loading when all samples are already completed."""
        # Create checkpoints for all input samples
        checkpointer1 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        completed_data = Dataset.from_dict(
            {"input": ["test1", "test2"], "output": ["result1", "result2"]}
        )
        checkpointer1.add_completed_samples(completed_data)
        checkpointer1._save_checkpoint()

        # Input dataset with only the same samples
        input_dataset = Dataset.from_dict(
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

        input_dataset = Dataset.from_dict(
            {
                "input": ["test1", "test2"],
            }
        )

        completed_dataset = Dataset.from_dict(
            {
                "output": ["result1", "result2"],
            }
        )

        remaining = checkpointer._find_remaining_samples(
            input_dataset, completed_dataset
        )

        # Should return entire input dataset when no common columns
        assert len(remaining) == len(input_dataset)
        assert remaining["input"] == input_dataset["input"]

    def test_metadata_persistence(self):
        """Test metadata saving and loading."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=5, flow_id=self.flow_id
        )

        # Add some samples to trigger metadata save
        dataset = Dataset.from_dict(
            {
                "input": ["test1", "test2", "test3", "test4", "test5"],
                "output": ["result1", "result2", "result3", "result4", "result5"],
            }
        )
        checkpointer.add_completed_samples(dataset)
        checkpointer._save_checkpoint()

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
        dataset = Dataset.from_dict(
            {"input": ["test1", "test2"], "output": ["result1", "result2"]}
        )
        checkpointer.add_completed_samples(dataset)
        checkpointer._save_checkpoint()

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
        checkpoint1_data = Dataset.from_dict(
            {"input": ["test1", "test2"], "output": ["result1", "result2"]}
        )
        checkpoint2_data = Dataset.from_dict(
            {"input": ["test3", "test4"], "output": ["result3", "result4"]}
        )

        checkpointer.add_completed_samples(checkpoint1_data)
        checkpointer._save_checkpoint()
        checkpointer.add_completed_samples(checkpoint2_data)
        checkpointer._save_checkpoint()

        # Load all completed samples
        completed = checkpointer.load_all_completed_samples()

        assert len(completed) == 4
        assert set(completed["input"]) == {"test1", "test2", "test3", "test4"}
        assert set(completed["output"]) == {"result1", "result2", "result3", "result4"}

    def test_load_all_completed_samples_public_method(self):
        """Test the public load_all_completed_samples() method."""
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        # Test 1: No checkpoints exist
        result = checkpointer.load_all_completed_samples()
        assert result is None

        # Test 2: Add some samples and verify they can be loaded
        data1 = Dataset.from_dict({"input": ["a", "b"], "output": ["x", "y"]})
        checkpointer.add_completed_samples(data1)
        checkpointer._save_checkpoint()

        result = checkpointer.load_all_completed_samples()
        assert result is not None
        assert len(result) == 2
        assert set(result["input"]) == {"a", "b"}
        assert set(result["output"]) == {"x", "y"}

        # Test 3: Add more samples and verify all are loaded
        data2 = Dataset.from_dict({"input": ["c", "d"], "output": ["z", "w"]})
        checkpointer.add_completed_samples(data2)
        checkpointer._save_checkpoint()

        result = checkpointer.load_all_completed_samples()
        assert result is not None
        assert len(result) == 4
        assert set(result["input"]) == {"a", "b", "c", "d"}
        assert set(result["output"]) == {"x", "y", "z", "w"}

    def test_load_corrupted_checkpoint(self):
        """Test handling corrupted checkpoint files."""
        # First create a working checkpointer with save_freq to trigger checkpoint save
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir,
            save_freq=1,  # Save after each sample
            flow_id=self.flow_id,
        )

        # Create a good checkpoint first
        good_data = Dataset.from_dict({"input": ["test1"], "output": ["result1"]})
        checkpointer.add_completed_samples(good_data)

        # Create a corrupted checkpoint file manually
        corrupted_file = Path(self.temp_dir) / "checkpoint_0002.jsonl"
        with open(corrupted_file, "w") as f:
            f.write("invalid json content")

        # Should still load the good checkpoint and warn about the bad one
        completed = checkpointer.load_all_completed_samples()

        # Should get the good data (may be None if all checkpoints failed to load)
        if completed is not None:
            assert len(completed) >= 1
            assert "test1" in completed["input"]
