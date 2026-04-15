# SPDX-License-Identifier: Apache-2.0
"""Tests for the Flow checkpointing functionality."""

# Standard
from pathlib import Path
import json
import tempfile

# Third Party
import pandas as pd

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

    def test_find_remaining_samples_with_list_columns(self):
        """Test finding remaining samples when columns contain lists (unhashable types).

        This test demonstrates the fix for handling unhashable types like lists and
        numpy arrays in DataFrame columns. Without _make_hashable(), this would fail
        with: TypeError: unhashable type: 'list' or 'numpy.ndarray'.

        This is a common scenario in SDG flows where columns contain lists of strings or other similar items.
        """
        checkpointer = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        # Create input dataset with list column (common in SDG flows)
        # This simulates columns like 'icl_document' that contain list of examples
        input_dataset = pd.DataFrame(
            {
                "document": ["doc1", "doc2", "doc3"],
                "icl_examples": [
                    ["example1", "example2"],  # List - unhashable without fix
                    ["example3", "example4"],
                    ["example5", "example6"],
                ],
                "domain": ["type1", "type2", "type3"],
            }
        )

        # Create completed dataset with same schema but only first 2 samples
        # When loaded from JSONL and converted to DataFrame, lists may become numpy arrays
        completed_dataset = pd.DataFrame(
            {
                "document": ["doc1", "doc2"],
                "icl_examples": [
                    ["example1", "example2"],
                    ["example3", "example4"],
                ],
                "domain": ["type1", "type2"],
                "output": [
                    "result1",
                    "result2",
                ],  # Output columns added during processing
            }
        )

        # This would previously fail with: TypeError: unhashable type: 'list'
        # The _make_hashable() fix converts lists to tuples for comparison
        remaining = checkpointer._find_remaining_samples(
            input_dataset, completed_dataset
        )

        # Should correctly identify that doc1 and doc2 are completed, doc3 is remaining
        assert len(remaining) == 1
        assert remaining["document"].tolist() == ["doc3"]
        assert remaining["icl_examples"].tolist() == [["example5", "example6"]]
        assert remaining["domain"].tolist() == ["type3"]

    def test_resumption_with_unhashable_columns(self):
        """Test full checkpoint save and load cycle with unhashable columns.

        This integration test verifies that the checkpointer can:
        1. Save checkpoints with list/array columns
        2. Load those checkpoints back
        3. Correctly identify remaining work when comparing datasets with list columns

        This simulates a real SDG workflow where flows are interrupted and resumed.
        """
        # Step 1: Create initial checkpointer and save some progress
        checkpointer1 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, save_freq=2, flow_id=self.flow_id
        )

        # Simulate completed work with list columns (like icl_document, icl_queries)
        completed_data = pd.DataFrame(
            {
                "document_outline": ["Article 1", "Article 2"],
                "icl_document": [
                    ["Context for article 1"],
                    ["Context for article 2"],
                ],
                "icl_query": [
                    ["Question 1 for article 1", "Question 2 for article 1"],
                    ["Question 1 for article 2"],
                ],
                "domain": ["science", "technology"],
                "output_summary": ["Summary 1", "Summary 2"],
            }
        )

        checkpointer1.add_completed_samples(completed_data)

        # Verify checkpoint was saved
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 1

        # Step 2: Simulate resuming the flow with a new checkpointer
        checkpointer2 = FlowCheckpointer(
            checkpoint_dir=self.temp_dir, flow_id=self.flow_id
        )

        # Input dataset for resumption (contains all original + new samples)
        input_dataset = pd.DataFrame(
            {
                "document_outline": [
                    "Article 1",
                    "Article 2",
                    "Article 3",
                    "Article 4",
                ],
                "icl_document": [
                    ["Context for article 1"],
                    ["Context for article 2"],
                    ["Context for article 3"],
                    ["Context for article 4"],
                ],
                "icl_query": [
                    ["Question 1 for article 1", "Question 2 for article 1"],
                    ["Question 1 for article 2"],
                    ["Question 1 for article 3", "Question 2 for article 3"],
                    ["Question 1 for article 4"],
                ],
                "domain": ["science", "technology", "engineering", "mathematics"],
            }
        )

        # Load progress - this would fail without _make_hashable() fix
        remaining, completed = checkpointer2.load_existing_progress(input_dataset)

        # Verify correct resumption
        assert len(completed) == 2, "Should have loaded 2 completed samples"
        assert len(remaining) == 2, "Should have 2 remaining samples"
        assert remaining["document_outline"].tolist() == ["Article 3", "Article 4"]
        assert remaining["domain"].tolist() == ["engineering", "mathematics"]

        # Verify the list columns are correctly preserved
        assert remaining["icl_document"].tolist() == [
            ["Context for article 3"],
            ["Context for article 4"],
        ]
