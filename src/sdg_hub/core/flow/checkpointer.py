# SPDX-License-Identifier: Apache-2.0
"""Flow-level checkpointing with sample-level tracking for data generation pipelines."""

# Standard
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import uuid

# Third Party
from datasets import Dataset
import numpy as np

# Local
from ..utils.datautils import safe_concatenate_with_validation
from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)


def _make_hashable(x):
    """Convert any value to a hashable representation.

    This is the same logic used in validate_no_duplicates() to handle
    numpy arrays, dicts, lists, etc. when comparing dataset rows.
    """
    try:
        hash(x)
        return x
    except TypeError:
        pass

    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return _make_hashable(x.item())
        return tuple(_make_hashable(i) for i in x)
    if isinstance(x, dict):
        return tuple(
            sorted(
                ((k, _make_hashable(v)) for k, v in x.items()),
                key=lambda kv: repr(kv[0]),
            )
        )
    if isinstance(x, (set, frozenset)):
        return frozenset(_make_hashable(i) for i in x)
    if hasattr(x, "__iter__"):
        return tuple(_make_hashable(i) for i in x)
    return repr(x)


class FlowCheckpointer:
    """Enhanced checkpointer for Flow execution with sample-level tracking.

    Provides data-level checkpointing where progress is saved after processing
    a specified number of samples through the entire flow pipeline.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        save_freq: Optional[int] = None,
        flow_id: Optional[str] = None,
    ):
        """Initialize the FlowCheckpointer.

        Parameters
        ----------
        checkpoint_dir : Optional[str]
            Directory to save/load checkpoints. If None, checkpointing is disabled.
        save_freq : Optional[int]
            Number of completed samples after which to save a checkpoint.
            If None, only final results are saved.
        flow_id : Optional[str]
            Unique ID of the flow for checkpoint identification.
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.flow_id = flow_id or "unknown_flow"

        # Internal state
        self._samples_processed = 0
        self._checkpoint_counter = 0
        self._remaining_input_indices: List[
            int
        ] = []  # Indices of remaining samples to process

        # Ensure checkpoint directory exists
        if self.checkpoint_dir:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @property
    def is_enabled(self) -> bool:
        """Check if checkpointing is enabled."""
        return self.checkpoint_dir is not None

    @property
    def metadata_path(self) -> str:
        """Path to the flow metadata file."""
        return os.path.join(self.checkpoint_dir, ".flow_metadata.json")

    def load_existing_progress(
        self, input_dataset: Dataset
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """Load existing checkpoint data and determine remaining work.

        Parameters
        ----------
        input_dataset : Dataset
            Original input dataset for the flow.

        Returns
        -------
        Tuple[Dataset, Optional[Dataset]]
            (remaining_samples_to_process, completed_samples_dataset)
            If no checkpoints exist, returns (input_dataset, None)
        """
        if not self.is_enabled:
            # No checkpoints, all samples need processing
            self._remaining_input_indices = list(range(len(input_dataset)))
            return input_dataset, None

        try:
            # Load flow metadata
            metadata = self._load_metadata()
            if not metadata:
                logger.info(f"No existing checkpoints found in {self.checkpoint_dir}")
                # No checkpoints, all samples need processing
                self._remaining_input_indices = list(range(len(input_dataset)))
                return input_dataset, None

            # Validate flow identity to prevent mixing checkpoints from different flows
            saved_flow_id = metadata.get("flow_id")
            if saved_flow_id and saved_flow_id != self.flow_id:
                logger.warning(
                    f"Flow ID mismatch: saved checkpoints are for flow ID '{saved_flow_id}' "
                    f"but current flow ID is '{self.flow_id}'. Starting fresh to avoid "
                    f"mixing incompatible checkpoint data."
                )
                # Starting fresh, all samples need processing
                self._remaining_input_indices = list(range(len(input_dataset)))
                return input_dataset, None

            # Load ONLY the _sdg_input_index values from checkpoints (memory efficient!)
            # This avoids materializing all completed samples into memory
            completed_input_indices = self._load_completed_input_indices()
            if not completed_input_indices:
                logger.info("No completed samples found in checkpoints")
                # No checkpoints, all samples need processing
                self._remaining_input_indices = list(range(len(input_dataset)))
                return input_dataset, None

            # Find samples that still need processing using input index tracking
            remaining_indices = self._find_remaining_indices(
                input_dataset, completed_input_indices
            )

            # Select only the remaining samples from input dataset
            if not remaining_indices:
                # All samples completed
                remaining_dataset = input_dataset.select([])
            else:
                remaining_dataset = input_dataset.select(remaining_indices)

            self._samples_processed = len(completed_input_indices)
            self._checkpoint_counter = metadata.get("checkpoint_counter", 0)
            self._remaining_input_indices = remaining_indices  # Store for later use

            logger.info(
                f"Loaded {len(completed_input_indices)} completed sample indices, "
                f"{len(remaining_dataset)} samples remaining"
            )

            # Return (remaining_dataset, None) - we don't return completed_dataset anymore
            # since it's not needed and would waste memory
            return remaining_dataset, None

        except Exception as exc:
            logger.warning(f"Failed to load checkpoints: {exc}. Starting from scratch.")
            # Failed to load, all samples need processing
            self._remaining_input_indices = list(range(len(input_dataset)))
            return input_dataset, None

    def save_chunk_immediately(self, samples: Dataset) -> None:
        """Save a chunk of samples directly to file without accumulating in memory.

        This is the most memory-efficient approach: stream directly from
        Arrow-backed Dataset to JSONL file without any intermediate storage.

        Parameters
        ----------
        samples : Dataset
            Samples to save immediately.
            Must contain '_sdg_input_index' column for checkpoint tracking.
        """
        if not self.is_enabled:
            return

        self._checkpoint_counter += 1
        checkpoint_file = os.path.join(
            self.checkpoint_dir, f"checkpoint_{self._checkpoint_counter:04d}.jsonl"
        )

        # Stream samples directly from Dataset to file
        # Convert to dict ONLY during file write (minimal memory footprint)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            for sample in samples:
                json.dump(dict(sample), f)
                f.write("\n")

        self._samples_processed += len(samples)
        self._save_metadata()

        logger.info(
            f"Saved checkpoint {self._checkpoint_counter} with "
            f"{len(samples)} samples to {checkpoint_file}"
        )

    def _save_metadata(self) -> None:
        """Save flow execution metadata."""
        metadata = {
            "flow_id": self.flow_id,
            "save_freq": self.save_freq,
            "samples_processed": self._samples_processed,
            "checkpoint_counter": self._checkpoint_counter,
            "last_updated": str(uuid.uuid4()),  # Simple versioning
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load flow execution metadata."""
        if not os.path.exists(self.metadata_path):
            return None

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to load metadata: {exc}")
            return None

    def load_all_completed_samples(self) -> Optional[Dataset]:
        """Load all completed samples from checkpoint files.

        Returns
        -------
        Optional[Dataset]
            Dataset containing all completed samples from checkpoint files,
            or None if no checkpoints exist.
        """
        checkpoint_files = []
        checkpoint_dir = Path(self.checkpoint_dir)

        # Find all checkpoint files
        for file_path in checkpoint_dir.glob("checkpoint_*.jsonl"):
            checkpoint_files.append(str(file_path))

        if not checkpoint_files:
            return None

        # Sort checkpoint files by number
        checkpoint_files.sort()

        # Load and concatenate all checkpoint datasets
        datasets = []
        for file_path in checkpoint_files:
            try:
                dataset = Dataset.from_json(file_path)
                if len(dataset) > 0:
                    datasets.append(dataset)
                    logger.debug(
                        f"Loaded checkpoint: {file_path} ({len(dataset)} samples)"
                    )
            except Exception as exc:
                logger.warning(f"Failed to load checkpoint {file_path}: {exc}")

        if not datasets:
            return None

        return safe_concatenate_with_validation(datasets, "checkpoint files")

    def _load_completed_input_indices(self) -> set[int]:
        """Load only the _sdg_input_index values from checkpoint files (memory efficient).

        Uses streaming to avoid loading entire checkpoint files into memory.

        Returns
        -------
        set[int]
            Set of unique input indices that have been completed.
        """
        from datasets import load_dataset

        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.jsonl"))

        if not checkpoint_files:
            return set()

        completed_indices = set()

        # Use streaming to load only the _sdg_input_index column
        for checkpoint_file in checkpoint_files:
            try:
                # Stream the checkpoint file and select only _sdg_input_index column
                ds_stream = load_dataset(
                    "json",
                    data_files=str(checkpoint_file),
                    split="train",
                    streaming=True,
                )
                ds_indices = ds_stream.select_columns(["_sdg_input_index"])

                # Extract the index values
                for row in ds_indices:
                    completed_indices.add(row["_sdg_input_index"])

                logger.debug(
                    f"Loaded indices from {checkpoint_file.name}: "
                    f"{len(completed_indices)} unique so far"
                )

            except Exception as exc:
                logger.warning(f"Failed to load indices from {checkpoint_file}: {exc}")

        logger.info(
            f"Loaded {len(completed_indices)} unique completed input indices from "
            f"{len(checkpoint_files)} checkpoint file(s)"
        )

        return completed_indices

    def _find_remaining_indices(
        self, input_dataset: Dataset, completed_indices: set[int]
    ) -> list[int]:
        """Find input sample indices that still need processing.

        Parameters
        ----------
        input_dataset : Dataset
            Original input dataset.
        completed_indices : set[int]
            Set of input indices that have been completed.

        Returns
        -------
        list[int]
            Sorted list of indices that still need processing.
        """
        total_input_indices = set(range(len(input_dataset)))
        remaining_indices = sorted(total_input_indices - completed_indices)

        logger.info(
            f"Input samples processed: {len(completed_indices)}/{len(input_dataset)}"
        )
        logger.info(f"Input samples remaining: {len(remaining_indices)}")

        return remaining_indices

    def _find_remaining_samples_by_index(
        self, input_dataset: Dataset, completed_dataset: Dataset
    ) -> tuple[Dataset, list[int]]:
        """Find input samples that haven't been processed using input index tracking.

        Uses the _sdg_input_index column in completed samples to determine which
        input rows have been processed. This works correctly even when the flow
        amplifies data (1 input → many outputs) or modifies columns.

        Parameters
        ----------
        input_dataset : Dataset
            Original input dataset.
        completed_dataset : Dataset
            Dataset of completed samples from checkpoints.

        Returns
        -------
        tuple[Dataset, list[int]]
            (remaining_dataset, remaining_indices) - The dataset of remaining samples
            and their original indices in the input dataset.
        """
        # Check if completed dataset has input index tracking
        if "_sdg_input_index" not in completed_dataset.column_names:
            logger.warning(
                "Checkpoints don't have _sdg_input_index column. "
                "Falling back to column-based comparison (may not work correctly "
                "if flow modifies input columns or amplifies data)."
            )
            remaining = self._find_remaining_samples(input_dataset, completed_dataset)
            # Without index tracking, assume sequential indices for remaining samples
            indices = list(range(len(remaining)))
            return remaining, indices

        # Get unique input indices that have been processed
        completed_indices = set(completed_dataset["_sdg_input_index"])
        logger.info(
            f"Found {len(completed_indices)} unique input indices in checkpoints: {sorted(completed_indices)}"
        )

        # Find remaining input indices
        total_input_indices = set(range(len(input_dataset)))
        remaining_indices = sorted(total_input_indices - completed_indices)

        logger.info(f"Input samples processed: {sorted(completed_indices)}")
        logger.info(f"Input samples remaining: {remaining_indices}")

        if not remaining_indices:
            # All samples completed
            return input_dataset.select([]), []

        return input_dataset.select(remaining_indices), remaining_indices

    def _find_remaining_samples(
        self, input_dataset: Dataset, completed_dataset: Dataset
    ) -> Dataset:
        """Find samples from input_dataset that are not in completed_dataset.

        Note: Assumes input_dataset contains unique samples. For datasets with
        duplicates, multiset semantics with collections.Counter would be needed.

        Parameters
        ----------
        input_dataset : Dataset
            Original input dataset (assumed to contain unique samples).
        completed_dataset : Dataset
            Dataset of completed samples.

        Returns
        -------
        Dataset
            Samples that still need processing.
        """
        # Get common columns for comparison
        input_columns = set(input_dataset.column_names)
        completed_columns = set(completed_dataset.column_names)
        common_columns = list(input_columns & completed_columns)

        logger.info(f"Checkpoint comparison - Input columns: {sorted(input_columns)}")
        logger.info(
            f"Checkpoint comparison - Completed columns: {sorted(completed_columns)[:10]}..."
        )  # First 10 to avoid clutter
        logger.info(
            f"Checkpoint comparison - Using {len(common_columns)} common columns: {sorted(common_columns)}"
        )

        if not common_columns:
            logger.warning(
                "No common columns found between input and completed datasets. "
                "Processing all input samples."
            )
            return input_dataset

        # Convert to pandas for easier comparison
        input_df = input_dataset.select_columns(common_columns).to_pandas()
        completed_df = completed_dataset.select_columns(common_columns).to_pandas()

        # Convert all cells to hashable representations (handles numpy arrays, lists, dicts, etc.)
        # This uses the same logic as validate_no_duplicates() to ensure robust comparison
        if hasattr(input_df, "map"):
            input_df_hashable = input_df.map(_make_hashable)
            completed_df_hashable = completed_df.map(_make_hashable)
        else:
            input_df_hashable = input_df.applymap(_make_hashable)
            completed_df_hashable = completed_df.applymap(_make_hashable)

        # Find rows that haven't been completed
        # Use tuple representation for comparison
        input_tuples = set(input_df_hashable.apply(tuple, axis=1))
        completed_tuples = set(completed_df_hashable.apply(tuple, axis=1))
        remaining_tuples = input_tuples - completed_tuples

        logger.info(
            f"Checkpoint comparison - Input samples (unique): {len(input_tuples)}"
        )
        logger.info(
            f"Checkpoint comparison - Completed samples (unique): {len(completed_tuples)}"
        )
        logger.info(
            f"Checkpoint comparison - Samples matched: {len(input_tuples & completed_tuples)}"
        )
        logger.info(
            f"Checkpoint comparison - Remaining samples: {len(remaining_tuples)}"
        )

        # Debug: Show sample data to understand mismatch
        if len(remaining_tuples) > 0 and len(remaining_tuples) == len(input_tuples):
            logger.warning("No input samples matched with completed checkpoints!")
            logger.info("Sample input data (first row, ALL common columns):")
            for col in common_columns:
                val = input_df.iloc[0][col]
                logger.info(f"  {col}: {type(val).__name__} = {repr(val)[:150]}")
            logger.info("Sample completed data (first row, ALL common columns):")
            for col in common_columns:
                val = completed_df.iloc[0][col]
                logger.info(f"  {col}: {type(val).__name__} = {repr(val)[:150]}")

            # Show hashable representations for comparison
            logger.info("Hashable comparison (first row):")
            input_hash = tuple(input_df_hashable.iloc[0])
            completed_hash = tuple(completed_df_hashable.iloc[0])
            logger.info(
                f"  Input hashable (len={len(input_hash)}): {str(input_hash)[:200]}"
            )
            logger.info(
                f"  Completed hashable (len={len(completed_hash)}): {str(completed_hash)[:200]}"
            )
            logger.info(f"  Are they equal? {input_hash == completed_hash}")

        # Filter input dataset to only remaining samples
        # Use the hashable version for comparison but return indices from original dataset
        remaining_mask = input_df_hashable.apply(tuple, axis=1).isin(remaining_tuples)
        remaining_indices = input_df[remaining_mask].index.tolist()

        if not remaining_indices:
            # Return empty dataset with same structure
            return input_dataset.select([])

        return input_dataset.select(remaining_indices)

    def get_progress_info(self) -> Dict[str, Any]:
        """Get information about current progress.

        Returns
        -------
        Dict[str, Any]
            Progress information including samples processed, checkpoints saved, etc.
        """
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "save_freq": self.save_freq,
            "flow_id": self.flow_id,
            "samples_processed": self._samples_processed,
            "checkpoint_counter": self._checkpoint_counter,
            "is_enabled": self.is_enabled,
        }

    def cleanup_checkpoints(self) -> None:
        """Remove all checkpoint files and metadata."""
        if not self.is_enabled:
            return

        checkpoint_dir = Path(self.checkpoint_dir)
        if not checkpoint_dir.exists():
            return

        # Remove all checkpoint files
        for file_path in checkpoint_dir.glob("checkpoint_*.jsonl"):
            file_path.unlink()
            logger.debug(f"Removed checkpoint file: {file_path}")

        # Remove metadata file
        metadata_path = Path(self.metadata_path)
        if metadata_path.exists():
            metadata_path.unlink()
            logger.debug(f"Removed metadata file: {metadata_path}")

        logger.info(f"Cleaned up all checkpoints in {self.checkpoint_dir}")
