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

# Local
from ..utils.datautils import safe_concatenate_with_validation
from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)


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
        self._pending_samples: List[Dict[str, Any]] = []

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
        return os.path.join(self.checkpoint_dir, "flow_metadata.json")

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
            return input_dataset, None

        try:
            # Load flow metadata
            metadata = self._load_metadata()
            if not metadata:
                logger.info(f"No existing checkpoints found in {self.checkpoint_dir}")
                return input_dataset, None

            # Validate flow identity to prevent mixing checkpoints from different flows
            saved_flow_id = metadata.get("flow_id")
            if saved_flow_id and saved_flow_id != self.flow_id:
                logger.warning(
                    f"Flow ID mismatch: saved checkpoints are for flow ID '{saved_flow_id}' "
                    f"but current flow ID is '{self.flow_id}'. Starting fresh to avoid "
                    f"mixing incompatible checkpoint data."
                )
                return input_dataset, None

            # Load all completed samples from checkpoints
            completed_dataset = self._load_completed_samples()
            if completed_dataset is None or len(completed_dataset) == 0:
                logger.info("No completed samples found in checkpoints")
                return input_dataset, None

            # Find samples that still need processing
            remaining_dataset = self._find_remaining_samples(
                input_dataset, completed_dataset
            )

            self._samples_processed = len(completed_dataset)
            self._checkpoint_counter = metadata.get("checkpoint_counter", 0)

            logger.info(
                f"Loaded {len(completed_dataset)} completed samples, "
                f"{len(remaining_dataset)} samples remaining"
            )

            return remaining_dataset, completed_dataset

        except Exception as exc:
            logger.warning(f"Failed to load checkpoints: {exc}. Starting from scratch.")
            return input_dataset, None

    def add_completed_samples(self, samples: Dataset) -> None:
        """Add samples that have completed the entire flow.

        Parameters
        ----------
        samples : Dataset
            Samples that have completed processing through all blocks.
        """
        if not self.is_enabled:
            return

        # Add to pending samples
        for sample in samples:
            self._pending_samples.append(sample)
            self._samples_processed += 1

            # Check if we should save a checkpoint
            if self.save_freq and len(self._pending_samples) >= self.save_freq:
                self._save_checkpoint()

    def save_final_checkpoint(self) -> None:
        """Save any remaining pending samples as final checkpoint."""
        if not self.is_enabled:
            return

        if self._pending_samples:
            sample_count = len(self._pending_samples)
            self._save_checkpoint()
            logger.info(f"Saved final checkpoint with {sample_count} samples")

    def _save_checkpoint(self) -> None:
        """Save current pending samples to a checkpoint file."""
        if not self._pending_samples:
            return

        self._checkpoint_counter += 1
        checkpoint_file = os.path.join(
            self.checkpoint_dir, f"checkpoint_{self._checkpoint_counter:04d}.jsonl"
        )

        # Convert pending samples to dataset and save
        checkpoint_dataset = Dataset.from_list(self._pending_samples)
        checkpoint_dataset.to_json(checkpoint_file, orient="records", lines=True)

        # Update metadata
        self._save_metadata()

        logger.info(
            f"Saved checkpoint {self._checkpoint_counter} with "
            f"{len(self._pending_samples)} samples to {checkpoint_file}"
        )

        # Clear pending samples
        self._pending_samples.clear()

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

    def _load_completed_samples(self) -> Optional[Dataset]:
        """Load all completed samples from checkpoint files."""
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

        if not common_columns:
            logger.warning(
                "No common columns found between input and completed datasets. "
                "Processing all input samples."
            )
            return input_dataset

        # Convert to pandas for easier comparison
        input_df = input_dataset.select_columns(common_columns).to_pandas()
        completed_df = completed_dataset.select_columns(common_columns).to_pandas()

        # Find rows that haven't been completed
        # Use tuple representation for comparison
        input_tuples = set(input_df.apply(tuple, axis=1))
        completed_tuples = set(completed_df.apply(tuple, axis=1))
        remaining_tuples = input_tuples - completed_tuples

        # Filter input dataset to only remaining samples
        remaining_mask = input_df.apply(tuple, axis=1).isin(remaining_tuples)
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
            "pending_samples": len(self._pending_samples),
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
