# SPDX-License-Identifier: Apache-2.0
"""Flow-level checkpointing with sample-level tracking for data generation pipelines."""

# Standard
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import os
import uuid

# Third Party
import pandas as pd

# Local
from ..utils.datautils import _make_hashable, safe_concatenate_with_validation
from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)


class FlowCheckpointer:
    """Enhanced checkpointer for Flow execution with sample-level tracking.

    Provides data-level checkpointing where progress is saved after processing
    a specified number of samples through the entire flow pipeline.

    Notes
    -----
    Dataset Validation:
        Uses a dataset signature (hash of columns + head/tail samples) to detect:
        - Different datasets with same checkpoint_dir
        - Modified content in beginning/end of dataset
        - Schema changes

        Does NOT detect changes in middle rows (performance trade-off).
        For large datasets (>10K rows), this is acceptable as head/tail changes
        are most common in iterative development.

    Input Index Tracking:
        Adds _sdg_input_index column to track original row positions.
        This enables checkpoint resumption even when flows:
        - Rename input columns
        - Remove input columns
        - Reorder columns

        The index survives all transformations and is saved in checkpoints.

    Memory Optimization:
        In chunked mode (save_freq > 0), chunks are saved immediately and
        not accumulated in memory. Final results are loaded from checkpoint
        files rather than kept in RAM, halving memory usage.

    save_freq Semantics:
        save_freq only controls INPUT chunking (in base.py), not output batching.
        Each processed chunk is saved immediately to a checkpoint file.
        This simplifies semantics and eliminates confusion about what save_freq means.
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

    @staticmethod
    def _compute_dataset_signature(dataset: pd.DataFrame, sample_size: int = 10) -> str:
        """Compute a fingerprint hash of the dataset for validation.

        Uses first and last N rows to create a lightweight signature that detects:
        - Different datasets
        - Modified content in head/tail rows
        - Column schema changes

        Note: Does not detect changes in middle (non head or tail) rows.

        Parameters
        ----------
        dataset : pd.DataFrame
            Input dataset to fingerprint.
        sample_size : int, default=10
            Number of rows to sample from head and tail.

        Returns
        -------
        str
            64-character hex hash representing dataset signature.
        """
        # Apply _make_hashable to handle unhashable types (numpy arrays, dicts, lists)
        head_sample = dataset.head(min(sample_size, len(dataset))).map(_make_hashable)
        tail_sample = dataset.tail(min(sample_size, len(dataset))).map(_make_hashable)

        # Create signature from columns + head + tail samples
        # Use tuples for deterministic hashing
        signature_data = {
            "columns": tuple(sorted(dataset.columns.tolist())),
            "size": len(dataset),
            "head_sample": tuple(head_sample.apply(tuple, axis=1)),
            "tail_sample": tuple(tail_sample.apply(tuple, axis=1)),
        }

        return hashlib.sha256(str(signature_data).encode()).hexdigest()

    @property
    def is_enabled(self) -> bool:
        """Check if checkpointing is enabled."""
        return self.checkpoint_dir is not None

    @property
    def metadata_path(self) -> str:
        """Path to the flow metadata file."""
        return os.path.join(self.checkpoint_dir, "flow_metadata.json")

    def load_existing_progress(
        self, input_dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load existing checkpoint data and determine remaining work.

        Implements hybrid error handling:
        - FATAL errors (raise exception): Dataset signature mismatch, Flow ID mismatch
        - RECOVERABLE errors (log warning, start fresh): Corrupted files, missing data

        Parameters
        ----------
        input_dataset : pd.DataFrame
            Original input dataset for the flow.

        Returns
        -------
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]
            (remaining_samples_to_process, completed_samples_dataset)
            If no checkpoints exist, returns (input_dataset, None)

        Raises
        ------
        FlowValidationError
            If dataset signature or flow ID mismatch detected (requires user action)
        """
        if not self.is_enabled:
            return input_dataset, None

        # Import here to avoid circular dependencies
        from ..utils.error_handling import FlowValidationError

        # Load flow metadata (wrapped to handle file system errors)
        try:
            metadata = self._load_metadata()
        except Exception as exc:
            logger.warning(
                f"Failed to load checkpoint metadata: {exc}. Starting from scratch."
            )
            return input_dataset, None

        if not metadata:
            logger.info(f"No existing checkpoints found in {self.checkpoint_dir}")
            # Save initial dataset signature for future validation
            try:
                self._save_metadata(input_dataset=input_dataset)
            except Exception as exc:
                logger.warning(f"Failed to save metadata: {exc}")
            return input_dataset, None

        # CRITICAL VALIDATION: Flow ID mismatch (FATAL - raise error)
        saved_flow_id = metadata.get("flow_id")
        if saved_flow_id and saved_flow_id != self.flow_id:
            raise FlowValidationError(
                f"Flow ID mismatch detected!\n"
                f"\n"
                f"Saved checkpoints are for flow: '{saved_flow_id}'\n"
                f"Current flow ID is: '{self.flow_id}'\n"
                f"\n"
                f"Mixing checkpoints from different flows can lead to incorrect results.\n"
                f"\n"
                f"To fix this issue, choose one of the following:\n"
                f"  1. Use a different checkpoint_dir for this flow:\n"
                f"     flow.generate(dataset, checkpoint_dir='checkpoints_{self.flow_id}')\n"
                f"  2. Delete all contents of '{self.checkpoint_dir}/' to start fresh\n"
                f"  3. Disable checkpointing entirely:\n"
                f"     flow.generate(dataset, checkpoint_dir=None, save_freq=None)"
            )

        # CRITICAL VALIDATION: Dataset signature mismatch (FATAL - raise error)
        if "dataset_signature" in metadata:
            current_signature = self._compute_dataset_signature(input_dataset)
            saved_signature = metadata.get("dataset_signature")
            saved_size = metadata.get("dataset_size", 0)
            current_size = len(input_dataset)

            # Strict validation: error on ANY change (signature OR size mismatch)
            if current_signature != saved_signature or current_size != saved_size:
                raise FlowValidationError(
                    f"Dataset has changed since checkpoints were created!\n"
                    f"\n"
                    f"Saved checkpoint info:\n"
                    f"  - Columns: {metadata.get('input_columns')}\n"
                    f"  - Size: {saved_size} rows\n"
                    f"  - Signature: {saved_signature[:16]}...\n"
                    f"\n"
                    f"Current dataset info:\n"
                    f"  - Columns: {input_dataset.columns.tolist()}\n"
                    f"  - Size: {current_size} rows\n"
                    f"  - Signature: {current_signature[:16]}...\n"
                    f"\n"
                    f"To fix this issue, choose one of the following:\n"
                    f"  1. Use a different checkpoint_dir for this dataset:\n"
                    f"     flow.generate(dataset, checkpoint_dir='new_checkpoint_dir')\n"
                    f"  2. Delete all contents of '{self.checkpoint_dir}/' (including flow_metadata.json)\n"
                    f"  3. Disable checkpointing entirely:\n"
                    f"     flow.generate(dataset, checkpoint_dir=None, save_freq=None)"
                )

        # RECOVERABLE OPERATIONS: Load checkpoint data (errors are logged, not fatal)
        try:
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

        except (ValueError, KeyError, TypeError) as exc:
            # Specific recoverable errors (corrupted data, schema issues, etc.)
            logger.warning(
                f"Failed to load checkpoint data: {exc}. Starting from scratch."
            )
            return input_dataset, None
        except Exception as exc:
            # Unexpected errors - log and continue
            logger.error(
                f"Unexpected error loading checkpoints: {exc}. Starting from scratch."
            )
            return input_dataset, None

    def add_completed_samples(self, samples: pd.DataFrame) -> None:
        """Add samples that have completed the entire flow.

        Parameters
        ----------
        samples : pd.DataFrame
            Samples that have completed processing through all blocks.
        """
        if not self.is_enabled:
            return

        # Convert all samples to dicts in one vectorized operation (10-100x faster than iterrows)
        samples_dicts = samples.to_dict(orient="records")

        # Batch append to pending samples
        self._pending_samples.extend(samples_dicts)
        self._samples_processed += len(samples_dicts)

        # Check if we should save checkpoints (may need to save multiple times)
        while self.save_freq and len(self._pending_samples) >= self.save_freq:
            self._save_checkpoint()

    def save_final_checkpoint(self) -> None:
        """Save any remaining pending samples as final checkpoint."""
        if not self.is_enabled:
            return

        if self._pending_samples:
            sample_count = len(self._pending_samples)
            self._save_checkpoint()
            logger.info(f"Saved final checkpoint with {sample_count} samples")

    def save_chunk_immediately(self, chunk: pd.DataFrame) -> None:
        """Save a chunk directly to a checkpoint file without buffering.

        This method saves processed chunks immediately without accumulating them in memory,
        reducing memory footprint. Each chunk becomes one checkpoint file.

        Parameters
        ----------
        chunk : pd.DataFrame
            Processed chunk to save immediately.
        """
        if not self.is_enabled:
            return

        if len(chunk) == 0:
            logger.warning("Attempted to save empty chunk, skipping")
            return

        self._checkpoint_counter += 1
        checkpoint_file = os.path.join(
            self.checkpoint_dir, f"checkpoint_{self._checkpoint_counter:04d}.jsonl"
        )

        # Save chunk directly to file
        chunk.to_json(checkpoint_file, orient="records", lines=True)

        self._samples_processed += len(chunk)

        # Update metadata (without dataset signature - already saved)
        self._save_metadata(input_dataset=None)

        logger.info(
            f"Saved checkpoint {self._checkpoint_counter} with "
            f"{len(chunk)} samples to {checkpoint_file}"
        )

    def load_all_checkpoints(self) -> pd.DataFrame:
        """Load and concatenate all checkpoint files.

        This is used instead of keeping processed chunks in memory.
        Loads from disk at the end of processing.

        Returns
        -------
        pd.DataFrame
            All completed samples from all checkpoint files.

        Raises
        ------
        ValueError
            If no checkpoint files found.
        """
        if not self.is_enabled:
            raise ValueError("Cannot load checkpoints when checkpointing is disabled")

        completed_dataset = self._load_completed_samples()

        if completed_dataset is None or len(completed_dataset) == 0:
            raise ValueError(
                f"No checkpoint files found in {self.checkpoint_dir}. "
                f"This should not happen after processing."
            )

        logger.info(
            f"Loaded {len(completed_dataset)} total samples from "
            f"{self._checkpoint_counter} checkpoint files"
        )

        return completed_dataset

    def _save_checkpoint(self) -> None:
        """Save current pending samples to a checkpoint file."""
        if not self._pending_samples:
            return

        self._checkpoint_counter += 1
        checkpoint_file = os.path.join(
            self.checkpoint_dir, f"checkpoint_{self._checkpoint_counter:04d}.jsonl"
        )

        # Convert pending samples to dataframe and save
        checkpoint_df = pd.DataFrame(self._pending_samples)
        checkpoint_df.to_json(checkpoint_file, orient="records", lines=True)

        # Update metadata
        self._save_metadata()

        logger.info(
            f"Saved checkpoint {self._checkpoint_counter} with "
            f"{len(self._pending_samples)} samples to {checkpoint_file}"
        )

        # Clear pending samples
        self._pending_samples.clear()

    def _save_metadata(self, input_dataset: Optional[pd.DataFrame] = None) -> None:
        """Save flow execution metadata.

        Parameters
        ----------
        input_dataset : Optional[pd.DataFrame]
            Input dataset to compute signature from. If provided, saves dataset
            validation fields (signature, size, columns). If None, only updates
            execution progress fields.
        """
        metadata = {
            "flow_id": self.flow_id,
            "save_freq": self.save_freq,
            "samples_processed": self._samples_processed,
            "checkpoint_counter": self._checkpoint_counter,
            "last_updated": str(uuid.uuid4()),  # Simple versioning
        }

        # Add dataset validation fields (only on first save with input_dataset)
        if input_dataset is not None:
            metadata["dataset_signature"] = self._compute_dataset_signature(
                input_dataset
            )
            metadata["dataset_size"] = len(input_dataset)
            metadata["input_columns"] = input_dataset.columns.tolist()

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

    def _load_completed_samples(self) -> Optional[pd.DataFrame]:
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

        # Load and concatenate all checkpoint dataframes
        dataframes = []
        for file_path in checkpoint_files:
            try:
                df = pd.read_json(file_path, lines=True)
                if len(df) > 0:
                    dataframes.append(df)
                    logger.debug(f"Loaded checkpoint: {file_path} ({len(df)} samples)")
            except Exception as exc:
                logger.warning(f"Failed to load checkpoint {file_path}: {exc}")

        if not dataframes:
            return None

        return safe_concatenate_with_validation(dataframes, "checkpoint files")

    def _find_remaining_samples(
        self, input_dataset: pd.DataFrame, completed_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """Find samples from input_dataset that are not in completed_dataset.

        Uses _sdg_input_index for matching when available (robust to column changes).
        Falls back to column-based matching for backward compatibility.

        Parameters
        ----------
        input_dataset : pd.DataFrame
            Original input dataset.
        completed_dataset : pd.DataFrame
            Dataset of completed samples.

        Returns
        -------
        pd.DataFrame
            Samples that still need processing.
        """
        # Preferred method: Match by _sdg_input_index (robust to column transformations)
        if "_sdg_input_index" in completed_dataset.columns:
            completed_indices = set(completed_dataset["_sdg_input_index"])

            # Add index to input if not present
            if "_sdg_input_index" not in input_dataset.columns:
                input_dataset = input_dataset.copy()
                input_dataset["_sdg_input_index"] = range(len(input_dataset))

            # Find rows not yet completed
            remaining_mask = ~input_dataset["_sdg_input_index"].isin(completed_indices)
            remaining_dataset = input_dataset[remaining_mask]

            logger.info(
                f"Matched by _sdg_input_index: {len(completed_indices)} completed, "
                f"{len(remaining_dataset)} remaining"
            )

            return remaining_dataset

        # Fallback: Column-based matching (old behavior for backward compatibility)
        logger.warning(
            "Checkpoints don't have _sdg_input_index. "
            "Using legacy column-based matching (may fail if flow modifies input columns)."
        )

        # Get common columns for comparison
        input_columns = set(input_dataset.columns.tolist())
        completed_columns = set(completed_dataset.columns.tolist())
        common_columns = list(input_columns & completed_columns)

        if not common_columns:
            logger.warning(
                "No common columns found between input and completed datasets. "
                "Processing all input samples."
            )
            return input_dataset

        # Select only common columns for comparison
        input_df = input_dataset[common_columns]
        completed_df = completed_dataset[common_columns]

        # Find rows that haven't been completed
        input_tuples = set(input_df.apply(tuple, axis=1))
        completed_tuples = set(completed_df.apply(tuple, axis=1))
        remaining_tuples = input_tuples - completed_tuples

        # Filter input dataset to only remaining samples
        remaining_mask = input_df.apply(tuple, axis=1).isin(remaining_tuples)
        remaining_indices = input_df[remaining_mask].index.tolist()

        if not remaining_indices:
            return input_dataset.iloc[0:0]  # Empty dataframe

        return input_dataset.iloc[remaining_indices]

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
