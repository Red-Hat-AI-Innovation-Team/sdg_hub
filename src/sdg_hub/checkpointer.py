# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import uuid

# Third Party
from datasets import Dataset, load_dataset
from datasets.data_files import EmptyDatasetError

# Local
from .logger_config import setup_logger
from .utils.datautils import safe_concatenate_datasets

logger = setup_logger(__name__)


class Checkpointer:
    """
    Handles checkpointing functionality for SDG data generation.
    Manages saving intermediate results and loading existing checkpoints.
    """

    def __init__(
        self, checkpoint_dir: Optional[str] = None, save_freq: Optional[int] = None
    ):
        """
        Initialize the Checkpointer.

        Args:
            checkpoint_dir: Directory to save/load checkpoints. If None, checkpointing is disabled.
            save_freq: Frequency for saving intermediate checkpoints during batch processing.
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq

    def load_existing_data(
        self, seed_dataset: Dataset
    ) -> tuple[Dataset, Optional[Dataset]]:
        """
        Load existing checkpoint data and determine what still needs to be generated.

        Args:
            seed_dataset: Original input dataset

        Returns:
            Tuple of (remaining_data_to_generate, pre_generated_data)
            If no checkpoints exist, returns (seed_dataset, None)
        """
        if self.checkpoint_dir is None:
            return seed_dataset, None

        try:
            # Load existing checkpoints
            pre_generated_data = load_dataset(
                "json", data_dir=self.checkpoint_dir, split="train"
            )
            logger.info(
                f"Loading existing checkpoints from {self.checkpoint_dir}, "
                f"with {pre_generated_data.num_rows} rows"
            )

            # Find missing data that still needs to be generated
            missing_data = self._get_missing_data(seed_dataset, pre_generated_data)

            if missing_data.num_rows == 0:
                logger.info(
                    f"All seed data has been generated, no missing rows found, "
                    f"returning data from {self.checkpoint_dir}"
                )
                return missing_data, pre_generated_data

            logger.info(f"Found {missing_data.num_rows} missing rows in the dataset")
            return missing_data, pre_generated_data

        except EmptyDatasetError:
            logger.info(
                f"No existing checkpoints found in {self.checkpoint_dir}, "
                f"generating from scratch"
            )
            return seed_dataset, None

    def _get_missing_data(self, seed_data: Dataset, generated_data: Dataset) -> Dataset:
        """
        Identify rows in seed_data that are not present in generated_data.

        Args:
            seed_data: Original seed dataset
            generated_data: Previously generated dataset

        Returns:
            Dataset containing only the missing rows from seed_data
        """
        # Get the common columns between the two datasets
        common_columns = list(
            set(seed_data.column_names) & set(generated_data.column_names)
        )

        # Extract the relevant data based on common columns
        seed_data_common = seed_data.select_columns(common_columns)
        generated_data_common = generated_data.select_columns(common_columns)

        # Convert to Pandas DataFrames for easier comparison
        seed_df = seed_data_common.to_pandas()
        generated_df = generated_data_common.to_pandas()

        # Identify missing rows
        missing_df = seed_df[
            ~seed_df.apply(tuple, 1).isin(generated_df.apply(tuple, 1))
        ]

        # Convert back to Dataset
        missing_data = Dataset.from_pandas(missing_df, preserve_index=False)

        return missing_data

    def save_intermediate_checkpoint(self, dataset: Dataset) -> None:
        """
        Save intermediate checkpoint data to disk.

        Args:
            dataset: Dataset to save as checkpoint
        """
        if self.checkpoint_dir is None:
            return

        checkpoint_id = uuid.uuid4().hex
        checkpoint_file = f"{self.checkpoint_dir}/data_checkpoint_{checkpoint_id}.jsonl"
        logger.info(f"Saving checkpoint to {checkpoint_file}")
        dataset.to_json(checkpoint_file, orient="records", lines=True)

    def should_save_checkpoint(self, current_split_index: int) -> bool:
        """
        Determine if a checkpoint should be saved based on save frequency.

        Args:
            current_split_index: Current split index (0-based)

        Returns:
            True if checkpoint should be saved, False otherwise
        """
        if self.save_freq is None or self.checkpoint_dir is None:
            return False
        return (current_split_index + 1) % self.save_freq == 0
