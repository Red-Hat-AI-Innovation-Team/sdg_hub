# SPDX-License-Identifier: Apache-2.0
"""
Checkpoint management module for SDG data generation.

This module provides functionality for saving and loading intermediate results during
data generation, allowing for resumable processing and efficient handling of large
datasets. It includes utilities for identifying missing data and managing checkpoint
files.
"""

# Standard
from typing import Optional, List
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
    
    This class manages the saving and loading of intermediate results during data
    generation, providing mechanisms for:
    - Loading existing checkpoints
    - Identifying missing data that needs to be generated
    - Saving intermediate results at specified intervals
    - Resuming generation from previous checkpoints
    
    Attributes:
        checkpoint_dir (Optional[str]): Directory for saving/loading checkpoints.
            If None, checkpointing is disabled.
        save_freq (Optional[int]): Frequency for saving intermediate checkpoints
            during batch processing. If None, checkpoints are only saved at the end.
    """
    
    def __init__(self, checkpoint_dir: Optional[str] = None, save_freq: Optional[int] = None):
        """
        Initialize the Checkpointer with configuration settings.
        
        Args:
            checkpoint_dir (Optional[str]): Directory to save/load checkpoints.
                If None, checkpointing is disabled.
            save_freq (Optional[int]): Frequency for saving intermediate checkpoints
                during batch processing. If None, checkpoints are only saved at the end.
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
    
    def load_existing_data(self, seed_dataset: Dataset) -> tuple[Dataset, Optional[Dataset]]:
        """
        Load existing checkpoint data and determine what still needs to be generated.
        
        This method:
        1. Attempts to load existing checkpoint data from the checkpoint directory
        2. If checkpoints exist, identifies which rows from the seed dataset
           still need to be generated
        3. If no checkpoints exist, returns the original seed dataset
        
        Args:
            seed_dataset (Dataset): Original input dataset containing the seed data
                that needs to be processed
            
        Returns:
            tuple[Dataset, Optional[Dataset]]: A tuple containing:
                - Dataset: Remaining data that needs to be generated
                - Optional[Dataset]: Previously generated data if checkpoints exist,
                  None otherwise
        """
        if self.checkpoint_dir is None:
            return seed_dataset, None
            
        try:
            # Load existing checkpoints from the checkpoint directory
            pre_generated_data = load_dataset(
                "json", data_dir=self.checkpoint_dir, split="train"
            )
            logger.info(
                f"Loading existing checkpoints from {self.checkpoint_dir}, "
                f"with {pre_generated_data.num_rows} rows"
            )
            
            # Identify which rows from the seed dataset still need to be generated
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
        
        This method:
        1. Identifies common columns between seed and generated datasets
        2. Converts datasets to pandas DataFrames for efficient comparison
        3. Uses tuple-based comparison to identify missing rows
        4. Converts the result back to a Dataset
        
        Args:
            seed_data (Dataset): Original seed dataset containing all rows
            generated_data (Dataset): Previously generated dataset to compare against
            
        Returns:
            Dataset: A new dataset containing only the rows from seed_data that
                are not present in generated_data
        """
        # Get the common columns between the two datasets for comparison
        common_columns = list(
            set(seed_data.column_names) & set(generated_data.column_names)
        )

        # Extract only the common columns for comparison
        seed_data_common = seed_data.select_columns(common_columns)
        generated_data_common = generated_data.select_columns(common_columns)

        # Convert to Pandas DataFrames for efficient row comparison
        seed_df = seed_data_common.to_pandas()
        generated_df = generated_data_common.to_pandas()

        # Identify missing rows using tuple-based comparison
        missing_df = seed_df[
            ~seed_df.apply(tuple, 1).isin(generated_df.apply(tuple, 1))
        ]

        # Convert the result back to a Dataset format
        missing_data = Dataset.from_pandas(missing_df, preserve_index=False)

        return missing_data
    
    def save_intermediate_checkpoint(self, dataset: Dataset) -> None:
        """
        Save intermediate checkpoint data to disk.
        
        This method:
        1. Generates a unique checkpoint ID
        2. Creates a checkpoint file in the checkpoint directory
        3. Saves the dataset in JSONL format
        
        Args:
            dataset (Dataset): Dataset to save as checkpoint
        """
        if self.checkpoint_dir is None:
            return
            
        # Generate a unique identifier for this checkpoint
        checkpoint_id = uuid.uuid4().hex
        checkpoint_file = f"{self.checkpoint_dir}/data_checkpoint_{checkpoint_id}.jsonl"
        logger.info(f"Saving checkpoint to {checkpoint_file}")
        dataset.to_json(checkpoint_file, orient="records", lines=True)
    
    def should_save_checkpoint(self, current_split_index: int) -> bool:
        """
        Determine if a checkpoint should be saved based on save frequency.
        
        This method checks if a checkpoint should be saved based on:
        1. Whether checkpointing is enabled (checkpoint_dir is not None)
        2. Whether a save frequency is specified (save_freq is not None)
        3. Whether the current split index matches the save frequency
        
        Args:
            current_split_index (int): Current split index (0-based) in the batch
                processing sequence
            
        Returns:
            bool: True if a checkpoint should be saved at this point,
                  False otherwise
        """
        if self.save_freq is None or self.checkpoint_dir is None:
            return False
        return (current_split_index + 1) % self.save_freq == 0