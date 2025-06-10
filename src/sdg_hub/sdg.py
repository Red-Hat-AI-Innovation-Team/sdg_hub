# SPDX-License-Identifier: Apache-2.0

"""Synthetic Data Generator (SDG) module for managing data generation flows."""

# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
import traceback

# Third Party
from datasets import Dataset
from tqdm import tqdm

# Local
from .checkpointer import Checkpointer
from .flow import Flow
from .logger_config import setup_logger
from .utils.datautils import safe_concatenate_datasets

logger = setup_logger(__name__)


class SDG:
    """Synthetic Data Generator class.

    This class manages the generation of synthetic data using one or more
    data generation flows.

    Parameters
    ----------
    flows : List[Flow]
        List of flows to execute.
    num_workers : int, optional
        Number of worker threads to use, by default 1
    batch_size : Optional[int], optional
        Size of batches to process, by default None
    save_freq : Optional[int], optional
        Frequency of checkpoint saves, by default None

    Attributes
    ----------
    flows : List[Flow]
        List of flows to execute.
    num_workers : int
        Number of worker threads to use.
    batch_size : Optional[int]
        Size of batches to process.
    save_freq : Optional[int]
        Frequency of checkpoint saves.
    """

    def __init__(
        self,
        flows: List[Flow],
        num_workers: int = 1,
        batch_size: Optional[int] = None,
        save_freq: Optional[int] = None,
    ) -> None:
        self.flows = flows
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.save_freq = save_freq

    def _split_dataset(
        self, dataset: Dataset, batch_size: int
    ) -> List[Tuple[int, int]]:
        """Split the dataset into smaller batches.

        Parameters
        ----------
        dataset : Dataset
            The dataset to split.
        batch_size : int
            Size of each batch.

        Returns
        -------
        List[Tuple[int, int]]
            List of (start, end) indices for each batch.
        """
        total_size = len(dataset)
        num_batches = (total_size + batch_size - 1) // batch_size

        batches = [
            (i * batch_size, min((i + 1) * batch_size, total_size))
            for i in tqdm(range(num_batches))
        ]

        return batches

    @staticmethod
    def _generate_data(
        flows: List[Flow],
        input_split: Tuple[int, int],
        ds: Dataset,
        i: Optional[int] = None,
    ) -> Optional[Dataset]:
        """Generate data for a single split using the provided flows.

        Parameters
        ----------
        flows : List[Flow]
            List of flows to execute.
        input_split : Tuple[int, int]
            (start, end) indices for the current split.
        ds : Dataset
            The full input dataset.
        i : Optional[int], optional
            Split index for logging, by default None

        Returns
        -------
        Optional[Dataset]
            Generated dataset for the split, or None if generation failed.
        """
        logger.info(f"Processing split {i}")
        input_split = ds.select(range(input_split[0], input_split[1]))
        try:
            for flow in flows:
                input_split = flow.generate(input_split)
            return input_split
        except Exception as e:
            logger.error(f"Error processing split {i}: {e}")
            traceback.print_exc()
            return None

    def generate(
        self, dataset: Dataset, checkpoint_dir: Optional[str] = None
    ) -> Dataset:
        """Generate synthetic data using the configured flows.

        Parameters
        ----------
        dataset : Dataset
            The input dataset to process.
        checkpoint_dir : Optional[str], optional
            Directory to save checkpoints, by default None

        Returns
        -------
        Dataset
            The generated dataset.

        Notes
        -----
        If checkpoint_dir is provided, the generation process can be resumed
        from the last checkpoint in case of interruption.
        """
        # Initialize checkpointer
        checkpointer = Checkpointer(checkpoint_dir, self.save_freq)

        # Load existing checkpoints and determine missing data
        seed_data, pre_generated_data = checkpointer.load_existing_data(dataset)

        # If all data has been generated, return the pre-generated data
        if seed_data.num_rows == 0 and pre_generated_data is not None:
            return pre_generated_data

        if not self.batch_size:
            # If batch size is not provided, generate the dataset in a single pass
            generated_dataset = seed_data
            # generated_data is initialized with seed_data, and it gets updated with each flow
            for flow in self.flows:
                generated_dataset = flow.generate(generated_dataset)
            return generated_dataset

        logger.info("Splitting the dataset into smaller batches")
        input_splits = self._split_dataset(seed_data, self.batch_size)
        logger.info(
            f"Generating dataset with {len(input_splits)} splits, "
            f"batch size {self.batch_size}, and {self.num_workers} workers"
        )

        generated_data = [pre_generated_data] if pre_generated_data else []
        last_saved_split_index = 0  # To track the last saved split

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_data, self.flows, input_split, seed_data, i
                )
                for i, input_split in enumerate(input_splits)
            ]

            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
                generated_data_split = future.result()  # Ensure each future completes

                if generated_data_split:
                    generated_data.append(generated_data_split)
                    logger.info(f"Finished future processing split {i} \n\n")

                    # Use checkpointer to handle intermediate saves
                    if checkpointer.should_save_checkpoint(i):
                        # Save only the new splits since the last checkpoint
                        new_splits = generated_data[last_saved_split_index : i + 1]
                        checkpoint_dataset = safe_concatenate_datasets(new_splits)
                        # check if checkpoint_dataset is not None
                        if checkpoint_dataset:
                            checkpointer.save_intermediate_checkpoint(
                                checkpoint_dataset
                            )
                            last_saved_split_index = i + 1

        generated_dataset = safe_concatenate_datasets(generated_data)

        return generated_dataset
