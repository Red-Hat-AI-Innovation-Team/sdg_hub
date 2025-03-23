# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import traceback
import uuid

# Third Party
from datasets import Dataset, load_dataset
from datasets.data_files import EmptyDatasetError
from tqdm import tqdm

# Local
from .logger_config import setup_logger
from .pipeline import Pipeline
from .utils.datautils import safe_concatenate_datasets


logger = setup_logger(__name__)


class SDG:
    def __init__(
        self, pipelines: List[Pipeline], num_workers=1, batch_size=None, save_freq=None
    ) -> None:
        self.pipelines = pipelines
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.save_freq = save_freq

    def _split_dataset(self, dataset: Dataset, batch_size: int) -> List[Dataset]:
        """Split the dataset into smaller batches."""
        total_size = len(dataset)
        num_batches = (total_size + batch_size - 1) // batch_size

        batches = [
            (i * batch_size, min((i + 1) * batch_size, total_size))
            for i in tqdm(range(num_batches))
        ]

        return batches

    def _get_missing_data(self, seed_data, generated_data):
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

    def _save_intermediate_checkpoint(self, dataset, checkpoint_dir):
        checkpoint_id = uuid.uuid4().hex
        checkpoint_file = f"{checkpoint_dir}/data_checkpoint_{checkpoint_id}.jsonl"
        logger.info(f"Saving checkpoint to {checkpoint_file}")
        dataset.to_json(checkpoint_file, orient="records", lines=True)

    @staticmethod
    def _generate_data(pipelines, input_split, ds, i=None):
        logger.info(f"Processing split {i}")
        input_split = ds.select(range(input_split[0], input_split[1]))
        try:
            for pipeline in pipelines:
                input_split = pipeline.generate(input_split)
            return input_split
        except Exception as e:
            logger.error(f"Error processing split {i}: {e}")
            traceback.print_exc()
            return None

    def generate(self, dataset: Dataset, checkpoint_dir=None) -> Dataset:
        # check if checkpoint_dir exists
        pre_generated_data = []
        if checkpoint_dir is not None:
            try:
                # check if there are any existing checkpoints
                pre_generated_data = load_dataset(
                    "json", data_dir=checkpoint_dir, split="train"
                )
                logger.info(
                    f"Loading existing checkpoints from {checkpoint_dir}, with {pre_generated_data.num_rows} rows"
                )
                seed_data = self._get_missing_data(dataset, pre_generated_data)
                if seed_data.num_rows == 0:
                    logger.info(
                        f"All seed data has been generated, no missing rows found, returning data from {checkpoint_dir}"
                    )
                    return pre_generated_data
                logger.info(f"Found {seed_data.num_rows} missing rows in the dataset")

            except EmptyDatasetError:
                logger.info(
                    f"No existing checkpoints found in {checkpoint_dir}, generating from scratch"
                )
                seed_data = dataset

        else:
            seed_data = dataset

        if not self.batch_size:
            # If batch size is not provided, generate the dataset in a single pass
            generated_dataset = seed_data
            # generated_data is initialized with seed_data, and it gets updated with each pipeline
            for pipeline in self.pipelines:
                generated_dataset = pipeline.generate(seed_data)
            return generated_dataset
        
        logger.info("Splitting the dataset into smaller batches")
        input_splits = (
            self._split_dataset(seed_data, self.batch_size)
            if self.batch_size
            else [seed_data]
        )
        logger.info(
            f"Generating dataset with {len(input_splits)} splits, batch size {self.batch_size}, and {self.num_workers} workers"
        )

        generated_data = [pre_generated_data] if pre_generated_data else []
        last_saved_split_index = 0  # To track the last saved split

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_data, self.pipelines, input_split, seed_data, i
                )
                for i, input_split in enumerate(input_splits)
            ]

            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
                generated_data_split = future.result()  # Ensure each future completes

                if generated_data_split:
                    generated_data.append(generated_data_split)
                    logger.info(f"Finished future processing split {i} \n\n")
                    if self.save_freq and (i + 1) % self.save_freq == 0:
                        # Save only the new splits since the last checkpoint
                        new_splits = generated_data[last_saved_split_index : i + 1]
                        checkpoint_dataset = safe_concatenate_datasets(new_splits)
                        # check if checkpoint_dataset is not None
                        if checkpoint_dataset:
                            self._save_intermediate_checkpoint(
                                checkpoint_dataset, checkpoint_dir
                            )

                            last_saved_split_index = i + 1

        generated_dataset = safe_concatenate_datasets(generated_data)

        return generated_dataset
