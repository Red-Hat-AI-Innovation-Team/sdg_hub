"""
Deprecated Pipeline class for data generation pipelines.

Use the Flow class directly for new code.

This module provides a flexible pipeline architecture that allows for sequential processing
of datasets through multiple blocks, with support for data transformation, column dropping,
and duplicate removal.
"""

# SPDX-License-Identifier: Apache-2.0
# Standard
import warnings
from typing import List, Dict, Any

# Third Party
from datasets import Dataset
from datasets.data_files import EmptyDatasetError

# Local
from .logger_config import setup_logger

logger = setup_logger(__name__)


class Pipeline:
    """A class representing a data generation pipeline.

    This class is deprecated and will be removed in a future version.
    Use the Flow class directly instead.

    Parameters
    ----------
    chained_blocks : List[Dict[str, Any]]
        List of block configurations to execute in sequence.

    Attributes
    ----------
    chained_blocks : List[Dict[str, Any]]
        List of block configurations to execute in sequence.
    """

    def __init__(self, chained_blocks: List[Dict[str, Any]]) -> None:
        """
        Initialize the Pipeline class with a configuration dictionary.
        
        DEPRECATED: This class is deprecated and will be removed in a future version.
        Use the Flow class directly instead.
        """
        warnings.warn(
            "Pipeline class is deprecated and will be removed in a future version. "
            "Use Flow class directly instead of wrapping it with Pipeline.",
            DeprecationWarning,
            stacklevel=2
        )
        # pipeline config is the run configuration that consists of the pipeline steps
        self.chained_blocks = chained_blocks

    def _drop_duplicates(self, dataset: Dataset, cols: List[str]) -> Dataset:
        """Drop duplicates from the dataset based on the columns provided.

        Parameters
        ----------
        dataset : Dataset
            The input dataset.
        cols : List[str]
            Columns to consider for duplicate detection.

        Returns
        -------
        Dataset
            Dataset with duplicates removed.
        """
        df = dataset.to_pandas()
        df = df.drop_duplicates(subset=cols).reset_index(drop=True)
        return Dataset.from_pandas(df)

    def generate(self, dataset: Dataset) -> Dataset:
        """Generate the dataset by running the pipeline steps.
        This method executes each block in sequence, applying transformations and
        handling intermediate data processing steps like column dropping and
        duplicate removal.
        Parameters
        ----------
        dataset : Dataset
            The input dataset to process.

        Returns
        -------
        Dataset
            The processed dataset.

        Raises
        ------
        EmptyDatasetError
            If a block produces an empty dataset.
        """
        for block_prop in self.chained_blocks:
            # Extract block configuration and parameters
            block_type = block_prop["block_type"]
            block_config = block_prop["block_config"]
            drop_columns = block_prop.get("drop_columns", [])
            gen_kwargs = block_prop.get("gen_kwargs", {})
            drop_duplicates_cols = block_prop.get("drop_duplicates", False)
            block = block_type(**block_config)

            # Log block execution details
            logger.debug("------------------------------------\n")
            logger.debug("Running block: %s", block_config["block_name"])
            logger.debug("Input dataset: %s", dataset)

            # Execute the block and process the dataset
            dataset = block.generate(dataset, **gen_kwargs)

            # Validate dataset is not empty after block execution
            if len(dataset) == 0:
                raise EmptyDatasetError(
                    f"Pipeline stopped: Empty dataset after running block: {block_config['block_name']}"
                )

            # Remove specified columns if they exist in the dataset
            drop_columns_in_ds = [e for e in drop_columns if e in dataset.column_names]
            if drop_columns:
                dataset = dataset.remove_columns(drop_columns_in_ds)

            # Remove duplicates if specified
            if drop_duplicates_cols:
                dataset = self._drop_duplicates(dataset, cols=drop_duplicates_cols)

            # Log output dataset details
            logger.debug("Output dataset: %s", dataset)
            logger.debug("------------------------------------\n\n")

        return dataset
