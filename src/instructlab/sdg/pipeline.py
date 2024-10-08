# SPDX-License-Identifier: Apache-2.0
# Third Party
from datasets import Dataset

# Local
from .logger_config import setup_logger

logger = setup_logger(__name__)


class EmptyDatasetError(Exception):
    pass


class Pipeline:
    def __init__(self, chained_blocks: list) -> None:
        """
        Initialize the Pipeline class with a configuration dictionary.
        config_dict: the run config py or yaml loaded into a dictionary
        """
        # pipeline config is the run configuration that consists of the pipeline steps
        self.chained_blocks = chained_blocks

    def _drop_duplicates(self, dataset, cols):
        """
        Drop duplicates from the dataset based on the columns provided.
        """
        df = dataset.to_pandas()
        df = df.drop_duplicates(subset=cols).reset_index(drop=True)
        return Dataset.from_pandas(df)

    def generate(self, dataset) -> Dataset:
        """
        Generate the dataset by running the pipeline steps.
        dataset: the input dataset
        """
        for block_prop in self.chained_blocks:
            block_type = block_prop["block_type"]
            block_config = block_prop["block_config"]
            drop_columns = block_prop.get("drop_columns", [])
            gen_kwargs = block_prop.get("gen_kwargs", {})
            drop_duplicates_cols = block_prop.get("drop_duplicates", False)
            block = block_type(**block_config)

            logger.debug("------------------------------------\n")
            logger.debug("Running block: %s", block_config["block_name"])
            logger.debug("Input dataset: %s", dataset)

            dataset = block.generate(dataset, **gen_kwargs)

            if len(dataset) == 0:
                raise EmptyDatasetError(
                    f"Pipeline stopped: Empty dataset after running block: {block_config['block_name']}"
                )

            drop_columns_in_ds = [e for e in drop_columns if e in dataset.column_names]
            if drop_columns:
                dataset = dataset.remove_columns(drop_columns_in_ds)

            if drop_duplicates_cols:
                dataset = self._drop_duplicates(dataset, cols=drop_duplicates_cols)

            logger.debug("Output dataset: %s", dataset)
            logger.debug("------------------------------------\n\n")

        return dataset
