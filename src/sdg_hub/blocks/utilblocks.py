# SPDX-License-Identifier: Apache-2.0
"""Utility blocks for dataset manipulation and transformation.

This module provides various utility blocks for operations like column manipulation,
data population, selection, and transformation of datasets.
"""

# Standard
from typing import Any, Dict, List, Optional, Type

# Third Party
from datasets import Dataset

# Local
from .block import Block
from ..registry import BlockRegistry
from ..logger_config import setup_logger

logger = setup_logger(__name__)



@BlockRegistry.register("SamplePopulatorBlock")
class SamplePopulatorBlock(Block):
    """Block for populating dataset with data from configuration files.

    This block reads data from one or more configuration files and populates a
    dataset with the data. The data is stored in a dictionary, with the keys
    being the names of the configuration files.

    Parameters
    ----------
    block_name : str
        Name of the block.
    config_paths : List[str]
        List of paths to configuration files to load.
    column_name : str
        Name of the column to use as key for populating data.
    post_fix : str, optional
        Suffix to append to configuration filenames, by default "".
    **batch_kwargs : Dict[str, Any]
        Additional keyword arguments for batch processing.
    """

    def __init__(
        self,
        block_name: str,
        config_paths: List[str],
        column_name: str,
        post_fix: str = "",
        **batch_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(block_name=block_name)
        self.configs = {}
        for config in config_paths:
            if post_fix:
                config_name = config.replace(".yaml", f"_{post_fix}.yaml")
            else:
                config_name = config
            config_key = config.split("/")[-1].split(".")[0]
            self.configs[config_key] = self._load_config(config_name)
        self.column_name = column_name
        self.num_procs = batch_kwargs.get("num_procs", 8)

    def _generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new sample by populating it with configuration data.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample to populate with configuration data.

        Returns
        -------
        Dict[str, Any]
            Sample populated with configuration data.
        """
        sample = {**sample, **self.configs[sample[self.column_name]]}
        return sample

    def generate(self, samples: Dataset) -> Dataset:
        """Generate a new dataset with populated configuration data.

        Parameters
        ----------
        samples : Dataset
            Input dataset to populate with configuration data.

        Returns
        -------
        Dataset
            Dataset populated with configuration data.
        """
        samples = samples.map(self._generate, num_proc=self.num_procs)
        return samples



@BlockRegistry.register("CombineColumnsBlock")
class CombineColumnsBlock(Block):
    r"""Block for combining multiple columns into a single column.

    This block concatenates values from multiple columns into a single output column,
    using a specified separator between values.

    Parameters
    ----------
    block_name : str
        Name of the block.
    columns : List[str]
        List of column names to combine.
    output_col : str
        Name of the column to store combined values.
    separator : str, optional
        String to use as separator between combined values, by default "\n\n".
    **batch_kwargs : Dict[str, Any]
        Additional keyword arguments for batch processing.
    """

    def __init__(
        self,
        block_name: str,
        columns: List[str],
        output_col: str,
        separator: str = "\n\n",
        **batch_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(block_name=block_name)
        self.columns = columns
        self.output_col = output_col
        self.separator = separator
        self.num_procs = batch_kwargs.get("num_procs", 8)

    def _generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new sample by combining multiple columns.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample to process.

        Returns
        -------
        Dict[str, Any]
            Sample with combined values stored in output column.
        """
        sample[self.output_col] = self.separator.join(
            [str(sample[col]) for col in self.columns]
        )
        return sample

    def generate(self, samples: Dataset) -> Dataset:
        """Generate a new dataset with combined columns.

        Parameters
        ----------
        samples : Dataset
            Input dataset to process.

        Returns
        -------
        Dataset
            Dataset with combined values stored in output column.
        """
        samples = samples.map(self._generate, num_proc=self.num_procs)
        return samples




@BlockRegistry.register("DuplicateColumns")
class DuplicateColumns(Block):
    """Block for duplicating existing columns with new names.

    This block creates copies of existing columns with new names as specified
    in the columns mapping dictionary.

    Parameters
    ----------
    block_name : str
        Name of the block.
    columns_map : Dict[str, str]
        Dictionary mapping existing column names to new column names.
        Keys are existing column names, values are new column names.
    """

    def __init__(
        self,
        block_name: str,
        columns_map: Dict[str, str],
    ) -> None:
        super().__init__(block_name=block_name)
        self.columns_map = columns_map

    def generate(self, samples: Dataset) -> Dataset:
        """Generate a dataset with duplicated columns.

        Parameters
        ----------
        samples : Dataset
            Input dataset to duplicate columns from.

        Returns
        -------
        Dataset
            Dataset with additional duplicated columns.
        """
        for col_to_dup in self.columns_map:
            samples = samples.add_column(
                self.columns_map[col_to_dup], samples[col_to_dup]
            )
        return samples


@BlockRegistry.register("RenameColumns")
class RenameColumns(Block):
    """Block for renaming columns in a dataset.

    This block renames columns in a dataset according to a mapping dictionary,
    where keys are existing column names and values are new column names.

    Parameters
    ----------
    block_name : str
        Name of the block.
    columns_map : Dict[str, str]
        Dictionary mapping existing column names to new column names.
        Keys are existing column names, values are new column names.
    """

    def __init__(
        self,
        block_name: str,
        columns_map: Dict[str, str],
    ) -> None:
        super().__init__(block_name=block_name)
        self.columns_map = columns_map

    def generate(self, samples: Dataset) -> Dataset:
        """Generate a dataset with renamed columns.

        Parameters
        ----------
        samples : Dataset
            Input dataset to rename columns in.

        Returns
        -------
        Dataset
            Dataset with renamed columns.
        """
        samples = samples.rename_columns(self.columns_map)
        return samples




@BlockRegistry.register("IterBlock")
class IterBlock(Block):
    """Block for iteratively applying another block multiple times.

    This block takes another block type and applies it repeatedly to generate
    multiple samples from the input dataset.

    Parameters
    ----------
    block_name : str
        Name of the block.
    num_iters : int
        Number of times to apply the block.
    block_type : Type[Block]
        The block class to instantiate and apply.
    block_kwargs : Dict[str, Any]
        Keyword arguments to pass to the block constructor.
    **kwargs : Dict[str, Any]
        Additional keyword arguments. Supports:
        - gen_kwargs: Dict[str, Any]
            Arguments to pass to the block's generate method.
    """

    def __init__(
        self,
        block_name: str,
        num_iters: int,
        block_type: Type[Block],
        block_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(block_name)
        self.num_iters = num_iters
        self.block = block_type(**block_kwargs)
        self.gen_kwargs = kwargs.get("gen_kwargs", {})

    def generate(self, samples: Dataset, **gen_kwargs: Dict[str, Any]) -> Dataset:
        """Generate multiple samples by iteratively applying the block.

        Parameters
        ----------
        samples : Dataset
            Input dataset to process.
        **gen_kwargs : Dict[str, Any]
            Additional keyword arguments to pass to the block's generate method.
            These are merged with the gen_kwargs provided at initialization.

        Returns
        -------
        Dataset
            Dataset containing all generated samples from all iterations.
        """
        generated_samples = []
        num_iters = self.num_iters

        for _ in range(num_iters):
            batch_generated = self.block.generate(
                samples, **{**self.gen_kwargs, **gen_kwargs}
            )
            generated_samples.extend(batch_generated)

        return Dataset.from_list(generated_samples)
