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


@BlockRegistry.register("SelectorBlock")
class SelectorBlock(Block):
    """Block for selecting and mapping values from one column to another.

    This block uses a mapping dictionary to select values from one column and
    store them in a new output column based on a choice column's value.

    Parameters
    ----------
    block_name : str
        Name of the block.
    choice_map : Dict[str, str]
        Dictionary mapping choice values to column names.
    choice_col : str
        Name of the column containing choice values.
    output_col : str
        Name of the column to store selected values.
    **batch_kwargs : Dict[str, Any]
        Additional keyword arguments for batch processing.
    """

    def __init__(
        self,
        block_name: str,
        choice_map: Dict[str, str],
        choice_col: str,
        output_col: str,
        **batch_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(block_name=block_name)
        self.choice_map = choice_map
        self.choice_col = choice_col
        self.output_col = output_col
        self.num_procs = batch_kwargs.get("num_procs", 8)

    def _generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new sample by selecting values based on choice mapping.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample to process.

        Returns
        -------
        Dict[str, Any]
            Sample with selected values stored in output column.
        """
        sample[self.output_col] = sample[self.choice_map[sample[self.choice_col]]]
        return sample

    def generate(self, samples: Dataset) -> Dataset:
        """Generate a new dataset with selected values.

        Parameters
        ----------
        samples : Dataset
            Input dataset to process.

        Returns
        -------
        Dataset
            Dataset with selected values stored in output column.
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






