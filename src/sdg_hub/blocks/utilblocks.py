# SPDX-License-Identifier: Apache-2.0
"""Utility blocks for dataset manipulation and transformation.

This module provides various utility blocks for operations like column manipulation,
data population, selection, and transformation of datasets.
"""

# Standard
import operator
from typing import Any, Callable, Dict, List, Optional, Type, Union

# Third Party
from datasets import Dataset

# Local
from .block import Block
from ..registry import BlockRegistry
from ..logger_config import setup_logger

logger = setup_logger(__name__)


@BlockRegistry.register("FilterByValueBlock")
class FilterByValueBlock(Block):
    """A block for filtering datasets based on column values.

    This block allows filtering of datasets using various operations (e.g., equals, contains)
    on specified column values, with optional data type conversion
    """

    def __init__(
        self,
        block_name: str,
        filter_column: str,
        filter_value: Union[Any, List[Any]],
        operation: Callable[[Any, Any], bool],
        convert_dtype: Optional[Union[Type[float], Type[int]]] = None,
        **batch_kwargs: Dict[str, Any],
    ) -> None:
        """Initialize a new FilterByValueBlock instance.

        Parameters
        ----------
        block_name : str
            Name of the block.
        filter_column : str
            The name of the column in the dataset to apply the filter on.
        filter_value : Union[Any, List[Any]]
            The value(s) to filter by.
        operation : Callable[[Any, Any], bool]
            A binary operator from the operator module (e.g., operator.eq, operator.contains)
            that takes two arguments and returns a boolean.
        convert_dtype : Optional[Union[Type[float], Type[int]]], optional
            Type to convert the filter column to. Can be either float or int.
            If None, no conversion is performed.
        **batch_kwargs : Dict[str, Any]
            Additional kwargs for batch processing.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the operation is not from the operator module.
        """
        super().__init__(block_name=block_name)
        # Validate that operation is from operator module
        if operation.__module__ != "_operator":
            logger.error("Invalid operation: %s", operation)
            raise ValueError("Operation must be from operator module")
            
        self.value = filter_value if isinstance(filter_value, list) else [filter_value]
        self.column_name = filter_column
        self.operation = operation
        self.convert_dtype = convert_dtype
        self.num_procs = batch_kwargs.get("num_procs", 1)

    def _convert_dtype(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the data type of the filter column.

        Parameters
        ----------
        sample : Dict[str, Any]
            The sample dictionary containing the column to convert.

        Returns
        -------
        Dict[str, Any]
            The sample with converted column value.
        """
        try:
            sample[self.column_name] = self.convert_dtype(sample[self.column_name])
        except ValueError as e:
            logger.error(
                "Error converting dtype: %s, filling with None to be filtered later", e
            )
            sample[self.column_name] = None
        return sample

    def generate(self, samples: Dataset) -> Dataset:
        """Generate filtered dataset based on specified conditions.

        Parameters
        ----------
        samples : Dataset
            The input dataset to filter.

        Returns
        -------
        Dataset
            The filtered dataset.
        """
        if self.convert_dtype:
            samples = samples.map(
                self._convert_dtype,
                num_proc=self.num_procs,
            )

        if self.operation == operator.contains:
            samples = samples.filter(
                lambda x: self.operation(self.value, x[self.column_name]),
                num_proc=self.num_procs,
            )

        samples = samples.filter(
            lambda x: x[self.column_name] is not None,
            num_proc=self.num_procs,
        )

        samples = samples.filter(
            lambda x: any(
                self.operation(x[self.column_name], value) for value in self.value
            ),
            num_proc=self.num_procs,
        )

        return samples


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


@BlockRegistry.register("FlattenColumnsBlock")
class FlattenColumnsBlock(Block):
    """Block for flattening multiple columns into a long format.

    This block transforms a wide dataset format into a long format by melting
    specified columns into rows, creating new variable and value columns.

    Parameters
    ----------
    block_name : str
        Name of the block.
    var_cols : List[str]
        List of column names to be melted into rows.
    value_name : str
        Name of the new column that will contain the values.
    var_name : str
        Name of the new column that will contain the variable names.
    """

    def __init__(
        self,
        block_name: str,
        var_cols: List[str],
        value_name: str,
        var_name: str,
    ) -> None:
        super().__init__(block_name=block_name)
        self.var_cols = var_cols
        self.value_name = value_name
        self.var_name = var_name

    def generate(self, samples: Dataset) -> Dataset:
        """Generate a flattened dataset in long format.

        Parameters
        ----------
        samples : Dataset
            Input dataset to flatten.

        Returns
        -------
        Dataset
            Flattened dataset in long format with new variable and value columns.
        """
        df = samples.to_pandas()
        id_cols = [col for col in samples.column_names if col not in self.var_cols]
        flatten_df = df.melt(
            id_vars=id_cols,
            value_vars=self.var_cols,
            value_name=self.value_name,
            var_name=self.var_name,
        )
        return Dataset.from_pandas(flatten_df)


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


@BlockRegistry.register("SetToMajorityValue")
class SetToMajorityValue(Block):
    """Block for setting all values in a column to the most frequent value.

    This block finds the most common value (mode) in a specified column and
    replaces all values in that column with this majority value.

    Parameters
    ----------
    block_name : str
        Name of the block.
    col_name : str
        Name of the column to set to majority value.
    """

    def __init__(
        self,
        block_name: str,
        col_name: str,
    ) -> None:
        super().__init__(block_name=block_name)
        self.col_name = col_name

    def generate(self, samples: Dataset) -> Dataset:
        """Generate a dataset with column set to majority value.

        Parameters
        ----------
        samples : Dataset
            Input dataset to process.

        Returns
        -------
        Dataset
            Dataset with specified column set to its majority value.
        """
        samples = samples.to_pandas()
        samples[self.col_name] = samples[self.col_name].mode()[0]
        return Dataset.from_pandas(samples)


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
