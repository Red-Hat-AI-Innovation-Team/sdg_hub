# SPDX-License-Identifier: Apache-2.0
# Standard
import operator

# Third Party
from datasets import Dataset

# Local
from .block import Block
from ..registry import BlockRegistry
from ..logger_config import setup_logger

logger = setup_logger(__name__)


@BlockRegistry.register("FilterByValueBlock")
class FilterByValueBlock(Block):
    def __init__(
        self, filter_column, filter_value, operation, convert_dtype=None, **batch_kwargs
    ) -> None:
        """
        Initializes a new instance of the FilterByValueBlock class.

        Parameters:
        - filter_column (str): The name of the column in the dataset to apply the filter on.
        - filter_value (any or list of any): The value(s) to filter by.
        - operation (callable): A function that takes two arguments (column value and filter value) and returns a boolean indicating whether the row should be included in the filtered dataset.
        - convert_dtype (callable, optional): A function to convert the data type of the filter column before applying the filter. Defaults to None.
        - **batch_kwargs: Additional kwargs for batch processing.

        Returns:
        None
        """
        super().__init__(block_name=self.__class__.__name__)
        self.value = filter_value if isinstance(filter_value, list) else [filter_value]
        self.column_name = filter_column
        self.operation = operation
        self.convert_dtype = convert_dtype
        self.num_procs = batch_kwargs.get("num_procs", 1)

    def _convert_dtype(self, sample):
        try:
            sample[self.column_name] = self.convert_dtype(sample[self.column_name])
        except ValueError as e:
            logger.error(
                "Error converting dtype: %s, filling with None to be filtered later", e
            )
            sample[self.column_name] = None
        return sample

    def generate(self, samples) -> Dataset:
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
