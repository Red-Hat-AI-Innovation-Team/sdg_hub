# SPDX-License-Identifier: Apache-2.0

"""Module containing the AddStaticValue block for adding constant values to dataset columns."""

# Standard
from typing import Any, Dict

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.blocks import Block, BlockRegistry


@BlockRegistry.register("AddStaticValue")
class AddStaticValue(Block):
    """A custom block that adds a static value to a specified column in a dataset.

    This block is designed to populate a new or existing column in a dataset with a constant
    value. It's useful for adding metadata, labels, or any other static information to
    your dataset entries.

    Examples
    --------
    >>> block = AddStaticValue("add_label", "label", "positive")
    >>> dataset = block.generate(input_dataset)
    """

    def __init__(self, block_name: str, column_name: str, static_value: str) -> None:
        """Initialize the AddStaticValue block.

        Parameters
        ----------
        block_name : str
            The name of this block instance
        column_name : str
            The name of the column to populate with the static value
        static_value : str
            The constant value to be added to the specified column
        """
        super().__init__(block_name)
        self.column_name = column_name
        self.static_value = static_value

    # Using a static method to avoid serializing self when using multiprocessing
    @staticmethod
    def _map_populate_column(
        samples: Dataset, column_name: str, static_value: str, num_proc: int = 1
    ) -> Dataset:
        """Map function to populate a column with a static value.

        Parameters
        ----------
        samples : Dataset
            The input dataset to modify
        column_name : str
            The name of the column to populate
        static_value : str
            The constant value to add to the column
        num_proc : int, optional
            Number of processes to use for parallel processing, by default 1

        Returns
        -------
        Dataset
            The modified dataset with the new column populated
        """

        def populate_column(sample: Dict[str, Any]) -> Dict[str, Any]:
            sample[column_name] = static_value
            return sample

        return samples.map(populate_column, num_proc=num_proc)

    def generate(self, samples: Dataset) -> Dataset:
        """Generate a new dataset with the static value added to the specified column.

        Parameters
        ----------
        samples : Dataset
            The input dataset to modify

        Returns
        -------
        Dataset
            The modified dataset with the new column populated with the static value
        """
        samples = self._map_populate_column(
            samples, self.column_name, self.static_value
        )
        return samples
