# SPDX-License-Identifier: Apache-2.0
# Third Party
from datasets import Dataset

# Local
from .block import Block
from ..registry import BlockRegistry
from ..logger_config import setup_logger

logger = setup_logger(__name__)


@BlockRegistry.register("SamplePopulatorBlock")
class SamplePopulatorBlock(Block):
    def __init__(self, config_paths, column_name, post_fix="", **batch_kwargs) -> None:
        super().__init__(
            block_name=self.__class__.__name__
        )  # Call the base class's __init__
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

    def _generate(self, sample) -> dict:
        sample = {**sample, **self.configs[sample[self.column_name]]}
        return sample

    def generate(self, samples) -> Dataset:
        samples = samples.map(self._generate, num_proc=self.num_procs)
        return samples


@BlockRegistry.register("SelectorBlock")
class SelectorBlock(Block):
    def __init__(self, choice_map, choice_col, output_col, **batch_kwargs) -> None:
        super().__init__(block_name=self.__class__.__name__)
        self.choice_map = choice_map
        self.choice_col = choice_col
        self.output_col = output_col
        self.num_procs = batch_kwargs.get("num_procs", 8)

    def _generate(self, sample) -> dict:
        sample[self.output_col] = sample[self.choice_map[sample[self.choice_col]]]
        return sample

    def generate(self, samples: Dataset) -> Dataset:
        samples = samples.map(self._generate, num_proc=self.num_procs)
        return samples


@BlockRegistry.register("CombineColumnsBlock")
class CombineColumnsBlock(Block):
    def __init__(self, columns, output_col, separator="\n\n", **batch_kwargs) -> None:
        super().__init__(block_name=self.__class__.__name__)
        self.columns = columns
        self.output_col = output_col
        self.separator = separator
        self.num_procs = batch_kwargs.get("num_procs", 8)

    def _generate(self, sample) -> dict:
        sample[self.output_col] = self.separator.join(
            [sample[col] for col in self.columns]
        )
        return sample

    def generate(self, samples: Dataset) -> Dataset:
        samples = samples.map(self._generate, num_proc=self.num_procs)
        return samples


@BlockRegistry.register("FlattenColumnsBlock")
class FlattenColumnsBlock(Block):
    def __init__(self, block_name: str, var_cols: list, value_name: str, var_name: str) -> None:
        super().__init__(block_name=block_name)
        self.var_cols = var_cols
        self.value_name = value_name
        self.var_name = var_name

    def generate(self, samples: Dataset) -> Dataset:
        df = samples.to_pandas()
        id_cols = [col for col in samples.column_names if col not in self.var_cols]
        flatten_df = df.melt(id_vars=id_cols, 
                             value_vars=self.var_cols, 
                             value_name=self.value_name,
                             var_name=self.var_name)
        
        return Dataset.from_pandas(flatten_df)


@BlockRegistry.register("DuplicateColumns")
class DuplicateColumns(Block):
    def __init__(self, block_name: str, columns_map: dict) -> None:
        """Create duplicate of columns specified in column map.

        Args:
            columns_map (dict): mapping of existing column to new column names
        """
        super().__init__(block_name=block_name)
        self.columns_map = columns_map    
    
    
    def generate(self, samples: Dataset):
        for col_to_dup in self.columns_map:
            samples = samples.add_column(self.columns_map[col_to_dup], samples[col_to_dup])
        return samples


@BlockRegistry.register("RenameColumns")
class RenameColumns(Block):
    def __init__(self, block_name: str, columns_map: dict) -> None:
        """Rename dataset columns.

        Args:
            columns_map (dict): mapping of existing column to new column names
        """
        self.columns_map = columns_map
        super().__init__(block_name=block_name)
    
    
    def generate(self, samples: Dataset):
        samples = samples.rename_columns(self.columns_map)
        return samples


@BlockRegistry.register("SetToMajorityValue")
class SetToMajorityValue(Block):
    def __init__(self, block_name: str, col_name) -> None:
        self.col_name = col_name
        super().__init__(block_name)
    
    def generate(self, samples: Dataset):
        samples = samples.to_pandas()
        samples[self.col_name] = samples[self.col_name].mode()[0]
        return Dataset.from_pandas(samples)