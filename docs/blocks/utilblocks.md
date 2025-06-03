# Documentation: src/sdg_hub/blocks/utilblocks.py - Utility Blocks

This module provides a collection of utility "Blocks" designed for common dataset manipulation and transformation tasks within the SDG Hub framework. These blocks handle operations such as filtering, column creation, data reshaping, and iterative processing.

---

## Module Overview

The `utilblocks.py` module offers a suite of pre-built components that can be chained together in SDG Hub flows to perform diverse data preparation and augmentation steps without requiring custom Python coding for these common operations.

* **SPDX-License-Identifier**: `Apache-2.0`

---

## Imports

The module uses:

* **Standard Library**:
    * `operator`: For accessing standard Python operators as functions (e.g., `operator.eq`, `operator.contains`).
    * `typing`: For type hinting (`Any, Callable, Dict, List, Optional, Type, Union`).
* **Third-Party**:
    * `datasets.Dataset`: From the Hugging Face `datasets` library, the primary data structure manipulated by these blocks.
* **Local Project Modules**:
    * `.block.Block`: The base `Block` class from which all utility blocks inherit.
    * `..registry.BlockRegistry`: For registering these utility blocks, making them discoverable by the framework.
    * `..logger_config.setup_logger`: For creating a structured logger instance.

---

## Logger Configuration

A logger is set up for this module:

```python
logger = setup_logger(__name__)
```

---

## Utility Block Implementations

Below are the detailed descriptions of each utility block provided in this module.

### 1. `FilterByValueBlock`

Filters a dataset based on values in a specified column using a given operation.

```python
@BlockRegistry.register("FilterByValueBlock")
class FilterByValueBlock(Block):
    # ... (implementation) ...
```

* **Purpose**: To select or discard rows from a dataset based on a condition applied to a specific column.
* **Registration**: Registered as "FilterByValueBlock".
* **`__init__(...)`**:
    * **Parameters**:
        * `block_name` (str): Name of the block instance.
        * `filter_column` (str): The column to apply the filter on.
        * `filter_value` (Union[Any, List[Any]]): The value or list of values to use for filtering.
        * `operation` (Callable[[Any, Any], bool]): A binary operator function from the `operator` module (e.g., `operator.eq`, `operator.gt`, `operator.contains`).
        * `convert_dtype` (Optional[Union[Type[float], Type[int]]], optional): The data type (e.g., `float`, `int`) to convert the `filter_column` to before applying the operation. Defaults to `None` (no conversion).
        * `**batch_kwargs` (Dict[str, Any]): Contains `num_procs` for parallel processing (defaults to 1).
    * **Logic**:
        * Validates that the `operation` is from the Python `operator` module.
        * Stores parameters and normalizes `filter_value` to always be a list.
* **`_convert_dtype(self, sample: Dict[str, Any]) -> Dict[str, Any]`**:
    * **Purpose**: Helper method to convert the data type of the `filter_column` in a given sample.
    * **Logic**: Tries to convert `sample[self.column_name]` to `self.convert_dtype`. If `ValueError` occurs, logs an error and sets the column value to `None` (to be filtered out later).
* **`generate(self, samples: Dataset) -> Dataset`**:
    * **Purpose**: Applies the filtering logic to the input dataset.
    * **Logic**:
        1.  **Type Conversion**: If `self.convert_dtype` is set, it maps `self._convert_dtype` over the dataset using `samples.map()` with `num_proc=self.num_procs`.
        2.  **`operator.contains` Special Case**: If `self.operation` is `operator.contains`, it applies a filter: `lambda x: self.operation(self.value, x[self.column_name])`. This checks if the value in `x[self.column_name]` is present in the list `self.value`.
        3.  **Filter `None`s**: Filters out any rows where `x[self.column_name]` is `None`. This is important if type conversion failed for some rows.
        4.  **Main Operation Filter**: Applies the primary filter. For each row `x` and for each `value` in `self.value` (the list of filter criteria), it checks `self.operation(x[self.column_name], value)`. The row is kept if *any* of these checks are true (due to `any()`).
    * **Returns**: `Dataset` - The filtered dataset.

---

### 2. `SamplePopulatorBlock`

Populates dataset samples with data loaded from specified configuration files.

```python
@BlockRegistry.register("SamplePopulatorBlock")
class SamplePopulatorBlock(Block):
    # ... (implementation) ...
```

* **Purpose**: To enrich dataset samples by merging data from external YAML configuration files. The specific configuration to merge is chosen based on a key column in the sample.
* **Registration**: Registered as "SamplePopulatorBlock".
* **`__init__(...)`**:
    * **Parameters**:
        * `block_name` (str): Name of the block instance.
        * `config_paths` (List[str]): A list of paths to YAML configuration files.
        * `column_name` (str): The name of the column in the dataset whose value will be used as a key to select which loaded configuration to merge with the sample.
        * `post_fix` (str, optional): A suffix to append to configuration filenames before loading (e.g., to differentiate versions). Defaults to `""`.
        * `**batch_kwargs` (Dict[str, Any]): Contains `num_procs` for parallel processing (defaults to 8).
    * **Logic**:
        * Loads each configuration file specified in `config_paths` (potentially modified by `post_fix`) using `self._load_config()`.
        * Stores these loaded configurations in a dictionary `self.configs`, keyed by the base name of the configuration file (e.g., `'my_config'` from path `'/path/to/my_config.yaml'`).
* **`_generate(self, sample: Dict[str, Any]) -> Dict[str, Any]`**:
    * **Purpose**: Helper method to populate a single sample.
    * **Logic**: Retrieves the configuration data from `self.configs` using the value of `sample[self.column_name]` as the key. It then merges this configuration dictionary into the sample dictionary.
* **`generate(self, samples: Dataset) -> Dataset`**:
    * **Purpose**: Applies the population logic to the entire dataset.
    * **Logic**: Uses `samples.map(self._generate, num_proc=self.num_procs)` to process each sample.
    * **Returns**: `Dataset` - The dataset with samples populated from the configurations.

---

### 3. `SelectorBlock`

Selects values from one column and maps them to an output column based on a choice made in another column.

```python
@BlockRegistry.register("SelectorBlock")
class SelectorBlock(Block):
    # ... (implementation) ...
```

* **Purpose**: To conditionally pick a value from one of several source columns based on a "choice" column and place it into a designated output column.
* **Registration**: Registered as "SelectorBlock".
* **`__init__(...)`**:
    * **Parameters**:
        * `block_name` (str): Name of the block instance.
        * `choice_map` (Dict[str, str]): A dictionary where keys are the possible values in `choice_col`, and values are the names of the columns from which to pick the data.
        * `choice_col` (str): The name of the column in the dataset that contains the "choice" value.
        * `output_col` (str): The name of the new column where the selected value will be stored.
        * `**batch_kwargs` (Dict[str, Any]): Contains `num_procs` for parallel processing (defaults to 8).
* **`_generate(self, sample: Dict[str, Any]) -> Dict[str, Any]`**:
    * **Purpose**: Helper method to process a single sample.
    * **Logic**:
        1.  Looks up the value of `sample[self.choice_col]` in `self.choice_map` to get the name of the source column.
        2.  Retrieves the value from `sample[<source_column_name>]`.
        3.  Assigns this retrieved value to `sample[self.output_col]`.
* **`generate(self, samples: Dataset) -> Dataset`**:
    * **Purpose**: Applies the selection logic to the entire dataset.
    * **Logic**: Uses `samples.map(self._generate, num_proc=self.num_procs)`.
    * **Returns**: `Dataset` - The dataset with the new `output_col` populated.

---

### 4. `CombineColumnsBlock`

Combines content from multiple columns into a single new column.

```python
@BlockRegistry.register("CombineColumnsBlock")
class CombineColumnsBlock(Block):
    # ... (implementation) ...
```

* **Purpose**: To concatenate the string representations of values from several specified columns into one target column, using a defined separator.
* **Registration**: Registered as "CombineColumnsBlock".
* **`__init__(...)`**:
    * **Parameters**:
        * `block_name` (str): Name of the block instance.
        * `columns` (List[str]): A list of names of the columns whose values are to be combined.
        * `output_col` (str): The name of the new column where the combined string will be stored.
        * `separator` (str, optional): The string to use as a separator between the values from different columns. Defaults to `"\n\n"`.
        * `**batch_kwargs` (Dict[str, Any]): Contains `num_procs` for parallel processing (defaults to 8).
* **`_generate(self, sample: Dict[str, Any]) -> Dict[str, Any]`**:
    * **Purpose**: Helper method to combine columns for a single sample.
    * **Logic**: Retrieves values from `sample` for each column in `self.columns`, joins them using `self.separator`, and stores the result in `sample[self.output_col]`.
* **`generate(self, samples: Dataset) -> Dataset`**:
    * **Purpose**: Applies the column combination logic to the entire dataset.
    * **Logic**: Uses `samples.map(self._generate, num_proc=self.num_procs)`.
    * **Returns**: `Dataset` - The dataset with the new combined `output_col`.

---

### 5. `FlattenColumnsBlock`

Transforms specified columns from a wide format to a long format (melts columns).

```python
@BlockRegistry.register("FlattenColumnsBlock")
class FlattenColumnsBlock(Block):
    # ... (implementation) ...
```

* **Purpose**: To reshape the dataset by unpivoting specified columns into two new columns: one for the original column names (variable names) and one for their corresponding values.
* **Registration**: Registered as "FlattenColumnsBlock".
* **`__init__(...)`**:
    * **Parameters**:
        * `block_name` (str): Name of the block instance.
        * `var_cols` (List[str]): A list of column names that will be "melted" or unpivoted.
        * `value_name` (str): The name for the new column that will store the values from `var_cols`.
        * `var_name` (str): The name for the new column that will store the names of the columns from `var_cols`.
* **`generate(self, samples: Dataset) -> Dataset`**:
    * **Purpose**: Performs the flattening operation.
    * **Logic**:
        1.  Converts the input `Dataset` to a pandas DataFrame using `samples.to_pandas()`.
        2.  Identifies `id_cols` (columns that are not in `var_cols` and will be preserved).
        3.  Uses `df.melt()` to perform the unpivoting operation.
        4.  Converts the resulting pandas DataFrame back to a `Dataset` using `Dataset.from_pandas()`.
    * **Returns**: `Dataset` - The reshaped dataset in a long format.

---

### 6. `DuplicateColumns`

Duplicates existing columns in a dataset under new names.

```python
@BlockRegistry.register("DuplicateColumns")
class DuplicateColumns(Block):
    # ... (implementation) ...
```

* **Purpose**: To create copies of one or more existing columns.
* **Registration**: Registered as "DuplicateColumns".
* **`__init__(...)`**:
    * **Parameters**:
        * `block_name` (str): Name of the block instance.
        * `columns_map` (Dict[str, str]): A dictionary where keys are the names of existing columns to duplicate, and values are the names for the new duplicated columns.
* **`generate(self, samples: Dataset) -> Dataset`**:
    * **Purpose**: Adds the duplicated columns to the dataset.
    * **Logic**: Iterates through `self.columns_map`. For each `col_to_dup` (existing column name) and its corresponding new name `self.columns_map[col_to_dup]`, it adds a new column to the dataset using `samples.add_column()`, copying the data from the existing column.
    * **Returns**: `Dataset` - The dataset with the added duplicated columns.

---

### 7. `RenameColumns`

Renames existing columns in a dataset.

```python
@BlockRegistry.register("RenameColumns")
class RenameColumns(Block):
    # ... (implementation) ...
```

* **Purpose**: To change the names of one or more columns in the dataset.
* **Registration**: Registered as "RenameColumns".
* **`__init__(...)`**:
    * **Parameters**:
        * `block_name` (str): Name of the block instance.
        * `columns_map` (Dict[str, str]): A dictionary where keys are the current column names and values are the new desired column names.
* **`generate(self, samples: Dataset) -> Dataset`**:
    * **Purpose**: Applies the renaming operations.
    * **Logic**: Uses the `samples.rename_columns(self.columns_map)` method directly.
    * **Returns**: `Dataset` - The dataset with columns renamed.

---

### 8. `SetToMajorityValue`

Sets all values in a specified column to its most frequent value (mode).

```python
@BlockRegistry.register("SetToMajorityValue")
class SetToMajorityValue(Block):
    # ... (implementation) ...
```

* **Purpose**: To normalize a column by replacing all its values with its statistical mode. This can be useful for imputation or standardization in some contexts.
* **Registration**: Registered as "SetToMajorityValue".
* **`__init__(...)`**:
    * **Parameters**:
        * `block_name` (str): Name of the block instance.
        * `col_name` (str): The name of the column to be processed.
* **`generate(self, samples: Dataset) -> Dataset`**:
    * **Purpose**: Finds the mode of the specified column and updates all values in that column to this mode.
    * **Logic**:
        1.  Converts the input `Dataset` to a pandas DataFrame.
        2.  Calculates the mode of `samples[self.col_name]` using `samples[self.col_name].mode()[0]` (takes the first mode if multiple exist).
        3.  Assigns this mode value to the entire column `samples[self.col_name]`.
        4.  Converts the modified pandas DataFrame back to a `Dataset`.
    * **Returns**: `Dataset` - The dataset with the specified column's values all set to its majority value.

---

### 9. `IterBlock`

Iteratively applies another specified block multiple times to generate an augmented dataset.

```python
@BlockRegistry.register("IterBlock")
class IterBlock(Block):
    # ... (implementation) ...
```

* **Purpose**: To repeat the data generation process of another block (`block_type`) for a specified number of iterations (`num_iters`), accumulating the results. This is useful for tasks like generating multiple variations from the same input or augmenting dataset size.
* **Registration**: Registered as "IterBlock".
* **`__init__(...)`**:
    * **Parameters**:
        * `block_name` (str): Name of the `IterBlock` instance.
        * `num_iters` (int): The number of times the inner block should be applied.
        * `block_type` (Type[Block]): The class of the block to be instantiated and applied iteratively (e.g., `LLMBlock`).
        * `block_kwargs` (Dict[str, Any]): Keyword arguments to be passed to the constructor of the `block_type`.
        * `**kwargs` (Dict[str, Any]): Additional keyword arguments, specifically supports `gen_kwargs` (a dictionary) which are arguments to be passed to the `generate` method of the inner block during initialization.
    * **Logic**:
        * Instantiates the inner block (`self.block`) using `block_type(**block_kwargs)`.
        * Stores `num_iters` and `gen_kwargs` (from `kwargs`).
* **`generate(self, samples: Dataset, **gen_kwargs: Dict[str, Any]) -> Dataset`**:
    * **Purpose**: Executes the iterative generation process.
    * **Parameters**:
        * `samples` (Dataset): The input dataset to be passed to the inner block in each iteration.
        * `**gen_kwargs` (Dict[str, Any]): Additional keyword arguments for the inner block's `generate` method. These are merged with `gen_kwargs` provided during `IterBlock`'s initialization, with these runtime `gen_kwargs` taking precedence.
    * **Logic**:
        1.  Initializes an empty list `generated_samples`.
        2.  Loops `self.num_iters` times.
        3.  In each iteration, calls the `generate` method of `self.block` (the inner block), passing the input `samples` and the merged `gen_kwargs`.
        4.  Extends `generated_samples` with the results from the inner block's generation for that iteration.
        5.  After all iterations, creates a new `Dataset` from the accumulated `generated_samples` list.
    * **Returns**: `Dataset` - A single dataset containing all samples generated across all iterations.

This set of utility blocks provides a robust toolkit for a wide range of data manipulation tasks within `sdg_hub`.
```