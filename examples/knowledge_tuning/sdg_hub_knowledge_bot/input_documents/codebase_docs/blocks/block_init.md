# Documentation: src/sdg_hub/blocks/__init__.py

This file serves as the entry point for the `sdg_hub.blocks` Python package. It makes key block implementations and the `BlockRegistry` conveniently accessible when importing from `sdg_hub.blocks`.

---

## File Purpose

The primary roles of `src/sdg_hub/blocks/__init__.py` are:

1.  **Package Initialization**: It signals to Python that the `blocks` directory should be treated as a package.
2.  **Public API Definition**: It explicitly defines which classes and objects are part of the public API of the `blocks` package through the `__all__` list. This makes it easier to import and use the various block types.
3.  **Convenience Imports**: It imports classes from different modules within the `blocks` package (and `sdg_hub.registry`) into the `sdg_hub.blocks` namespace, allowing users to import them directly (e.g., `from sdg_hub.blocks import LLMBlock`) rather than needing to know the specific submodule (e.g., `from sdg_hub.blocks.llmblock import LLMBlock`).

---

## Code and Explanation

```python
"""Block implementations for SDG Hub.

This package provides various block implementations for data generation, processing, and transformation.
"""

# Local
from .block import Block
from .llmblock import LLMBlock, ConditionalLLMBlock
from .utilblocks import (
    SamplePopulatorBlock,
    SelectorBlock,
    CombineColumnsBlock,
    FlattenColumnsBlock,
    DuplicateColumns,
    RenameColumns,
    SetToMajorityValue,
    FilterByValueBlock,
    IterBlock,
)
from ..registry import BlockRegistry

__all__ = [
    "Block",
    "FilterByValueBlock",
    "IterBlock",
    "LLMBlock",
    "ConditionalLLMBlock",
    "SamplePopulatorBlock",
    "SelectorBlock",
    "CombineColumnsBlock",
    "FlattenColumnsBlock",
    "DuplicateColumns",
    "RenameColumns",
    "SetToMajorityValue",
    "BlockRegistry",
]
```

### Imports

The file performs several imports to gather the necessary classes:

* **Base Block**:
    * `from .block import Block`: Imports the foundational abstract `Block` class from the local `block.py` module. All other blocks inherit from this class.

* **LLM Blocks**:
    * `from .llmblock import LLMBlock, ConditionalLLMBlock`: Imports the `LLMBlock` for standard LLM interactions and `ConditionalLLMBlock` for LLM calls based on conditions, both from the local `llmblock.py` module.

* **Utility Blocks**:
    * `from .utilblocks import ...`: Imports a suite of utility blocks from the local `utilblocks.py` module. These include:
        * `SamplePopulatorBlock`: Populates samples with data.
        * `SelectorBlock`: Selects data based on mapping.
        * `CombineColumnsBlock`: Merges multiple columns.
        * `FlattenColumnsBlock`: Converts wide to long format.
        * `DuplicateColumns`: Creates column copies.
        * `RenameColumns`: Renames dataset columns.
        * `SetToMajorityValue`: Replaces values with the majority.
        * `FilterByValueBlock`: Filters datasets based on column values.
        * `IterBlock`: Allows for iterative processing or execution of a sub-flow.

* **Block Registry**:
    * `from ..registry import BlockRegistry`: Imports the `BlockRegistry` class from the parent package's `registry.py` module. The `BlockRegistry` is used to keep track of all available block types, allowing them to be dynamically loaded by the framework based on YAML configurations.

### Public API (`__all__`)

The `__all__` list explicitly defines the public interface of the `sdg_hub.blocks` package. When a user performs a wildcard import like `from sdg_hub.blocks import *`, only the names listed in `__all__` will be imported. This practice helps prevent namespace pollution and clearly indicates which components are intended for external use.

The exported names are:

* **`Block`**: The base class for all blocks.
* **`FilterByValueBlock`**: For filtering data.
* **`IterBlock`**: For iterative operations.
* **`LLMBlock`**: For standard LLM interactions.
* **`ConditionalLLMBlock`**: For conditional LLM interactions.
* **`SamplePopulatorBlock`**: For populating samples.
* **`SelectorBlock`**: For data selection.
* **`CombineColumnsBlock`**: For merging columns.
* **`FlattenColumnsBlock`**: For data reshaping.
* **`DuplicateColumns`**: For duplicating columns.
* **`RenameColumns`**: For renaming columns.
* **`SetToMajorityValue`**: For setting values to the majority.
* **`BlockRegistry`**: The registry for managing block types.

This `__init__.py` file effectively organizes and exposes the various block functionalities provided by `sdg_hub`.
```