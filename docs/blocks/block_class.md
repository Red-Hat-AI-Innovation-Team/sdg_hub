# Documentation: src/sdg_hub/blocks/block.py - Base Block Class

This module defines the abstract base class (`Block`) for all computational units within the Synthetic Data Generation (SDG) Hub framework. It provides core functionalities common to all blocks, such as configuration loading and input validation, particularly for blocks that utilize templating.

---

## Module Overview

The `block.py` module is central to the `sdg_hub`'s architecture. It ensures that all specialized blocks adhere to a common interface and inherit baseline functionalities.

* **SPDX-License-Identifier**: `Apache-2.0` (Indicates the licensing terms for this file).

---

## Imports

The module utilizes several standard Python libraries, third-party libraries, and local project modules:

* **Standard Library**:
    * `abc.ABC`: Used to declare abstract base classes.
    * `collections.ChainMap`: To combine multiple dictionaries for template rendering, allowing defaults or overrides.
    * `typing.Any, Dict, Optional`: For type hinting.
* **Third-Party**:
    * `jinja2.Template, UndefinedError`: For handling Jinja2 templating and associated errors.
    * `yaml`: For parsing YAML configuration files.
* **Local Project Modules**:
    * `..registry.BlockRegistry`: Imports the `BlockRegistry` to allow block classes to register themselves, making them discoverable by the framework.
    * `..logger_config.setup_logger`: Imports a function to set up a structured logger for the module.

---

## Logger Configuration

A logger instance is initialized for this module to facilitate consistent logging practices:

```python
logger = setup_logger(__name__)
```

---

## The `Block` Class

The `Block` class is the cornerstone of the framework's modular design.

```python
@BlockRegistry.register("Block")
class Block(ABC):
    """Base abstract class for all blocks in the system.

    This class provides common functionality for block validation and configuration loading.
    All specific block implementations should inherit from this class.
    """

    def __init__(self, block_name: str) -> None:
        self.block_name = block_name

    @staticmethod
    def _validate(prompt_template: Template, input_dict: Dict[str, Any]) -> bool:
        # ... (implementation details below)
        pass

    def _load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        # ... (implementation details below)
        pass
```

### 1. Class Definition and Purpose

* `class Block(ABC)`: Defines `Block` as an abstract base class. This means `Block` itself is not intended to be instantiated directly. Instead, it serves as a blueprint for concrete block implementations (e.g., `LLMBlock`, `FilterByValueBlock`).
* **Registration**:
    * `@BlockRegistry.register("Block")`: This class decorator registers the `Block` class with the `BlockRegistry` under the name "Block". While this base class itself might not be instantiated from a YAML flow by this name, this mechanism is typically used by its subclasses to register themselves under their specific `block_type` names.

### 2. `__init__(self, block_name: str) -> None`

* **Purpose**: The constructor initializes a new block instance.
* **Parameters**:
    * `block_name` (str): A unique name assigned to this instance of the block within a flow. This is useful for logging, checkpointing, and potentially referencing outputs from this block.
* **Action**: Sets the `self.block_name` attribute.

```python
    def __init__(self, block_name: str) -> None:
        self.block_name = block_name
```

### 3. `_validate(prompt_template: Template, input_dict: Dict[str, Any]) -> bool`

This is a static method designed to validate input data against a Jinja2 template.

* **Purpose**: Checks if all the variables expected by a `prompt_template` (commonly used in LLM blocks) are present in the `input_dict`.
* **Static Method**: `@staticmethod` means this method can be called on the class itself (`Block._validate(...)`) without needing an instance, or on an instance.
* **Parameters**:
    * `prompt_template` (jinja2.Template): The Jinja2 template object that contains variable placeholders (e.g., `{{document}}`).
    * `input_dict` (Dict[str, Any]): A dictionary containing the actual data intended to be rendered into the template.
* **Returns**:
    * `bool`: `True` if all variables required by the template are found in `input_dict`. `False` if any variable is missing, logging an error in that case.
* **Implementation Details**:
    * It defines an inner class `Default(dict)` which overrides the `__missing__` method to explicitly raise a `KeyError`.
    * It attempts to render the `prompt_template` using a `ChainMap`. `ChainMap` links `input_dict` and an instance of `Default`. This setup ensures that if a variable is not found in `input_dict`, the custom `__missing__` method is triggered.
    * If `jinja2.UndefinedError` (a subclass of `KeyError` in this context due to the custom `Default` class, or directly if `Default` weren't used and strictundefined was set) is caught during rendering, it means a required variable was missing. The error is logged, and the method returns `False`.
    * If rendering is successful, it returns `True`.

```python
    @staticmethod
    def _validate(prompt_template: Template, input_dict: Dict[str, Any]) -> bool:
        """Validate the input data for this block.

        This method validates whether all required variables in the Jinja template are provided in the input_dict.
        # ... (rest of docstring)
        """

        class Default(dict):
            def __missing__(self, key: str) -> None:
                raise KeyError(key)

        try:
            # Try rendering the template with the input_dict
            prompt_template.render(ChainMap(input_dict, Default()))
            return True
        except UndefinedError as e: # UndefinedError is raised by Jinja for missing variables
            logger.error(f"Missing key: {e}")
            return False
```
*Note: The `Default` class combined with `ChainMap` is a specific way to detect missing keys. Jinja2's `StrictUndefined` option could also be used for similar purposes.*

### 4. `_load_config(self, config_path: str) -> Optional[Dict[str, Any]]`

This method is responsible for loading block-specific configurations from a YAML file.

* **Purpose**: Reads a YAML file specified by `config_path` and parses its content into a Python dictionary. This configuration often contains parameters specific to the block's operation (e.g., prompt details for an LLM block, filter conditions for a filter block).
* **Parameters**:
    * `config_path` (str): The file system path to the YAML configuration file.
* **Returns**:
    * `Optional[Dict[str, Any]]`: A dictionary containing the loaded configuration if successful. Returns `None` if the file cannot be parsed due to YAML errors or other unexpected issues during reading (excluding `FileNotFoundError`).
* **Error Handling**:
    * `FileNotFoundError`: If the `config_path` does not point to an existing file, the error is logged, and the exception is re-raised, typically halting the flow execution unless caught upstream.
    * `yaml.YAMLError`: If the file content is not valid YAML, the error is logged, and `None` is returned.
    * Other `Exception`: Any other unexpected errors during file reading are caught, logged, and `None` is returned.

```python
    def _load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Load the configuration file for this block.
        # ... (rest of docstring)
        """
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                try:
                    return yaml.safe_load(config_file)
                except yaml.YAMLError as e:
                    logger.error(f"Error parsing YAML from {config_path}: {e}")
                    return None
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading config file {config_path}: {e}")
            return None
```

### 5. Abstract Nature and Further Implementation

The `Block` class itself does not perform any specific data generation or transformation. Concrete subclasses are expected to:

1.  Inherit from `Block`.
2.  Implement their specific logic, typically within a `generate()` method (as suggested by the project's README, though not enforced as an `@abstractmethod` in this base class).
3.  Call `super().__init__(block_name)` in their constructor.
4.  Utilize `_load_config()` to load their configurations and `_validate()` if they use Jinja2 templates.
5.  Register themselves with `BlockRegistry` using the `@BlockRegistry.register("UniqueBlockTypeName")` decorator.

This base `Block` provides a solid foundation for creating a wide array of modular and configurable data processing units.
```
