# BaseBlock Development Guide

This guide provides comprehensive documentation for creating blocks using the new standardized BaseBlock system in SDG Hub.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [BaseBlock Features](#baseblock-features)
4. [Creating Your First Block](#creating-your-first-block)
5. [Advanced Features](#advanced-features)
6. [Validation System](#validation-system)
7. [Rich Logging](#rich-logging)
8. [Error Handling](#error-handling)
9. [Testing Your Block](#testing-your-block)
10. [Best Practices](#best-practices)
11. [Migration Guide](#migration-guide)

## Overview

The BaseBlock system provides a standardized foundation for creating blocks in SDG Hub. It offers:

- **Standardized Constructor Patterns**: Consistent parameter handling across all blocks
- **Automatic Validation**: Built-in dataset and column validation
- **Rich Logging**: Beautiful console output with progress tracking
- **Error Handling**: Specific exceptions for better debugging
- **Immutable Properties**: Thread-safe access to block metadata
- **Flexible Column Handling**: Support for various column specification formats

## Getting Started

### Import the BaseBlock

```python
from sdg_hub.blocks.base import BaseBlock
from sdg_hub.registry import BlockRegistry
from datasets import Dataset
```

### Basic Block Structure

```python
@BlockRegistry.register("MyBlock")
class MyBlock(BaseBlock):
    def __init__(self, block_name: str, **kwargs):
        super().__init__(
            block_name=block_name,
            input_cols=["input_text"],  # Required input columns
            output_cols=["output_text"],  # Expected output columns
            **kwargs
        )
    
    def generate(self, samples: Dataset, **kwargs) -> Dataset:
        """Your block's processing logic goes here."""
        # Process the dataset
        return processed_dataset
```

## BaseBlock Features

### 1. Standardized Constructor

The BaseBlock constructor provides consistent parameter handling:

```python
def __init__(
    self,
    block_name: str,
    input_cols: Optional[Union[str, List[str]]] = None,
    output_cols: Optional[Union[str, List[str]]] = None,
    **kwargs: Any,
) -> None:
```

**Parameters:**
- `block_name`: Unique identifier for the block instance
- `input_cols`: Required input columns (flexible format)
- `output_cols`: Expected output columns (flexible format)
- `**kwargs`: Additional block-specific parameters

### 2. Column Specification Flexibility

The BaseBlock automatically normalizes column specifications:

```python
# All of these are valid:
MyBlock("test", input_cols="single_col")           # str → ["single_col"]
MyBlock("test", input_cols=["col1", "col2"])       # List[str] → ["col1", "col2"]
MyBlock("test", input_cols=None)                   # None → []
```

### 3. Immutable Properties

Block properties are immutable and return defensive copies:

```python
block = MyBlock("test", input_cols=["col1", "col2"])

# These return copies - safe to modify
input_cols = block.input_cols
output_cols = block.output_cols
block_name = block.block_name

# Original properties remain unchanged
input_cols.append("new_col")  # Safe - doesn't affect the block
```

### 4. Automatic Validation and Logging

When you call your block, validation and logging happen automatically:

```python
# This automatically:
# 1. Logs input dataset info with Rich panels
# 2. Validates dataset is not empty
# 3. Validates required input columns exist
# 4. Validates output columns won't collide
# 5. Calls your generate() method
# 6. Logs output dataset info with change tracking
result = block(dataset)
```

## Validation System

The BaseBlock provides comprehensive validation:

### 1. Empty Dataset Validation

```python
# Automatically prevents processing empty datasets
empty_dataset = Dataset.from_list([])
block = MyBlock("test")

try:
    result = block(empty_dataset)
except EmptyDatasetError as e:
    print(f"Block '{e.block_name}' received empty dataset")
```

### 2. Missing Column Validation

```python
# Automatically validates required input columns exist
dataset = Dataset.from_list([{"wrong_col": "data"}])
block = MyBlock("test", input_cols=["required_col"])

try:
    result = block(dataset)
except MissingColumnError as e:
    print(f"Missing columns: {e.missing_columns}")
    print(f"Available columns: {e.available_columns}")
```

### 3. Output Column Collision Validation

```python
# Automatically prevents overwriting existing data
dataset = Dataset.from_list([{"text": "data", "result": "existing"}])
block = MyBlock("test", output_cols=["result"])  # Would overwrite existing 'result'

try:
    result = block(dataset)
except OutputColumnCollisionError as e:
    print(f"Column collision: {e.collision_columns}")
    print(f"Existing columns: {e.existing_columns}")
```

### 4. Custom Validation

You can add custom validation in your block:

```python
class MyBlock(BaseBlock):
    def generate(self, samples: Dataset, **kwargs) -> Dataset:
        # Custom validation before processing
        for sample in samples:
            if len(sample["text"]) < 10:
                raise ValueError(f"Text too short: {sample['text']}")
        
        # Your processing logic
        return processed_samples
```

## Rich Logging

The BaseBlock provides beautiful console logging with Rich panels:

### Input Panel Example
```
╭─────────────────────── MyBlock ───────────────────────╮
│ 📊 Processing Input Data                              │
│ Block Type: TextTransformBlock                       │
│ Input Rows: 1,000                                    │
│ Input Columns: 3                                     │
│ Column Names: text, category, metadata               │
│ Expected Output Columns: transformed_text, sentiment │
╰──────────────────────────────────────────────────────╯
```

### Output Panel Example
```
╭─────────────── MyBlock — Complete ────────────────╮
│ ✅ Processing Complete                             │
│ • Rows: 1,000 → 1,200  (+200)                     │
│ • Columns: 3 → 5  (+2)                            │
│ 🟢 Added: transformed_text, sentiment             │
│ 📋 Final Columns: text, category, metadata, ...   │
╰───────────────────────────────────────────────────╯
```

### Color Coding
- 🔵 **Blue**: Input processing information
- 🟢 **Green**: Successful completion, added items
- 🔴 **Red**: Removed items, errors
- 🟡 **Yellow**: Changes indicator
- ⚪ **White**: Column names
- 🔅 **Dim**: Labels and unchanged items

## Error Handling

### Custom Exception Hierarchy

```python
from sdg_hub.utils.error_handling import (
    BlockValidationError,
    MissingColumnError,
    EmptyDatasetError,
    OutputColumnCollisionError,
)

# All validation errors inherit from BlockValidationError
try:
    result = block(dataset)
except BlockValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Details: {e.details}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Structured Error Information

```python
try:
    result = block(dataset)
except MissingColumnError as e:
    # Structured error data
    print(f"Block: {e.block_name}")
    print(f"Missing: {e.missing_columns}")
    print(f"Available: {e.available_columns}")
    
    # Formatted message
    print(f"Error: {e}")
```

## Best Practices

### 1. Constructor Design

```python
class MyBlock(BaseBlock):
    def __init__(
        self,
        block_name: str,
        # Required parameters first
        required_param: str,
        # Optional parameters with defaults
        optional_param: str = "default",
        # Flexible parameters
        **kwargs
    ):
        super().__init__(
            block_name=block_name,
            input_cols=["required_input"],
            output_cols=["output"],
            **kwargs
        )
        self.required_param = required_param
        self.optional_param = optional_param
```

### 2. Error Handling

```python
def generate(self, samples: Dataset, **kwargs) -> Dataset:
    try:
        # Your processing logic
        return processed_samples
    except Exception as e:
        # Log the error with context
        logger.error(f"Processing failed in block '{self.block_name}': {e}")
        raise
```

### 3. Performance Considerations

```python
def generate(self, samples: Dataset, **kwargs) -> Dataset:
    # Use map for parallel processing
    return samples.map(
        self.process_sample,
        num_proc=self.kwargs.get("num_procs", 1)
    )

def process_sample(self, sample):
    # Process individual samples
    return sample
```

### 4. Documentation

```python
@BlockRegistry.register("MyBlock")
class MyBlock(BaseBlock):
    """A block that performs specific text processing.
    
    This block transforms input text according to specified parameters
    and generates processed output with metadata.
    
    Parameters
    ----------
    block_name : str
        Unique identifier for the block instance.
    transform_type : str, optional
        Type of transformation to apply, by default "uppercase".
    preserve_metadata : bool, optional
        Whether to preserve original metadata, by default True.
    **kwargs : Any
        Additional parameters passed to BaseBlock.
    
    Examples
    --------
    >>> block = MyBlock("transformer", transform_type="title")
    >>> result = block(dataset)
    """
```

## Migration Guide

### From Old Block System

**Old Way:**
```python
from sdg_hub.blocks import Block

class MyBlock(Block):
    def __init__(self, block_name: str, custom_param: str):
        super().__init__(block_name)
        self.custom_param = custom_param
    
    def generate(self, samples: Dataset, **kwargs) -> Dataset:
        # Manual validation
        if len(samples) == 0:
            raise ValueError("Empty dataset")
        
        # Processing logic
        return processed_samples
```

**New Way:**
```python
from sdg_hub.blocks.base import BaseBlock

class MyBlock(BaseBlock):
    def __init__(self, block_name: str, custom_param: str, **kwargs):
        super().__init__(
            block_name=block_name,
            input_cols=["required_input"],
            output_cols=["output"],
            **kwargs
        )
        self.custom_param = custom_param
    
    def generate(self, samples: Dataset, **kwargs) -> Dataset:
        # Validation happens automatically
        # Processing logic only
        return processed_samples
```

### Key Changes

1. **Import Change**: `from sdg_hub.blocks.base import BaseBlock`
2. **Constructor**: Add `input_cols` and `output_cols` parameters
3. **Validation**: Remove manual validation - it's now automatic
4. **Logging**: Rich panels are automatic
5. **Error Handling**: Use specific exception types
6. **Properties**: Use immutable properties for metadata


## Conclusion

The BaseBlock system provides a robust foundation for creating blocks in SDG Hub. It handles the common concerns of validation, logging, and error handling, allowing you to focus on your block's core functionality.

Key benefits:
- **Standardized**: Consistent patterns across all blocks
- **Robust**: Comprehensive validation and error handling
- **User-friendly**: Beautiful Rich logging and clear error messages
- **Testable**: Easy to test with built-in validation
- **Maintainable**: Clean separation of concerns

---

*This documentation is for the BaseBlock system introduced in SDG Hub v0.2.0. For the previous block system, see [blocks.md](blocks.md).*