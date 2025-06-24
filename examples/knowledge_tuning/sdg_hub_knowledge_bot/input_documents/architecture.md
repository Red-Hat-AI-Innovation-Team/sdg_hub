# Architecture Guide

This document outlines the core concepts and design principles of SDG Hub, a flexible framework for synthetic data generation.

## Overview

SDG Hub is designed around a modular, composable architecture that enables scalable and flexible synthetic data generation. The system follows a flow-based paradigm where data transformations are defined as sequences of processing blocks.

## Core Components

### 1. Flow System

The **Flow** (`src/sdg_hub/flow.py`) is the central orchestration component that:

- Loads and parses YAML configuration files defining data generation pipelines
- Manages the execution sequence of blocks
- Handles block configuration and dependency injection
- Provides backward compatibility with the deprecated Pipeline class

**Key Features:**
- YAML-based configuration for declarative pipeline definition
- Automatic block registration and instantiation
- Template and prompt resolution
- Resource path management

### 2. Block Architecture

**Blocks** are the fundamental processing units in SDG Hub. The base `Block` class (`src/sdg_hub/blocks/block.py`) provides:

- Abstract interface for all processing blocks
- Template validation using Jinja2
- Configuration loading and management
- Common utilities for data transformation

**Block Types:**
- **LLMBlock** (`src/sdg_hub/blocks/llmblock.py`): Interfaces with language models for text generation
- **UtilBlocks** (`src/sdg_hub/blocks/utilblocks.py`): Data manipulation utilities (filtering, sampling, formatting)

### 3. Registry System

The **Registry** (`src/sdg_hub/registry.py`) provides a plugin-like architecture with two main components:

- **BlockRegistry**: Manages registration and discovery of block types
- **PromptRegistry**: Manages Jinja2 templates for consistent prompt formatting

This enables:
- Dynamic block discovery
- Extensible block ecosystem
- Template reuse across different flows

### 4. SDG Orchestrator

The **SDG** class (`src/sdg_hub/sdg.py`) provides high-level orchestration with:

- Multi-pipeline execution support
- Concurrent processing with configurable worker threads
- Batch processing for large datasets
- Checkpoint management for long-running jobs

**Scalability Features:**
- Configurable batch sizes and worker pools
- Thread-safe dataset splitting and merging
- Fault tolerance through checkpointing

### 5. Checkpointing System

The **Checkpointer** (`src/sdg_hub/checkpointer.py`) enables:

- Resume interrupted generation jobs
- Incremental progress saving
- Efficient state management for large-scale operations

## Design Principles

### 1. Modularity
Each component has a single, well-defined responsibility with clear interfaces. Blocks can be composed in various ways without tight coupling.

### 2. Configurability
All pipeline logic is externalized to YAML configuration files, enabling rapid iteration without code changes.

### 3. Extensibility
The registry system allows new blocks and prompts to be added through simple decorators, following the plugin pattern.

### 4. Scalability
Built-in support for concurrent processing, batching, and checkpointing enables handling of large datasets.

### 5. Consistency
Standardized interfaces and validation ensure predictable behavior across different block types.

## Data Flow

The data flow follows these key stages:

1. **Configuration Loading**: Flow reads YAML configuration and resolves block types from registry
2. **Block Instantiation**: Blocks are created with resolved configurations and dependencies
3. **Sequential Processing**: Dataset flows through blocks in defined order
4. **Transformation**: Each block transforms the dataset according to its logic
5. **Output Generation**: Final processed dataset is returned

## Extension Points

### Adding New Blocks

```python
@BlockRegistry.register("MyCustomBlock")
class MyCustomBlock(Block):
    def __init__(self, block_name: str, **kwargs):
        super().__init__(block_name)
        # Custom initialization
    
    def generate(self, dataset: Dataset, **kwargs) -> Dataset:
        # Custom processing logic
        return processed_dataset
```

### Adding New Prompts

```python
@PromptRegistry.register("my_prompt")
def my_prompt_template():
    return """
    {%- for message in messages %}
        {{ message.role }}: {{ message.content }}
    {%- endfor %}
    """
```

## Configuration Structure

SDG Hub uses YAML files to define flows:

```yaml
- block_type: "LLMBlock"
  block_config:
    block_name: "generate_questions"
    model_id: "llama3.1-70b"
    config_path: "prompts/question_generation.yaml"
  gen_kwargs:
    num_samples: 5
  drop_columns: ["intermediate_data"]

- block_type: "FilterBlock"
  block_config:
    block_name: "quality_filter"
    filter_column: "quality_score"
    min_value: 0.7
```

This architecture provides a robust foundation for building sophisticated synthetic data generation pipelines while maintaining flexibility and extensibility.