# sdg_hub: Synthetic Data Generation Toolkit for LLMs

![Build](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/pypi.yaml/badge.svg?branch=main)
![Release](https://img.shields.io/github/v/release/Red-Hat-AI-Innovation-Team/sdg_hub)
![License](https://img.shields.io/github/license/Red-Hat-AI-Innovation-Team/sdg_hub)
[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub/graph/badge.svg?token=SP75BCXWO2)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub)

sdg_hub is a modular, scalable, and efficient solution for creating synthetic data generation flows in a "no-code" manner.
At its core, this framework is designed to simplify data creation for LLMs, allowing users to chain computational units and build powerful flows for generating data and processing tasks.

## 📺 Video Tutorial

For a comprehensive walkthrough of sdg_hub, check out our video tutorial:

[![SDG Hub Tutorial](https://img.youtube.com/vi/aGKCViWjAmA/0.jpg)](https://www.youtube.com/watch?v=aGKCViWjAmA)

## Installation

Latest release from PyPI

```sh
pip install sdg-hub
```

Latest main branch
```sh
pip install git+https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
```

## Core Design Principles

The framework is built around the following principles:

1. **Modular Design**: Highly composable blocks form the building units of the framework, allowing users to build flows effortlessly.
2. **No-Code Flow Creation**: Specify flows using simple YAML configuration files.
3. **Scalability and Performance**: Optimized for handling large-scale flows with millions of records.

---

## Framework Architecture

![overview](assets/imgs/fig-workflow.png)

### Key Concepts

1. **Blocks**: The fundamental computational units that perform specific tasks
2. **Flows**: A sequence of blocks that work together to process data
3. **Prompts**: YAML-based configurations that define how LLMs should behave

### Blocks: The Fundamental Unit

At the heart of the framework is the **Block**. Each block is a self-contained computational unit that performs specific tasks, such as:

- Making LLM calls
- Performing data transformations
- Applying filters

Blocks are designed to be:
- **Modular**: Reusable across multiple flows.
- **Composable**: Easily chained together to create flows.

These blocks are implemented in the [src/sdg_hub/blocks](src/sdg_hub/blocks) directory.

### Prompts

Prompts are at the core of how LLMs are instructed within SDG Hub. Each `LLMBlock` is associated with a prompt configuration file written in YAML, allowing users to define the exact behavior of the language model — including system instructions, generation principles, and output formatting.

#### Prompt YAML Structure

A typical prompt YAML file looks like this:

```yaml
system: You are a helpful assistant that can summarize text.
introduction: Give me a short summary of the text.
principles:
  - Do not add any new information.
  - Do not miss any key points from the provided text.
examples:
  - input: Red Hat announced the acquisition of Neural Magic...
    output: Red Hat acquired Neural Magic to enhance its AI optimization capabilities.
generation: Here is the document to summarize: {{document}}
```

#### Key Fields
* `system`: A high-level instruction that sets the persona or behavior of the model.
* `introduction`: Optional introduction to set context for the user.
* `principles`: A list of guiding constraints or rules the model should follow during generation.
* `examples`: Few-shot examples (optional) to guide output format or tone.
* `generation`: The actual template used to generate the model input. This supports variable injection using {{variable_name}}.

### YAML-Based Flow

The YAML configuration file, known as the **Flow**, is central to defining data generation workflows in the SDG Framework. A Flow describes how blocks are orchestrated to process and generate data efficiently. By leveraging YAML, users can create highly customizable and modular flows without writing any code.

#### Key Features of a Flow

1. **Modular Design**:
   - Flows are composed of blocks, which can be chained together
   - Each block performs a specific task, such as generating, filtering, or transforming data.

2. **Reusability**:
   - Blocks and configurations defined in a Flow can be reused across different flows
   - YAML makes it easy to tweak or extend flows without significant changes

3. **Ease of Configuration**:
   - Users can specify block types, configurations, and data processing details in a simple and intuitive manner

## Hello World Example

Let's say you have a document and want to generate a concise summary using an LLM. Here's how simple that is in sdg_hub:

```yaml
- block_type: LLMBlock
  block_config:
    block_name: gen_summary
    config_path: prompts/summarization.yaml
    model_id: meta-llama/Llama-3.3-70B-Instruct
    output_cols:
      - summary
  gen_kwargs:
    max_tokens: 512
```

Want to go further? Add another block to extract keywords from the summary:

```yaml
- block_type: LLMBlock
  block_config:
    block_name: gen_keywords
    config_path: prompts/keywords.yaml
    model_id: meta-llama/Llama-3.3-70B-Instruct
    output_cols:
      - keywords
  gen_kwargs:
    max_tokens: 64
```

Just like that, you've built a multi-step LLM workflow using nothing but YAML.

## Available Blocks

The SDG Framework provides a rich set of blocks for different data processing needs. Here's a comprehensive overview of the available blocks and when to use them:

### Base Block Class

The framework is built around the abstract `Block` class, which serves as the foundation for all other blocks:

- **Purpose**: Provides core functionality and interface for all blocks
- **Key Features**:
  - Template validation for input data
  - Configuration loading from YAML files
  - Standardized block initialization
  - Common interface for all blocks
- **Core Methods**:
  - `_validate`: Validates input data against templates
  - `_load_config`: Loads configuration from YAML files
  - `generate`: Abstract method for block execution

All blocks inherit from this base class, ensuring consistent behavior and interface across the framework.

### LLM Blocks

1. **LLMBlock**
   - **Purpose**: Generate text using language models
   - **Use Cases**: 
     - Generating questions, responses, or any text content
     - Single-prompt generation with structured outputs
   - **Features**: 
     - Supports batched processing
     - Configurable output parsing
     - Template-based prompt generation

2. **ConditionalLLMBlock**
   - **Purpose**: Generate text based on conditional logic
   - **Use Cases**:
     - Different prompt templates based on input conditions
     - Multi-path text generation workflows
   - **Features**:
     - Multiple config paths for different conditions
     - Dynamic prompt selection


### Filtering and Processing Blocks

1. **FilterByValueBlock**
   - **Purpose**: Filter datasets based on column values
   - **Use Cases**:
     - Removing unwanted samples
     - Data cleaning
     - Quality filtering
   - **Features**:
     - Multiple filter operations
     - Type conversion support
     - Parallel processing

2. **IterBlock**
   - **Purpose**: Iterative processing of data
   - **Use Cases**:
     - Multiple generation attempts
     - Iterative refinement
   - **Features**:
     - Configurable number of iterations
     - Nested block execution

### Utility Blocks

1. **SamplePopulatorBlock**
   - **Purpose**: Populate samples with configuration data
   - **Use Cases**:
     - Adding metadata
     - Configuration injection

2. **SelectorBlock**
   - **Purpose**: Select data based on mapping
   - **Use Cases**:
     - Conditional data selection
     - Data routing

3. **CombineColumnsBlock**
   - **Purpose**: Merge multiple columns
   - **Use Cases**:
     - Text concatenation
     - Feature combination

4. **FlattenColumnsBlock**
   - **Purpose**: Convert wide to long format
   - **Use Cases**:
     - Data reshaping
     - Variable-value pairs

5. **DuplicateColumns**
   - **Purpose**: Create column copies
   - **Use Cases**:
     - Data preservation
     - Multiple processing paths

6. **RenameColumns**
   - **Purpose**: Rename dataset columns
   - **Use Cases**:
     - Standardizing column names
     - Data reorganization

7. **SetToMajorityValue**
   - **Purpose**: Replace values with majority
   - **Use Cases**:
     - Data normalization
     - Outlier handling

---
### Dataflow and Storage

- **Data Representation**: Dataflow between blocks and flows is handled using **Hugging Face Datasets**, which are based on Arrow tables. This provides:
  - Native parallelization capabilities (e.g., maps, filters).
  - Support for efficient data transformations.

- **Data Checkpoints**: Intermediate caches of generated data. Checkpoints allow users to:
  - Resume flows from the last successful state if interrupted.
  - Improve reliability for long-running flows.

## Examples

For sample use cases and implementation examples, please refer to the [examples](examples) directory. This directory contains various examples demonstrating different flows and use cases of the SDG Framework.

## Additional Resources

- [Video Tutorial](https://www.youtube.com/watch?v=aGKCViWjAmA): A comprehensive walkthrough of sdg_hub
- [Examples Directory](examples/): Sample flows and use cases
- [GitHub Repository](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub): Source code and issue tracking
