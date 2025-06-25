# Available Blocks

This document provides a complete reference of all blocks available in SDG Hub, their purposes, parameters, and usage examples.

## Overview

Blocks are the fundamental processing units in SDG Hub. Each block performs a specific transformation on datasets, and blocks can be chained together in flows to create complex data processing pipelines.

## Block Types

### Core Framework

#### Block (Base Class)
- **Registered Name**: `Block`
- **Purpose**: Abstract base class providing common functionality for all blocks
- **Key Features**:
  - Template validation using Jinja2
  - Configuration file loading (YAML)
  - Input validation methods
- **Parameters**:
  - `block_name: str` - Name of the block instance

## LLM Blocks

### OpenAIChatBlock
- **Registered Name**: `OpenAIChatBlock`
- **Purpose**: Modern chat completion block using OpenAI Chat Completions API
- **Key Features**:
  - Direct OpenAI message format support (system/user/assistant roles)
  - All OpenAI Chat Completions API parameters supported
  - Automatic retry logic for rate limits and API errors
  - Comprehensive structured logging for monitoring
  - Works with any OpenAI-compatible endpoint

**Parameters:**
- `block_name: str` - Name of the block
- `input_cols: Union[str, List[str]]` - Input column containing messages (must be exactly one)
- `output_cols: Union[str, List[str]]` - Output column for responses (must be exactly one)
- `client: openai.OpenAI` - OpenAI client instance
- `model_id: str` - Model ID to use (e.g., "gpt-4", "gpt-3.5-turbo")
- **OpenAI API Parameters** (all optional):
  - `frequency_penalty: Optional[float]` - Penalize frequent tokens (-2.0 to 2.0)
  - `logit_bias: Optional[Dict[str, int]]` - Modify likelihood of specified tokens
  - `logprobs: Optional[bool]` - Whether to return log probabilities
  - `max_completion_tokens: Optional[int]` - Maximum tokens in completion
  - `max_tokens: Optional[int]` - Maximum tokens in completion (legacy)
  - `n: Optional[int]` - Number of completions to generate
  - `presence_penalty: Optional[float]` - Penalize repeated tokens (-2.0 to 2.0)
  - `response_format: Optional[Dict[str, Any]]` - Response format (e.g., JSON mode)
  - `seed: Optional[int]` - Seed for deterministic outputs
  - `stop: Optional[Union[str, List[str]]]` - Stop sequences
  - `stream: Optional[bool]` - Whether to stream responses
  - `temperature: Optional[float]` - Sampling temperature (0.0 to 2.0)
  - `tool_choice: Optional[Union[str, Dict[str, Any]]]` - Tool selection strategy
  - `tools: Optional[List[Dict[str, Any]]]` - Available tools for function calling
  - `top_logprobs: Optional[int]` - Number of top log probabilities to return
  - `top_p: Optional[float]` - Nucleus sampling parameter (0.0 to 1.0)
  - `user: Optional[str]` - End-user identifier
  - `extra_body: Optional[dict]` - Additional parameters for custom endpoints

**Example Usage:**
```yaml
- block_type: OpenAIChatBlock
  block_config:
    block_name: chat_generator
    input_cols: messages
    output_cols: response
    model_id: gpt-4
    temperature: 0.7
    max_tokens: 500
```

**Example with Messages Dataset:**
```python
import openai
from datasets import Dataset
from sdg_hub.blocks import OpenAIChatBlock

# Create client
client = openai.OpenAI(api_key="your-api-key")

# Prepare dataset with messages in OpenAI format
messages_data = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    [
        {"role": "user", "content": "What is the capital of France?"}
    ]
]
dataset = Dataset.from_dict({"messages": messages_data})

# Create and use block
block = OpenAIChatBlock(
    block_name="qa_generator",
    input_cols="messages",
    output_cols="response", 
    client=client,
    model_id="gpt-4",
    temperature=0.7,
    max_tokens=150
)

result = block.generate(dataset)
print(result["response"])
```

### OpenAIAsyncChatBlock
- **Registered Name**: `OpenAIAsyncChatBlock`
- **Purpose**: Async version of OpenAIChatBlock for concurrent processing and better performance
- **Key Features**:
  - Concurrent async requests for improved throughput
  - All features of OpenAIChatBlock
  - Better performance for large batches
  - Automatic concurrency management

**Parameters:**
- Same as `OpenAIChatBlock` except:
  - `async_client: openai.AsyncOpenAI` - Async OpenAI client instance

**Example Usage:**
```yaml
- block_type: OpenAIAsyncChatBlock
  block_config:
    block_name: async_chat_generator
    input_cols: messages
    output_cols: response
    model_id: gpt-4
    temperature: 0.7
    max_tokens: 500
```

**Example with Async Client:**
```python
import asyncio
import openai
from datasets import Dataset
from sdg_hub.blocks import OpenAIAsyncChatBlock

# Create async client
async_client = openai.AsyncOpenAI(api_key="your-api-key")

# Same dataset format as sync version
messages_data = [
    [{"role": "user", "content": f"Generate a creative story about topic {i}"}]
    for i in range(100)  # Large batch for demonstration
]
dataset = Dataset.from_dict({"messages": messages_data})

# Create and use async block
block = OpenAIAsyncChatBlock(
    block_name="async_story_generator",
    input_cols="messages",
    output_cols="story",
    async_client=async_client,
    model_id="gpt-4",
    temperature=0.8,
    max_tokens=200
)

# Process large batch concurrently
result = block.generate(dataset)
print(f"Generated {len(result)} stories concurrently")
```

**OpenAI-Compatible Endpoints:**
Both blocks work with any OpenAI-compatible endpoint:

```python
# Example with local endpoint
client = openai.OpenAI(
    api_key="not-needed-for-local",
    base_url="http://localhost:8000/v1"
)

# Example with other providers (Azure, Anthropic, etc.)
client = openai.OpenAI(
    api_key="your-provider-key",
    base_url="https://your-provider-endpoint.com/v1"
)
```

**Monitoring and Logging:**
Both blocks provide comprehensive structured logging:
- Initialization logs with model and parameters
- Generation start/completion logs with batch metrics
- Effective parameter tracking (including runtime overrides)
- Error tracking and retry information

Log output example:
```
INFO: Initialized OpenAIChatBlock 'chat_generator' with model 'gpt-4'
  {"block_name": "chat_generator", "model_id": "gpt-4", "generation_params": {"temperature": 0.7}}

INFO: Starting generation for 10 samples
  {"block_name": "chat_generator", "model_id": "gpt-4", "batch_size": 10, "effective_params": {"temperature": 0.9}}

INFO: Generation completed successfully for 10 samples
  {"block_name": "chat_generator", "model_id": "gpt-4", "batch_size": 10}
```

### LLMBlock
- **Registered Name**: `LLMBlock`
- **Purpose**: Core block for text generation using language models
- **Key Features**:
  - OpenAI-compatible API integration
  - Jinja2 prompt templating
  - Configurable output parsing
  - Batch processing support
  - Automatic server capability detection

**Parameters:**
- `block_name: str` - Name of the block
- `config_path: str` - Path to configuration file
- `client: openai.OpenAI` - OpenAI client instance
- `output_cols: List[str]` - Output column names
- `parser_kwargs: Dict[str, Any]` - Parser configuration
- `model_prompt: str` - Template for model prompt (default: "{prompt}")
- `model_id: Optional[str]` - Model ID to use
- `**batch_kwargs` - Additional batch processing arguments

**Example Usage:**
```yaml
- block_type: LLMBlock
  block_config:
    block_name: gen_knowledge
    config_path: configs/knowledge/simple_generate_qa.yaml
    model_id: mistralai/Mixtral-8x7B-Instruct-v0.1
    output_cols:
      - output
  gen_kwargs:
    temperature: 0.7
    max_tokens: 2048
```

### ConditionalLLMBlock
- **Registered Name**: `ConditionalLLMBlock`
- **Purpose**: LLM block that selects different prompt templates based on a selector column
- **Key Features**:
  - Multiple configuration file support
  - Conditional prompt selection
  - Inherits all LLMBlock functionality

**Parameters:**
- `block_name: str` - Name of the block
- `config_paths: Dict[str, str]` - Mapping of selector values to config paths
- `client: openai.OpenAI` - OpenAI client instance
- `model_id: str` - Model ID to use
- `output_cols: List[str]` - Output column names
- `selector_column_name: str` - Column used for template selection
- `model_prompt: str` - Template for model prompt
- `**batch_kwargs` - Additional batch processing arguments

**Example Usage:**
```yaml
- block_type: ConditionalLLMBlock
  block_config:
    block_name: conditional_gen
    config_paths:
      "math": configs/skills/math.yaml
      "coding": configs/skills/coding.yaml
    selector_column_name: category
    output_cols: [response]
```

## Utility Blocks

### FilterByValueBlock
- **Registered Name**: `FilterByValueBlock`
- **Purpose**: Filter datasets based on column values using various operations
- **Key Features**:
  - Multiple filter operations (eq, contains, ge, le, gt, lt, ne)
  - Optional data type conversion
  - Parallel processing support

**Parameters:**
- `block_name: str` - Name of the block
- `filter_column: str` - Column to filter on
- `filter_value: Union[Any, List[Any]]` - Value(s) to filter by
- `operation: Callable[[Any, Any], bool]` - Filter operation
- `convert_dtype: Optional[Union[Type[float], Type[int]]]` - Data type conversion
- `**batch_kwargs` - Additional batch processing arguments

**Example Usage:**
```yaml
- block_type: FilterByValueBlock
  block_config:
    block_name: filter_high_quality
    filter_column: quality_score
    filter_value: 0.8
    operation: operator.ge
    convert_dtype: float
```

### SamplePopulatorBlock
- **Registered Name**: `SamplePopulatorBlock`
- **Purpose**: Populate dataset samples with data from configuration files
- **Key Features**:
  - Multiple configuration file loading
  - Data mapping based on column values
  - Configuration file postfix support

**Parameters:**
- `block_name: str` - Name of the block
- `config_paths: List[str]` - List of configuration file paths
- `column_name: str` - Column used as key for data mapping
- `post_fix: str` - Suffix for configuration filenames
- `**batch_kwargs` - Additional batch processing arguments

### SelectorBlock
- **Registered Name**: `SelectorBlock`
- **Purpose**: Select and map values from one column to another based on choice mapping

**Parameters:**
- `block_name: str` - Name of the block
- `choice_map: Dict[str, str]` - Mapping of choice values to column names
- `choice_col: str` - Column containing choice values
- `output_col: str` - Column to store selected values
- `**batch_kwargs` - Additional batch processing arguments

**Example Usage:**
```yaml
- block_type: SelectorBlock
  block_config:
    block_name: select_best_response
    choice_map:
      "A": "response_a"
      "B": "response_b"
    choice_col: preferred_choice
    output_col: selected_response
```

### CombineColumnsBlock
- **Registered Name**: `CombineColumnsBlock`
- **Purpose**: Combine multiple columns into a single column with a separator

**Parameters:**
- `block_name: str` - Name of the block
- `columns: List[str]` - List of column names to combine
- `output_col: str` - Name of output column
- `separator: str` - Separator between combined values (default: "\n\n")
- `**batch_kwargs` - Additional batch processing arguments

**Example Usage:**
```yaml
- block_type: CombineColumnsBlock
  block_config:
    block_name: combine_qa_pair
    columns: [question, answer]
    output_col: qa_text
    separator: "\n\nAnswer: "
```

### FlattenColumnsBlock
- **Registered Name**: `FlattenColumnsBlock`
- **Purpose**: Transform wide format to long format by melting columns into rows

**Parameters:**
- `block_name: str` - Name of the block
- `var_cols: List[str]` - Columns to be melted into rows
- `value_name: str` - Name of new value column
- `var_name: str` - Name of new variable column

### DuplicateColumns
- **Registered Name**: `DuplicateColumns`
- **Purpose**: Create copies of existing columns with new names

**Parameters:**
- `block_name: str` - Name of the block
- `columns_map: Dict[str, str]` - Mapping of existing to new column names

**Example Usage:**
```yaml
- block_type: DuplicateColumns
  block_config:
    block_name: backup_columns
    columns_map:
      original_text: backup_text
      processed_text: backup_processed
```

### RenameColumns
- **Registered Name**: `RenameColumns`
- **Purpose**: Rename columns in a dataset according to mapping dictionary

**Parameters:**
- `block_name: str` - Name of the block
- `columns_map: Dict[str, str]` - Mapping of old to new column names

**Example Usage:**
```yaml
- block_type: RenameColumns
  block_config:
    block_name: standardize_names
    columns_map:
      input_text: prompt
      output_text: response
```

### SetToMajorityValue
- **Registered Name**: `SetToMajorityValue`
- **Purpose**: Set all values in a column to the most frequent (majority) value

**Parameters:**
- `block_name: str` - Name of the block
- `col_name: str` - Name of column to set to majority value

### IterBlock
- **Registered Name**: `IterBlock`
- **Purpose**: Apply another block multiple times iteratively

**Parameters:**
- `block_name: str` - Name of the block
- `num_iters: int` - Number of iterations
- `block_type: Type[Block]` - Block class to instantiate
- `block_kwargs: Dict[str, Any]` - Arguments for block constructor
- `**kwargs` - Additional arguments including gen_kwargs

**Example Usage:**
```yaml
- block_type: IterBlock
  block_config:
    block_name: iterative_improvement
    num_iters: 3
    block_type: LLMBlock
    block_kwargs:
      config_path: configs/improve_response.yaml
      output_cols: [improved_response]
```

## Specialized Blocks

These are custom blocks implemented in [examples](../examples)

### [AddStaticValue](../examples/skills_tuning/instructlab/blocks/add_question.py)
- **Registered Name**: `AddStaticValue`
- **Purpose**: Add a static value to a specified column in a dataset

**Parameters:**
- `block_name: str` - Name of the block
- `column_name: str` - Column to populate
- `static_value: str` - Constant value to add

### [DoclingParsePDF](../examples/skills_tuning/instructlab/blocks/docling_parse_pdf.py)
- **Registered Name**: `DoclingParsePDF`
- **Purpose**: Parse PDF documents into markdown format using Docling

**Parameters:**
- `block_name: str` - Name of the block
- `pdf_path_column: str` - Column containing PDF file paths
- `output_column: str` - Column to store markdown output

### [JSONFormat](../examples/skills_tuning/instructlab/blocks/json_format.py)
- **Registered Name**: `JSONFormat`
- **Purpose**: Format and standardize JSON output from text analysis results

**Parameters:**
- `block_name: str` - Name of the block
- `output_column: str` - Column to store formatted JSON

### [PostProcessThinkingBlock](../examples/knowledge_tuning/knowledge_tuning_with_reasoning_model/blocks/blocks.py)
- **Registered Name**: `PostProcessThinkingBlock`
- **Purpose**: Post-process thinking tokens from model outputs

**Parameters:**
- `block_name: str` - Name of the block
- `column_name: str` - Column to process

### [RegexParserBlock](../examples/knowledge_tuning/knowledge_tuning_with_reasoning_model/blocks/blocks.py)
- **Registered Name**: `RegexParserBlock`
- **Purpose**: Parse text using regular expressions and extract structured data

**Parameters:**
- `block_name: str` - Name of the block
- `column_name: str` - Column to parse
- `parsing_pattern: str` - Regex pattern for parsing
- `parser_cleanup_tags: List[str]` - Tags to clean up
- `output_cols: List[str]` - Output columns

## Key Block Features

### Registry System
All blocks are registered using the `@BlockRegistry.register()` decorator, enabling dynamic discovery and instantiation.

### Configuration Management
Blocks can load YAML configuration files containing prompts, templates, and other settings.

### Parallel Processing
Most blocks support multiprocessing through the `num_procs` parameter in batch_kwargs.

### Template Validation
Blocks validate inputs using Jinja2 templates to ensure required variables are provided.

### Flow Integration
Blocks are designed to work seamlessly in data processing pipelines, with consistent input/output interfaces.

## Creating Custom Blocks

To create a custom block:

```python
from sdg_hub.blocks import Block
from sdg_hub.registry import BlockRegistry
from datasets import Dataset

@BlockRegistry.register("MyCustomBlock")
class MyCustomBlock(Block):
    def __init__(self, block_name: str, custom_param: str, **kwargs):
        super().__init__(block_name)
        self.custom_param = custom_param
    
    def generate(self, dataset: Dataset, **kwargs) -> Dataset:
        # Custom processing logic
        processed_dataset = dataset.map(
            lambda x: {"processed": f"{self.custom_param}: {x['input']}"}
        )
        return processed_dataset
```

Then use it in a flow:

```yaml
- block_type: MyCustomBlock
  block_config:
    block_name: my_processor
    custom_param: "Processed"
```

## Best Practices

1. **Descriptive Names**: Use clear, descriptive block names for easier debugging
2. **Configuration Files**: Store complex prompts and templates in separate YAML files
3. **Error Handling**: Blocks should handle edge cases gracefully
4. **Documentation**: Include docstrings describing block purpose and parameters
5. **Testing**: Test blocks with various input formats and edge cases
6. **Performance**: Use batch processing and parallel execution for large datasets