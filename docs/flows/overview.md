# Flow System Overview

Flows are the orchestration layer of SDG Hub, enabling you to chain multiple blocks together into sophisticated data generation pipelines. Built on YAML configuration files, flows provide a declarative way to define complex workflows with proper validation, metadata tracking, and runtime parameter management.

## 🌊 Flow Philosophy

Flows embody the principle of **composable data pipelines**:

```
Input Dataset → Block₁ → Block₂ → Block₃ → ... → Enhanced Dataset
```

Each block in the sequence:
1. **Receives** the output dataset from the previous block
2. **Processes** the data according to its configuration
3. **Validates** inputs and outputs
4. **Passes** the enhanced dataset to the next block

This sequential processing model ensures data integrity while enabling complex transformations through simple composition.

## 🏗️ Flow Architecture

### Core Components

Every flow consists of three main sections:

1. **Metadata** - Flow identification, versioning, and requirements
2. **Parameters** - Runtime configuration options
3. **Blocks** - Ordered sequence of processing blocks

### Execution Model

Flows provide several execution modes:

- **Standard Execution** - Process the full dataset through all blocks
- **Dry Run** - Test with a small sample to validate the pipeline
- **Step-by-Step** - Execute individual blocks for debugging
- **Parallel Processing** - Async execution where supported

## 📋 Flow Structure

### Basic YAML Structure

```yaml
metadata:
  name: "My Data Processing Flow"
  description: "Processes documents to generate Q&A pairs"
  version: "1.0.0"
  author: "Your Name"
  
  recommended_models:
    default: "meta-llama/Llama-3.3-70B-Instruct"
    compatible: ["microsoft/phi-4", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
  
  tags:
    - "question-generation"
    - "document-processing"
  
  dataset_requirements:
    required_columns: ["document", "context"]
    description: "Input documents for processing"


blocks:
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "question_generator"
      input_cols: ["document"]
      output_cols: ["question"]
      max_tokens: 100
  
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "answer_generator"
      input_cols: ["document", "question"]
      output_cols: ["answer"]
      max_tokens: 200
```

### Metadata Section

The metadata section provides essential information about the flow:

```yaml
metadata:
  name: "Unique Flow Name"
  description: "Detailed description of what this flow does"
  version: "1.0.0"  # Semantic versioning
  author: "Author Name"
  license: "Apache-2.0"
  min_sdg_hub_version: "0.2.0"  # Minimum required SDG Hub version
  
  # Model recommendations for optimal performance
  recommended_models:
    default: "meta-llama/Llama-3.3-70B-Instruct"
    compatible: 
      - "microsoft/phi-4"
      - "mistralai/Mixtral-8x7B-Instruct-v0.1"
    experimental: 
      - "google/gemini-pro"
  
  # Tags for discovery and categorization
  tags:
    - "question-generation"
    - "document-processing"
    - "educational"
  
  # Dataset requirements and validation
  dataset_requirements:
    required_columns:
      - "document"
      - "context"
      - "domain"
    description: "Documents with context for Q&A generation"
    min_samples: 1
    max_samples: 10000
```

### Metadata Fields Reference

The metadata section supports the following fields for flow configuration:

#### Core Metadata Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `string` | Yes | - | Human-readable name of the flow. Must be at least 1 character. |
| `id` | `string` | No | Auto-generated | Unique identifier for the flow. Auto-generated from name if not provided. Must be lowercase, contain only alphanumeric characters and hyphens, and not start/end with hyphens. |
| `description` | `string` | No | `""` | Detailed description of what the flow does and its purpose. |
| `version` | `string` | No | `"1.0.0"` | Semantic version following the format `MAJOR.MINOR.PATCH` (e.g., "1.0.0", "2.1.3-beta"). |
| `author` | `string` | No | `""` | Name of the flow author or contributor. |
| `license` | `string` | No | `"Apache-2.0"` | License identifier for the flow (e.g., "Apache-2.0", "MIT", "GPL-3.0"). |
| `tags` | `List[string]` | No | `[]` | List of tags for categorization and discovery. Tags are automatically converted to lowercase. |
| `recommended_models` | `RecommendedModels` | No | `None` | Recommended LLM models for optimal flow performance. See below for structure. |
| `dataset_requirements` | `DatasetRequirements` | No | `None` | Input dataset requirements and validation rules. See below for structure. |

#### RecommendedModels Structure

The `recommended_models` field helps users choose appropriate LLM models for the flow:

```yaml
recommended_models:
  default: "meta-llama/Llama-3.3-70B-Instruct"
  compatible:
    - "microsoft/phi-4"
    - "mistralai/Mixtral-8x7B-Instruct-v0.1"
  experimental:
    - "google/gemini-pro"
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `default` | `string` | Yes | - | The default model recommended for this flow. This is the primary model users should use. |
| `compatible` | `List[string]` | No | `[]` | List of models known to work well with this flow. Alternative options with good performance. |
| `experimental` | `List[string]` | No | `[]` | List of experimental models that may work but haven't been extensively tested with this flow. |

**Model Selection Behavior:**

When the framework needs to select a model, it prioritizes in this order:
1. `default` model if available
2. First available model from `compatible` list
3. First available model from `experimental` list

#### DatasetRequirements Structure

The `dataset_requirements` field validates input datasets and documents expected data format:

```yaml
dataset_requirements:
  required_columns:
    - "document"
    - "context"
  optional_columns:
    - "metadata"
    - "source"
  min_samples: 1
  max_samples: 10000
  column_types:
    document: "string"
    context: "string"
  description: "Documents with context for Q&A generation"
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `required_columns` | `List[string]` | No | `[]` | Column names that must be present in the input dataset. Flow validation will fail if these are missing. |
| `optional_columns` | `List[string]` | No | `[]` | Column names that are optional but can enhance flow performance if provided. |
| `min_samples` | `integer` | No | `1` | Minimum number of samples required in the input dataset. Must be at least 1. |
| `max_samples` | `integer` | No | `None` | Maximum number of samples to process. Useful for resource management and preventing excessive processing. |
| `column_types` | `Dict[string, string]` | No | `{}` | Expected data types for specific columns (e.g., "string", "integer", "float"). Used for documentation purposes. |
| `description` | `string` | No | `""` | Human-readable description of the dataset requirements and expected format. |

**Validation Behavior:**

- The flow will validate the input dataset against `required_columns` before execution
- Missing required columns will cause the flow to fail with a clear error message
- Sample count validation ensures the dataset meets `min_samples` and respects `max_samples` if set
- `max_samples` must be greater than or equal to `min_samples` if both are specified

#### Complete Metadata Example

Here's a comprehensive example using all available metadata fields:

```yaml
metadata:
  name: "Advanced Document Q&A Generation"
  id: "advanced-document-qa-generation"
  description: |
    A sophisticated flow that processes documents to generate high-quality
    question-answer pairs with faithfulness evaluation and quality filtering.
    Designed for educational content and training data generation.
  version: "2.1.0"
  author: "SDG Hub Team"
  license: "Apache-2.0"

  recommended_models:
    default: "meta-llama/Llama-3.3-70B-Instruct"
    compatible:
      - "microsoft/phi-4"
      - "mistralai/Mixtral-8x7B-Instruct-v0.1"
      - "meta-llama/Llama-3.1-70B-Instruct"
    experimental:
      - "google/gemini-pro"
      - "anthropic/claude-3-opus"

  tags:
    - "question-generation"
    - "document-processing"
    - "educational"
    - "qa-pairs"

  dataset_requirements:
    required_columns:
      - "document"
      - "context"
    optional_columns:
      - "domain"
      - "difficulty_level"
      - "source_url"
    min_samples: 10
    max_samples: 5000
    column_types:
      document: "string"
      context: "string"
      domain: "string"
      difficulty_level: "integer"
    description: |
      Input dataset should contain documents with contextual information.
      Each document should be well-formed text suitable for Q&A generation.
      Optional domain and difficulty_level fields help tailor generation.
```

### Blocks Section

The blocks section defines the processing pipeline:

```yaml
blocks:
  # Document preprocessing
  - block_type: "DuplicateColumnsBlock"
    block_config:
      block_name: "backup_document"
      input_cols: {document: "original_document"}
  
  # Prompt construction
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_question_prompt"
      input_cols: ["document", "context"]
      output_cols: ["question_prompt"]
      prompt_template: |
        Based on this document: {document}
        
        Context: {context}
        
        Generate a thoughtful question:
      format_as_messages: true
  
  # Question generation
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate_question"
      input_cols: ["question_prompt"]
      output_cols: ["question"]
      max_tokens: 100
      temperature: 0.8
      async_mode: true
  
  # Answer generation
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate_answer"
      input_cols: ["document", "question"]
      output_cols: ["answer"]
      prompt_template: |
        Document: {document}
        
        Question: {question}
        
        Provide a comprehensive answer:
      max_tokens: 300
      async_mode: true
  
  # Quality evaluation using basic blocks
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "faithfulness_prompt"
      input_cols: ["document", "answer"]
      output_cols: ["eval_prompt"]
      prompt_template: "Evaluate if this answer is faithful to the document..."

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "eval_faithfulness_llm"
      input_cols: ["eval_prompt"]
      output_cols: ["eval_response"]
      async_mode: true

  - block_type: "LLMParserBlock"
    block_config:
      block_name: "extract_eval_content"
      input_cols: ["eval_response"]
      extract_content: true

  - block_type: "TextParserBlock"
    block_config:
      block_name: "parse_evaluation"
      input_cols: ["extract_eval_content_content"]
      output_cols: ["explanation", "judgment"]
      start_tags: ["[Start of Explanation]", "[Start of Answer]"]
      end_tags: ["[End of Explanation]", "[End of Answer]"]

  - block_type: "ColumnValueFilterBlock"
    block_config:
      block_name: "filter_faithful"
      input_cols: ["judgment"]
      filter_value: "YES"
      operation: "eq"
  
  # Quality filtering
  - block_type: "ColumnValueFilterBlock"
    block_config:
      block_name: "quality_filter"
      filter_column: "faithfulness_score"
      operator: "greater_than"
      threshold: 0.8
```

## ⚙️ Model Configuration

The new v0.2 model configuration system provides runtime flexibility:

### Model Discovery

```python
from sdg_hub.core.flow import Flow

# Load a flow
flow = Flow.from_yaml("path/to/flow.yaml")

# Discover recommended models
default_model = flow.get_default_model()
print(f"Default model: {default_model}")

# Get all recommendations
recommendations = flow.get_model_recommendations()
print(f"Compatible models: {recommendations['compatible']}")
print(f"Experimental models: {recommendations['experimental']}")
```

### Runtime Model Configuration

```python
# Configure model at runtime
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="your_key"
)

# Alternative: Use cloud providers
flow.set_model_config(
    model="openai/gpt-4o",
    api_key="your-openai-key"
)

# The configuration applies to all LLM blocks in the flow
```

## 🚀 Flow Execution

### Basic Execution

```python
from datasets import Dataset

# Create input dataset
dataset = Dataset.from_dict({
    "document": ["Your document content here..."],
    "context": ["Relevant context information..."],
    "domain": ["Technology"]
})

# Execute the complete flow
result = flow.generate(dataset)

# Access generated content
questions = result["question"]
answers = result["answer"]
quality_scores = result["faithfulness_score"]
```

### Dry Run Testing

Always test with dry runs before processing large datasets:

```python
# Test with a small sample
dry_result = flow.dry_run(dataset, sample_size=2)

print(f"Dry run completed in {dry_result['execution_time_seconds']:.2f}s")
print(f"Input columns: {dry_result['input_columns']}")
print(f"Output columns: {dry_result['final_dataset']['columns']}")
print(f"Sample output: {dry_result['sample_output']}")
```

### Time Estimation

Predict execution time for your full dataset before running:

```python
# Get dry run results AND time estimation in one call
result = flow.dry_run(
    dataset, 
    sample_size=5, 
    enable_time_estimation=True,
    max_concurrency=100
)

# Time estimation is automatically displayed in a Rich table format
# The table shows estimated time, total API requests, and per-block breakdowns
print(f"Dry run completed with {result['sample_size']} samples")
print(f"Output columns: {result['final_dataset']['columns']}")
```

**How It Works:**

The estimation uses 2 dry runs to accurately predict execution time:
- Extracts startup overhead (one-time costs)
- Calculates per-sample throughput (variable costs)
- Uses linear regression to separate fixed from variable costs

**Accuracy:**
- Includes a 20% conservative buffer to account for API variability
- Typical accuracy: within 15-40% of actual runtime depending on workload
- Better to finish early than run over time!

**When to Use:**
- Before processing with your full dataset
- To identify bottleneck blocks and optimize your pipeline

### Runtime Parameters

Runtime parameters allow you to customize block behavior at execution time without modifying flow YAML files. You can override global parameters for all blocks or configure specific blocks individually.

**Global Parameter Override:**

Apply parameters to all compatible blocks in the flow:

```python
# Override global parameters
result = flow.generate(
    dataset,
    runtime_params={
        "temperature": 0.7,
        "max_tokens": 200,
        "top_p": 0.95
    }
)
```

**Block-Specific Configuration:**

Target individual blocks by their `block_name` for fine-grained control:

```python
# Configure different parameters for each block
result = flow.generate(
    dataset,
    runtime_params={
        # LLM blocks - control generation parameters
        "question_generator": {
            "temperature": 0.9,
            "max_tokens": 100,
            "top_p": 0.95,
            "frequency_penalty": 0.5
        },
        "answer_generator": {
            "temperature": 0.5,
            "max_tokens": 300,
            "presence_penalty": 0.3
        },

        # LLM parser blocks - configure extraction
        "extract_eval_content": {
            "extract_content": True,
            "extract_reasoning_content": True,
            "field_prefix": "llm_"
        },

        # Text parsing blocks - override parsing tags
        "parse_evaluation": {
            "start_tags": ["[Answer]", "[Explanation]", "[Score]"],
            "end_tags": ["[/Answer]", "[/Explanation]", "[/Score]"],
            "parser_cleanup_tags": ["```", "###", "---"]
        },

        # Filter blocks - adjust filter criteria
        "quality_filter": {
            "filter_value": 0.9,
            "operation": "ge"
        },
        "faithfulness_filter": {
            "filter_value": "YES",
            "operation": "eq"
        }
    }
)
```

**Common Runtime Parameters by Block Type:**

| Block Type | Parameter | Description | Example Values |
|------------|-----------|-------------|----------------|
| **LLMChatBlock** | `temperature` | Control randomness in generation | `0.0` - `2.0` |
| | `max_tokens` | Maximum response length | `50`, `200`, `1000` |
| | `top_p` | Nucleus sampling threshold | `0.0` - `1.0` |
| | `frequency_penalty` | Penalize token repetition | `-2.0` - `2.0` |
| | `presence_penalty` | Penalize new topics | `-2.0` - `2.0` |
| **LLMParserBlock** | `extract_content` | Extract main content field | `True`, `False` |
| | `extract_reasoning_content` | Extract reasoning/thinking | `True`, `False` |
| | `extract_tool_calls` | Extract tool call data | `True`, `False` |
| | `field_prefix` | Prefix for output fields | `"llm_"`, `"parsed_"` |
| **TextParserBlock** | `start_tags` | Opening tags for extraction | `["<answer>", "[Q]"]` |
| | `end_tags` | Closing tags for extraction | `["</answer>", "[/Q]"]` |
| | `parsing_pattern` | Custom regex pattern | `r"Answer:\s*(.+)"` |
| | `parser_cleanup_tags` | Tags to remove from output | `["```", "###"]` |
| **ColumnValueFilterBlock** | `filter_value` | Value to filter by | `0.8`, `"YES"`, `[1, 2]` |
| | `operation` | Comparison operation | `"eq"`, `"gt"`, `"contains"` |
| | `convert_dtype` | Type conversion | `"float"`, `"int"` |

**Practical Examples:**

```python
# Experiment with different generation styles
result = flow.generate(
    dataset,
    runtime_params={
        "temperature": 0.9,  # More creative
        "top_p": 0.95
    }
)

# Adjust parsing for different prompt formats
result = flow.generate(
    dataset,
    runtime_params={
        "text_parser": {
            "start_tags": ["<thinking>", "<answer>"],
            "end_tags": ["</thinking>", "</answer>"]
        }
    }
)

# Increase quality thresholds for production
result = flow.generate(
    dataset,
    runtime_params={
        "quality_filter": {"filter_value": 0.95},
        "relevancy_filter": {"filter_value": 0.90}
    }
)

# Mix global and block-specific parameters
result = flow.generate(
    dataset,
    runtime_params={
        "temperature": 0.7,  # Global default
        "creative_generator": {"temperature": 1.0},  # Override for one block
        "quality_filter": {"filter_value": 0.85}
    }
)
```

### Concurrency Control

For flows containing LLM blocks, you can control the maximum number of concurrent API requests to prevent overwhelming servers or hitting rate limits:

```python
# Basic concurrency control
result = flow.generate(
    dataset,
    max_concurrency=5  # Max 5 concurrent requests per LLM block execution
)

# Combined with other parameters
result = flow.generate(
    dataset,
    max_concurrency=10,
    runtime_params={
        "temperature": 0.7,
        "max_tokens": 200
    }
)
```

**When to Use Concurrency Control:**

- **Large Datasets** - Process thousands of samples without overwhelming APIs
- **Rate Limit Management** - Respect provider-specific concurrent request limits
- **Production Workloads** - Ensure stable, predictable resource usage
- **Cost Optimization** - Prevent burst API charges from uncontrolled parallelism

**Recommended Settings:**

```python
# Conservative (recommended for production)
result = flow.generate(dataset, max_concurrency=5)

# Moderate (good for development/testing)  
result = flow.generate(dataset, max_concurrency=10)

# Aggressive (only for robust APIs and small datasets)
result = flow.generate(dataset, max_concurrency=20)

# No limit (maximum speed, use with caution)
result = flow.generate(dataset)  # Default behavior
```

### Checkpointing

Flow checkpointing enables resuming interrupted executions by saving progress periodically. This is essential for long-running flows that process large datasets, preventing data loss from failures or interruptions.

**Basic Checkpointing:**

```python
# Enable checkpointing with automatic resume
result = flow.generate(
    dataset,
    checkpoint_dir="./my_flow_checkpoints",
    save_freq=100  # Save every 100 completed samples
)
```

**How It Works:**

1. **Progress Tracking** - Flow saves completed samples to checkpoint files after every `save_freq` samples
2. **Automatic Resume** - On restart, Flow detects existing checkpoints and processes only remaining samples
3. **Final Merge** - Completed and newly processed samples are automatically combined in the final result

**Use Cases:**

- **Long-Running Flows** - Process thousands of samples safely over hours or days
- **Unreliable Infrastructure** - Protect against network failures, rate limits, or system crashes
- **Iterative Development** - Test and refine flows without reprocessing completed samples
- **Cost Management** - Avoid wasting API credits by restarting from failures

**Configuration Options:**

```python
# Save checkpoints every N samples (recommended for large datasets)
result = flow.generate(
    dataset,
    checkpoint_dir="./checkpoints",
    save_freq=50  # Checkpoint after each 50 samples
)

# Only save final result (minimal overhead)
result = flow.generate(
    dataset,
    checkpoint_dir="./checkpoints"
    # No save_freq - only saves at completion
)

# Combine with other execution features
result = flow.generate(
    dataset,
    checkpoint_dir="./checkpoints",
    save_freq=100,
    max_concurrency=10
)
```

**Checkpoint Structure:**

Checkpoint directories contain:
- `checkpoint_NNNN.jsonl` - Completed sample batches in JSONL format
- `flow_metadata.json` - Flow ID, progress counters, and validation data

**Important Notes:**

- Checkpoints are flow-specific using `flow_id` to prevent mixing incompatible data
- Remaining samples are identified by comparing input dataset with completed samples using common columns
- If all samples are completed, Flow skips processing and returns merged results immediately
- Clean up checkpoint directories manually when no longer needed

## 📊 Flow Metrics and Reporting

SDG Hub automatically tracks and reports detailed execution metrics for every flow run, providing visibility into performance, data transformations, and success/failure status. This built-in monitoring system helps you understand bottlenecks, debug issues, and optimize your pipelines.

### Automatic Metrics Collection

The flow execution system automatically collects comprehensive metrics for each block without any configuration required:

**Collected Metrics:**
- **Block Identification** - Block name and type for clear tracking
- **Execution Time** - Precise timing for each block's execution
- **Row Changes** - Input and output row counts to track data filtering
- **Column Changes** - Added and removed columns to understand data transformations
- **Status** - Success or failure status for each block
- **Error Details** - Full error messages and types when blocks fail

### Rich Console Output

After every flow execution (whether successful or failed), a beautifully formatted summary table is automatically displayed in your terminal using the Rich library:

```python
from sdg_hub.core.flow import Flow
from datasets import Dataset

# Load and configure flow
flow = Flow.from_yaml("path/to/flow.yaml")
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1"
)

# Execute flow - metrics displayed automatically at completion
result = flow.generate(dataset)
```

**Example Console Output:**

```
┌─────────────────── Advanced Document Q&A Generation - Complete ───────────────────┐
│                          Flow Execution Summary                                     │
│ ┌──────────────────────┬─────────────────┬──────────┬──────────────┬─────────┬──┐│
│ │ Block Name           │ Type            │ Duration │ Rows         │ Columns │  ││
│ ├──────────────────────┼─────────────────┼──────────┼──────────────┼─────────┼──┤│
│ │ backup_document      │ DuplicateCol... │    0.05s │ 100 → 100    │ +1      │ ✓││
│ │ build_question_...   │ PromptBuilder...│    0.12s │ 100 → 100    │ +1      │ ✓││
│ │ generate_question    │ LLMChatBlock    │   45.30s │ 100 → 100    │ +1      │ ✓││
│ │ generate_answer      │ LLMChatBlock    │   78.45s │ 100 → 100    │ +1      │ ✓││
│ │ eval_faithfulness... │ LLMChatBlock    │   52.20s │ 100 → 100    │ +1      │ ✓││
│ │ extract_eval_con...  │ LLMParserBlock  │    0.15s │ 100 → 100    │ +2      │ ✓││
│ │ parse_evaluation     │ TextParserBlock │    0.22s │ 100 → 100    │ +2      │ ✓││
│ │ filter_faithful      │ ColumnValueF... │    0.08s │ 100 → 87     │ —       │ ✓││
│ ├──────────────────────┼─────────────────┼──────────┼──────────────┼─────────┼──┤│
│ │ TOTAL                │ 8 blocks        │  176.57s │ 87 final     │ 9 final │ ✓││
│ └──────────────────────┴─────────────────┴──────────┴──────────────┴─────────┴──┘│
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Table Columns Explained:**

| Column | Description |
|--------|-------------|
| **Block Name** | The unique name of the block as defined in the flow YAML |
| **Type** | The block class name (e.g., LLMChatBlock, PromptBuilderBlock) |
| **Duration** | Execution time in seconds for that specific block |
| **Rows** | Row transformation showing `input_count → output_count` |
| **Columns** | Column changes: `+N` for added, `-N` for removed, `+N/-M` for both |
| **Status** | `✓` for success, `✗` for failure |

**Status Indicators:**

The panel border color and title reflect the overall execution status:

- **Green border + "Complete"** - All blocks executed successfully
- **Red border + "Failed"** - Flow execution failed (exception thrown)
- **Yellow border + "Partial"** - Some blocks completed but others failed

### JSON Metrics Export

For production workflows, detailed metrics can be automatically saved to JSON files for analysis, monitoring, and debugging:

```python
# Enable JSON metrics export by providing a log directory
result = flow.generate(
    dataset,
    log_dir="./flow_logs"
)

# Metrics automatically saved to: ./flow_logs/{flow_name}_{timestamp}_metrics.json
```

**JSON Structure:**

```json
{
  "flow_name": "Advanced Document Q&A Generation",
  "flow_version": "2.1.0",
  "execution_timestamp": "20250113_143052",
  "execution_successful": true,
  "total_execution_time": 176.57,
  "total_wall_time": 178.23,
  "total_blocks": 8,
  "successful_blocks": 8,
  "failed_blocks": 0,
  "block_metrics": [
    {
      "block_name": "backup_document",
      "block_type": "DuplicateColumnsBlock",
      "execution_time": 0.05,
      "input_rows": 100,
      "output_rows": 100,
      "added_cols": ["original_document"],
      "removed_cols": [],
      "status": "success"
    },
    {
      "block_name": "generate_question",
      "block_type": "LLMChatBlock",
      "execution_time": 45.30,
      "input_rows": 100,
      "output_rows": 100,
      "added_cols": ["question"],
      "removed_cols": [],
      "status": "success"
    }
  ]
}
```

**JSON Fields Reference:**

| Field | Type | Description |
|-------|------|-------------|
| `flow_name` | string | Human-readable flow name from metadata |
| `flow_version` | string | Flow version string |
| `execution_timestamp` | string | Timestamp when execution started (YYYYMMDD_HHMMSS format) |
| `execution_successful` | boolean | `true` if all blocks succeeded, `false` if any failed |
| `total_execution_time` | float | Sum of all block execution times in seconds |
| `total_wall_time` | float | End-to-end wall clock time including overhead |
| `total_blocks` | integer | Number of blocks in the flow |
| `successful_blocks` | integer | Count of blocks that executed successfully |
| `failed_blocks` | integer | Count of blocks that failed |
| `block_metrics` | array | Detailed metrics for each block (see below) |

**Block Metrics Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `block_name` | string | Unique block identifier |
| `block_type` | string | Block class name |
| `execution_time` | float | Block execution duration in seconds |
| `input_rows` | integer | Number of rows received by the block |
| `output_rows` | integer | Number of rows produced by the block |
| `added_cols` | array | List of column names added by this block |
| `removed_cols` | array | List of column names removed by this block |
| `status` | string | `"success"` or `"failed"` |
| `error` | string | Error message (only present if `status` is `"failed"`) |
| `error_type` | string | Error class name (only present if `status` is `"failed"`) |

### Metrics Aggregation

When using checkpointing with `save_freq`, blocks may execute multiple times on different chunks of data. The metrics system automatically aggregates these executions per block:

- **Execution times** are summed across all chunks
- **Row counts** are totaled for input and output
- **Column changes** are merged (duplicates removed)
- **Status** reflects the worst case (any failure marks the block as failed)

This ensures the metrics summary and JSON export always show a cohesive view of the entire flow execution.

### Use Cases

**Performance Optimization:**
```python
# Identify slow blocks for optimization
result = flow.generate(dataset, log_dir="./optimization_analysis")
# Review metrics JSON to find blocks with high execution_time
```

**Data Quality Monitoring:**
```python
# Track how filtering affects dataset size
result = flow.generate(dataset)
# Check console output for row count changes: "100 → 87" indicates 13 filtered
```

**Production Monitoring:**
```python
# Continuous metrics collection for production pipelines
for batch in data_batches:
    result = flow.generate(
        batch,
        log_dir=f"./production_metrics/{date}",
        checkpoint_dir=f"./checkpoints/{batch_id}"
    )
# Aggregate metrics JSON files for dashboards and alerting
```

**Debugging Failed Runs:**
```python
# Automatic error capture in metrics
try:
    result = flow.generate(dataset, log_dir="./debug_logs")
except Exception as e:
    # Metrics JSON contains full error details for failed blocks
    print(f"Check ./debug_logs for detailed failure metrics")
```

### Important Notes

- **Always Displayed** - Metrics are shown even if the flow fails, helping debug issues
- **Zero Configuration** - No setup required, metrics collection is automatic
- **Minimal Overhead** - Metrics collection adds negligible performance impact
- **Thread-Safe** - Metrics are properly collected during concurrent block execution
- **Checkpoint Aware** - Metrics correctly aggregate across checkpointed chunks

## 🚀 Next Steps

Ready to master the flow system? Explore these detailed guides:

- **[Flow Discovery](discovery.md)** - Advanced discovery and organization techniques
- **[Custom Flows](custom-flows.md)** - Building your own sophisticated pipelines
- **[API Reference](../api-reference.md)** - Complete technical documentation