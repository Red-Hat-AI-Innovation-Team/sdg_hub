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

#TODO: Add metadata fields information

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

## 🚀 Next Steps

Ready to master the flow system? Explore these detailed guides:

- **[Flow Discovery](discovery.md)** - Advanced discovery and organization techniques
- **[Custom Flows](custom-flows.md)** - Building your own sophisticated pipelines
- **[API Reference](../api-reference.md)** - Complete technical documentation