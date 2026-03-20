# Block System Overview

The block system is the foundation of SDG Hub's composable architecture. Blocks are self-contained, reusable units that transform datasets in specific ways, designed to work together like building blocks.

## 🏗️ Architecture Philosophy

Every block in SDG Hub follows the same fundamental pattern:

```python
# Universal block interface
input_dataset → Block.generate() → output_dataset
```

This consistency enables:
- **🔄 Composability** - Chain any blocks together
- **🛡️ Predictability** - All blocks follow the same interface
- **🔍 Debuggability** - Consistent logging and error handling
- **⚡ Performance** - Optimized execution patterns

## 📋 BaseBlock Foundation

All blocks inherit from `BaseBlock`, which provides:

### Core Features
- **Type Safety** - Pydantic-based validation for all configuration
- **Column Management** - Automatic input/output column validation
- **Rich Logging** - Detailed progress tracking and summaries
- **Error Handling** - Comprehensive validation and error reporting
- **Serialization** - Save and load block configurations

### Standard Configuration
```python
# Import the specific block you need
from sdg_hub.core.blocks import LLMChatBlock

# Every block has these standard fields
block = LLMChatBlock(
    block_name="my_unique_block",     # Required: unique identifier
    input_cols=["input_text"],        # Column this block needs
    output_cols=["response"],         # Column this block creates
    model="openai/gpt-4o",            # Required: provider/model format
    # ... block-specific configuration
)
```

## 🗂️ Block Categories

SDG Hub organizes blocks into logical categories based on their functionality:

### 🧠 LLM Blocks (`llm/`)
AI-powered blocks for language model operations:
- **LLMChatBlock** - Direct chat with language models
- **PromptBuilderBlock** - Construct prompts from templates
- **TextParserBlock** - Extract structured data from LLM responses
- **JSONParserBlock** - Parse JSON responses into structured columns

### 🔄 Transform Blocks (`transform/`)
Data manipulation and transformation:
- **DuplicateColumnsBlock** - Copy columns with new names
- **RenameColumnsBlock** - Rename existing columns
- **TextConcatBlock** - Concatenate text from multiple columns
- **IndexBasedMapperBlock** - Map values based on indices
- **MeltColumnsBlock** - Reshape wide data to long format
- **SamplerBlock** - Randomly sample values from list or weighted pools

### 🔍 Filtering Blocks (`filtering/`)
Quality control and data validation:
- **ColumnValueFilterBlock** - Filter rows based on column values


## 🔧 Block Lifecycle

### 1. Discovery and Registration
```python
from sdg_hub.core.blocks import BlockRegistry

# Auto-discover all blocks (happens automatically)
BlockRegistry.discover_blocks()

# List available blocks
available_blocks = BlockRegistry.list_blocks()
print(f"Found {len(available_blocks)} blocks")
```

### 2. Block Instantiation
```python
# Import the specific block you need
from sdg_hub.core.blocks import LLMChatBlock

# Create an instance with configuration
chat_block = LLMChatBlock(
    block_name="question_answerer",
    model="openai/gpt-4o",
    input_cols=["question"],
    output_cols=["answer"],
    prompt_template="Answer this question: {question}"
)
```

### 3. Validation and Execution
```python
from datasets import Dataset

# Create input dataset
dataset = Dataset.from_dict({
    "question": ["What is Python?", "Explain machine learning"]
})

# Block automatically validates and processes
result = chat_block.generate(dataset)
# or equivalently: result = chat_block(dataset)
```

### 4. Monitoring and Logging

Every block automatically produces Rich-formatted panels on execution showing input/output summaries:

```
┌──────────────────── question_answerer ────────────────────┐
│ 📊 Processing Input Data                                  │
│ Block Type: LLMChatBlock                                  │
│ Input Rows: 2                                             │
│ Input Columns: 1                                          │
│ Column Names: question                                    │
│ Expected Output Columns: answer                           │
└───────────────────────────────────────────────────────────┘

┌──────────── question_answerer - Complete ─────────────────┐
│ ✅ Processing Complete                                     │
│ Rows: 2 → 2                                               │
│ Columns: 1 → 2                                            │
│ 🟢 Added: answer                                          │
│ 📋 Final Columns: answer, question                        │
└───────────────────────────────────────────────────────────┘
```

**Controlling log level:**

SDG Hub uses Python's standard logging with a Rich handler. Set the `LOG_LEVEL` environment variable to control verbosity:

```bash
# Show all logs (default)
LOG_LEVEL=INFO python my_script.py

# Only warnings and errors
LOG_LEVEL=WARNING python my_script.py

# Debug output (includes per-sample progress)
LOG_LEVEL=DEBUG python my_script.py

# Reduce LiteLLM noise separately
LITELLM_LOG_LEVEL=ERROR python my_script.py
```

**Saving logs to file:**

When running flows, pass `log_dir` to `flow.generate()` to save execution metrics to JSON:

```python
result = flow.generate(
    dataset,
    log_dir="./logs"  # Saves {flow_name}_{timestamp}_metrics.json
)
```

The metrics JSON includes per-block execution time, row counts, column changes, and error details. See [Flow Metrics and Reporting](../flows/overview.md#-flow-metrics-and-reporting) for the full specification.

## 🛡️ Built-in Validation

### Input Validation
- **Column Existence** - Ensures required input columns are present
- **Data Type Checking** - Validates expected data types
- **Empty Dataset Handling** - Graceful handling of edge cases

### Output Validation  
- **Column Collision Prevention** - Prevents overwriting existing columns
- **Schema Consistency** - Ensures output matches expected structure
- **Data Integrity** - Validates output data quality

### Example Validation
```python
# This will raise MissingColumnError
dataset = Dataset.from_dict({"wrong_column": ["data"]})
block = SomeBlock(input_cols=["required_column"])
result = block.generate(dataset)  # ❌ Error!

# This will raise OutputColumnCollisionError
dataset = Dataset.from_dict({"existing_col": ["data"]})  
block = SomeBlock(output_cols=["existing_col"])
result = block.generate(dataset)  # ❌ Error!
```

## 🚀 Next Steps

Ready to dive deeper? Explore specific block categories:

- **[LLM Blocks](blocks/llm-blocks.md)** - AI-powered language model operations
- **[Transform Blocks](blocks/transform-blocks.md)** - Data manipulation and reshaping
- **[Filtering Blocks](blocks/filtering-blocks.md)** - Quality control and validation
- **[Custom Blocks](blocks/custom-blocks.md)** - Build your own processing blocks
