# Core Concepts

Understanding SDG Hub's architecture is key to building effective synthetic data pipelines. This guide covers the fundamental concepts that power the framework.

## 🧱 Blocks: The Building Blocks

**Blocks** are the atomic units of data processing in SDG Hub. Think of them as specialized Lego pieces that each perform a specific transformation on your dataset.

### Block Characteristics

Every block in SDG Hub:
- ✅ **Takes a dataset as input** and returns a transformed dataset
- ✅ **Has a specific purpose** (LLM chat, filtering, evaluation, etc.)
- ✅ **Is composable** - can be combined with other blocks
- ✅ **Is type-safe** - uses Pydantic validation for configuration
- ✅ **Includes monitoring** - provides detailed logging and progress tracking

### Block Categories

SDG Hub organizes blocks into logical categories:

| Category | Purpose | Examples |
|----------|---------|----------|
| **LLM** | Language model operations | Chat, prompt building, text parsing |
| **Transform** | Data manipulation | Column operations, text concatenation |
| **Filtering** | Quality control | Value-based filtering, threshold checks |
| **Evaluation** | Quality assessment | Faithfulness scoring, relevancy evaluation |

### Block Example
#TODO: Add block example

## 🌊 Flows: Orchestrating Pipelines

**Flows** are YAML-defined pipelines that orchestrate multiple blocks into complete data processing workflows.

### Flow Structure
#TODO: Add flow structure

### Flow Execution Model

Flows execute blocks sequentially:

```
Input Dataset → Block₁ → Block₂ → Block₃ → Final Dataset
```

Each block:
1. **Receives** the output dataset from the previous block
2. **Validates** that required input columns exist
3. **Processes** the data according to its configuration
4. **Outputs** a new dataset with additional/modified columns

### Flow Benefits

- **📋 Declarative Configuration** - Define pipelines in readable YAML
- **🔄 Reusability** - Save and share complete workflows
- **⚙️ Parameterization** - Customize behavior without code changes
- **🛡️ Validation** - Built-in checks for configuration and data compatibility
- **📊 Monitoring** - Execution tracking and performance metrics

## 🔍 Auto-Discovery System

SDG Hub automatically discovers and registers components with zero configuration.

### How Discovery Works

```python
from sdg_hub.core.flow import FlowRegistry
from sdg_hub.core.blocks import BlockRegistry

# Auto-discover everything
FlowRegistry.discover_flows()    # Scans src/sdg_hub/flows/
BlockRegistry.discover_blocks()  # Scans src/sdg_hub/core/blocks/

# Use discovered components
available_flows = FlowRegistry.list_flows()
available_blocks = BlockRegistry.list_blocks()
```

### Discovery Benefits

- **🚀 Zero Setup** - No manual registration required
- **🔎 Searchable** - Find components by name, tag, or category
- **📦 Modular** - Add new blocks/flows by dropping files in directories
- **🛡️ Type Safe** - Automatic validation of discovered components

## 🔧 Configuration System

SDG Hub uses a layered configuration approach for maximum flexibility.

### Model Configuration

The new v0.2 model configuration system provides runtime flexibility:

```python
# Discover what models work best for this flow
default_model = flow.get_default_model()
recommendations = flow.get_model_recommendations()

# Configure model at runtime
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="your_key"
)
```

## 📊 Data Flow Model

Understanding how data moves through SDG Hub is crucial for building effective pipelines.

### Dataset Requirements

SDG Hub operates on HuggingFace `datasets.Dataset` objects:

```python
from datasets import Dataset

# Create a dataset
dataset = Dataset.from_dict({
    "input_text": ["Hello", "World"],
    "metadata": ["info1", "info2"]
})
```

### Column Management

Blocks specify exactly which columns they need and produce:

```python
block_config = {
    "input_cols": ["question", "context"],     # Required input columns
    "output_cols": ["answer"],                 # New columns to create
    # Block will validate inputs and prevent output collisions
}
```

### Data Validation

Every block validates data at runtime:

- ✅ **Input Validation** - Ensures required columns exist
- ✅ **Output Validation** - Prevents column name collisions
- ✅ **Type Checking** - Validates data types where specified
- ✅ **Empty Dataset Handling** - Graceful handling of edge cases


## 🚀 Best Practices

### 1. Start Small
- Use `dry_run()` to test with small samples
- Validate your pipeline before scaling up

### 2. Layer Validation
- Use basic block composition (PromptBuilder → LLMChat → Parser → Filter) to assess quality
- Implement filtering to maintain data standards

### 3. Monitor Performance
- Watch execution logs for bottlenecks
- Use async-friendly blocks for LLM operations

### 4. Optimize for Scale
- Use `max_concurrency` parameter to control API request rates
- Start with conservative concurrency limits (5-10) for production
- Increase concurrency carefully while monitoring error rates
- Consider provider-specific rate limits and costs

### 5. Design for Reuse
- Create modular flows that can be combined
- Use parameters for customization points

## 🎯 Next Steps

Now that you understand the core concepts:

1. **[Explore Block Types](blocks/overview.md)** - Learn about specific block categories
2. **[Master Flow Configuration](flows/yaml-configuration.md)** - Deep dive into YAML structure
3. **[Build Custom Components](blocks/custom-blocks.md)** - Create your own blocks
4. **[Advanced Patterns](flows/custom-flows.md)** - Build sophisticated pipelines