# Quick Start Guide

Get up and running with SDG Hub in minutes! This guide walks through discovering flows, running your first pipeline, and understanding the basic workflow.

## 🔍 Step 1: Discover Available Components

SDG Hub automatically discovers all available blocks and flows - no manual setup required!

```python
from sdg_hub.core.flow import FlowRegistry
from sdg_hub.core.blocks import BlockRegistry

# Auto-discover all components
FlowRegistry.discover_flows()
BlockRegistry.discover_blocks()

# See what's available
print("📋 Available Flows:")
for flow_name in FlowRegistry.list_flows():
    print(f"  • {flow_name}")

print("\n🧱 Available Blocks:")
for block_name in BlockRegistry.list_blocks():
    print(f"  • {block_name}")
```

## 🚀 Step 2: Run Your First Flow

Let's use the built-in document-grounded QA generation flow:

```python
from sdg_hub.core.flow import FlowRegistry, Flow
from datasets import Dataset

# Load a pre-built flow
flow_name = "Extractive Summary Knowledge Tuning Dataset Generation Flow"
flow_path = FlowRegistry.get_flow_path(flow_name)
flow = Flow.from_yaml(flow_path)

# Discover recommended models
default_model = flow.get_default_model()
recommendations = flow.get_model_recommendations()

# Configure model settings at runtime
flow.set_model_config(
    model=f"hosted_vllm/{default_model}",
    api_base="http://localhost:8000/v1",
    api_key="your_key",
)

# Create a simple dataset
dataset = Dataset.from_dict({
    'document': ['Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.'],
    'document_outline': ['1. Python Introduction; 2. Programming Paradigms; 3. Language Features'],
    'domain': ['Computer Science'],
    'icl_document': ['Java is an object-oriented programming language that runs on the Java Virtual Machine.'],
    'icl_query_1': ['What type of language is Java?'],
    'icl_query_2': ['Where does Java run?'],
    'icl_query_3': ['What are the benefits of Java?']
})

# Test with a small sample AND get time estimate (recommended!)
print("🧪 Running dry run with time estimation...")
dry_result = flow.dry_run(dataset, sample_size=5, enable_time_estimation=True, max_concurrency=100)
print(f"✅ Dry run completed in {dry_result['execution_time_seconds']:.2f}s")
print(f"📊 Output columns: {list(dry_result['final_dataset']['columns'])}")

# Time estimation is automatically displayed in a Rich table format
# The table shows estimated time, total API calls, and per-block breakdowns
```

## 📊 Step 3: Generate Synthetic Data

Once the dry run succeeds, generate the full dataset:

```python
# Configure the model before generation
print("🔧 Configuring model...")
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="your_key",
)

# Generate high-quality QA pairs
print("🏗️ Generating synthetic data...")
result = flow.generate(dataset)

# Explore the results
print(f"\n📈 Generated {len(result)} QA pairs!")
print(f"📝 Sample Question: {result['question'][0]}")
print(f"💬 Sample Answer: {result['response'][0]}")
print(f"🎯 Faithfulness Score: {result['faithfulness_judgment'][0]}")
```

## 🔧 Step 5: Search and Filter Components

Find exactly what you need:

```python
# Search for specific types of flows
qa_flows = FlowRegistry.search_flows(tag="question-generation")
print(f"🔎 QA Generation Flows: {qa_flows}")

# Search for evaluation flows  
eval_flows = FlowRegistry.search_flows(tag="evaluation")
print(f"📊 Evaluation Flows: {eval_flows}")

# List all blocks by categories
all_blocks = BlockRegistry.list_blocks(grouped=True)
for category, blocks in all_blocks.items():
    print(f"Blocks for category {category}: {blocks}")

# Find blocks by category
llm_blocks = BlockRegistry.list_blocks(category="llm")
print(f"🧠 LLM Blocks: {llm_blocks}")

transform_blocks = BlockRegistry.list_blocks(category="transform") 
print(f"🔄 Transform Blocks: {transform_blocks}")
```

## ⚙️ Step 6: Model Configuration

SDG Hub provides a flexible model configuration system for runtime setup:

### Discover Model Recommendations
```python
# Get the recommended default model for this flow
default_model = flow.get_default_model()
print(f"🎯 Default model: {default_model}")

# See all model recommendations
recommendations = flow.get_model_recommendations()
print(f"💡 Recommended models: {recommendations}")
```

### Configure Models
```python
# Configure model settings dynamically
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="your_key",
)

# Alternative: Use cloud providers
flow.set_model_config(
    model="gpt-4o",
    api_key="your-openai-key",
)

# Or use environment variables (still supported)
# OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
```

### Flow Runtime Parameters

Customize block behavior at runtime without modifying flow YAML files:

```python
# Global parameters (apply to all compatible blocks)
result = flow.generate(
    dataset,
    runtime_params={
        "temperature": 0.7,
        "max_tokens": 200,
    }
)

# Block-specific configuration
result = flow.generate(
    dataset,
    runtime_params={
        "question_generator": {"temperature": 0.9, "max_tokens": 100},
        "answer_generator": {"temperature": 0.5, "max_tokens": 300},
        "text_parser": {"start_tags": ["<answer>"], "end_tags": ["</answer>"]},
        "quality_filter": {"filter_value": 0.9, "operation": "ge"}
    }
)
```

Runtime parameters work with any block type - LLM blocks, parser blocks, filter blocks, etc. For detailed parameter options by block type, see [Flow Execution](flows/overview.md#-flow-execution).

### Error Handling
#TODO: Add error handling


## 🚀 Next Steps

Now that you're familiar with the basics:

1. **[Understand Core Concepts](concepts.md)** - Deep dive into blocks and flows
2. **[Explore Block Types](blocks/overview.md)** - Learn about different block categories  
3. **[Build Custom Flows](flows/custom-flows.md)** - Create your own pipelines
4. **[API Reference](api-reference.md)** - Complete technical documentation

Happy building! 🎉