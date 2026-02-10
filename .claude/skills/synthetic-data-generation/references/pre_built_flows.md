# Pre-Built Flows

Available flows in SDG Hub and how to use them.

## Discovering Flows

```python
# play.py
from sdg_hub import FlowRegistry

# List all
for flow in FlowRegistry.list_flows():
    print(f"\n{flow['name']}")
    print(f"  Tags: {flow.get('tags', [])}")
    print(f"  Path: {flow['path']}")

# Search by tag
FlowRegistry.search_flows(tag="qa-generation")
FlowRegistry.search_flows(tag="summarization")
```

## Text Analysis Flows

### Structured Text Insights Extraction Flow

**Location:** `src/sdg_hub/flows/text_analysis/structured_insights/`

**Purpose:** Extract structured insights from text documents.

**Input:**
```python
df = pd.DataFrame({
    "text": ["Your document text here..."]
})
```

**Output:** Summary, keywords, entities, sentiment

**Usage:**
```python
# play.py
from sdg_hub import Flow, FlowRegistry
import pandas as pd

flow = Flow.from_yaml(
    FlowRegistry.get_flow_path("Structured Text Insights Extraction Flow")
)

flow.set_model_config(
    model="openai/gpt-4o-mini",
    api_key="sk-..."
)

df = pd.DataFrame({
    "text": ["Climate change is accelerating global warming..."]
})

result = flow.generate(df)
print(result.columns)
# ['text', 'summary', 'keywords', 'entities', 'sentiment']
```

## QA Generation Flows

### Document Grounded QA

**Location:** `src/sdg_hub/flows/qa_generation/document_grounded_qa/`

**Purpose:** Generate question-answer pairs grounded in documents.

**Input:**
```python
df = pd.DataFrame({
    "document": ["Document content here..."],
    "domain": ["science"]  # Optional
})
```

**Output:** question, response

### Multi-Summary QA (InstructLab)

**Location:** `src/sdg_hub/flows/qa_generation/document_grounded_qa/multi_summary_qa/instructlab/`

**Purpose:** Generate diverse QA pairs using multiple summary strategies.

**Features:**
- Multiple summary types (detailed, extractive, atomic facts)
- Quality evaluation and filtering
- High-quality QA pairs for instruction tuning

**Input:**
```python
df = pd.DataFrame({
    "document": ["Long document content..."],
    "domain": ["technology"]
})
```

**Usage:**
```python
# play.py
from sdg_hub import Flow, FlowRegistry
import pandas as pd

flow = Flow.from_yaml(
    FlowRegistry.get_flow_path("Advanced Document Grounded QA")
)

flow.set_model_config(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY"
)

df = pd.DataFrame({
    "document": ["Python was created by Guido van Rossum..."],
    "domain": ["programming"]
})

# Dry run first
dry = flow.dry_run(df, sample_size=1)
print(f"Success: {dry['execution_successful']}")

# Full run
result = flow.generate(df)
print(result[["document", "question", "response"]])
```

## Evaluation Flows

### RAG Evaluation

**Location:** `src/sdg_hub/flows/evaluation/rag/`

**Purpose:** Evaluate RAG system outputs for faithfulness and relevancy.

**Input:**
```python
df = pd.DataFrame({
    "query": ["User question"],
    "context": ["Retrieved context"],
    "response": ["RAG system response"]
})
```

**Output:** Evaluation scores

## Using Any Flow

### General Pattern

```python
# play.py
from sdg_hub import Flow, FlowRegistry
import pandas as pd

# 1. Find flow
flows = FlowRegistry.list_flows()
# Pick one...

# 2. Load
flow = Flow.from_yaml(FlowRegistry.get_flow_path("Flow Name"))

# 3. Check requirements
flow.print_info()
reqs = flow.get_dataset_requirements()
print(f"Required: {reqs.required_columns}")

# 4. Configure model
flow.set_model_config(model="...", api_key="...")

# 5. Prepare data
df = pd.DataFrame({...})

# 6. Validate
errors = flow.validate_dataset(df)
assert not errors, errors

# 7. Dry run
dry = flow.dry_run(df, sample_size=2)
assert dry['execution_successful']

# 8. Generate
result = flow.generate(df)

# 9. Save
result.to_parquet("output.parquet")
```

### Exploring Flow Structure

```python
# play.py
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")

# Print detailed info
flow.print_info()

# Get metadata
print(f"Name: {flow.metadata.name}")
print(f"Version: {flow.metadata.version}")
print(f"Default model: {flow.get_default_model()}")

# Get block list
for block in flow.blocks:
    print(f"  {block.block_name} ({block.block_type})")
```
