# Pre-Built Flows

Available flows in SDG Hub. Use `FlowRegistry.list_flows()` or `FlowRegistry.discover_flows()` to see the latest list.

## Discovering Flows

```python
from sdg_hub import FlowRegistry

# List all
for f in FlowRegistry.list_flows():
    print(f"- {f['name']} (tags: {f.get('tags', [])})")

# Search by tag or author
FlowRegistry.search_flows(tag="qa-generation")
FlowRegistry.search_flows(author="Red Hat")

# By category
FlowRegistry.get_flows_by_category()

# Rich table display
FlowRegistry.discover_flows()
```

---

## Knowledge Infusion / QA Generation

### English Multi-Summary QA (4 variants)

**Location:** `src/sdg_hub/flows/knowledge_infusion/enhanced_multi_summary_qa/`

Generate diverse QA pairs from documents using different summarization strategies. Each variant produces grounded question-answer pairs optimized for instruction tuning.

| Variant | Path | Strategy |
|---------|------|----------|
| Detailed Summary | `detailed_summary/flow.yaml` | Full detailed summary, then QA |
| Extractive Summary | `extractive_summary/flow.yaml` | Key sentences extraction, then QA |
| Key Facts | `key_facts/flow.yaml` | Atomic facts extraction, 5 QA pairs per fact |
| Document Direct QA | `doc_direct_qa/flow.yaml` | QA directly from document |

**Required input:** `document` column (text content)

```python
from sdg_hub import Flow, FlowRegistry
import pandas as pd

flow = Flow.from_yaml(FlowRegistry.get_flow_path("flow-id-here"))
flow.set_model_config(model="openai/gpt-4o-mini", api_key="sk-...")

df = pd.DataFrame({
    "document": ["Python was created by Guido van Rossum in 1991..."]
})

dry = flow.dry_run(df, sample_size=1)
if dry['execution_successful']:
    result = flow.generate(df)
```

### Spanish Multi-Summary QA (4 variants)

**Location:** `src/sdg_hub/flows/knowledge_infusion/enhanced_multi_summary_qa_es/`

Same as English variants but for Spanish language processing. Same 4 variants (detailed, extractive, key facts, doc direct).

### Japanese Multi-Summary QA

**Location:** `src/sdg_hub/flows/knowledge_infusion/japanese_multi_summary_qa/`

Advanced Japanese document-grounded QA generation using multiple LLM blocks.

**Required input:** `document` column (Japanese text)

---

## Text Analysis

### Structured Text Insights Extraction

**Location:** `src/sdg_hub/flows/text_analysis/structured_insights/`

Multi-step pipeline extracting structured insights from text: summary, keywords, entities, and sentiment combined into JSON output.

**Required input:** `text` column

```python
flow = Flow.from_yaml(FlowRegistry.get_flow_path("Structured Text Insights Extraction Flow"))
flow.set_model_config(model="openai/gpt-4o-mini", api_key="sk-...")

df = pd.DataFrame({"text": ["Climate change is accelerating..."]})
result = flow.generate(df)
# Output includes: summary, keywords, entities, sentiment
```

---

## Red Teaming

### Prompt Generation Flow

**Location:** `src/sdg_hub/flows/red_team/prompt_generation/`

Generates adversarial prompts for red-team testing across multiple harm categories using multi-dimensional sampling (demographics, expertise, geography, language styles, exploit stages, task medium, temporal, trust signals).

---

## Evaluation

### RAG Evaluation Dataset Flow

**Location:** `src/sdg_hub/flows/evaluation/rag_evaluation/`

Generates Q&A pairs for evaluating RAG systems.

---

## Agentic

### MCP Server Distillation

**Location:** `src/sdg_hub/flows/agentic/mcp_distillation/`

Generates high-quality tool-use training data through expert distillation. A frontier model actively explores the MCP server to understand tool behavior, then generates grounded questions and expert-quality trajectories.

Use this to create training data for teaching models to use MCP tools effectively.

---

## General Usage Pattern

```python
from sdg_hub import Flow, FlowRegistry
import pandas as pd

# 1. Find flow
flows = FlowRegistry.list_flows()

# 2. Load
flow = Flow.from_yaml(FlowRegistry.get_flow_path("Flow Name or ID"))

# 3. Inspect
flow.print_info()
reqs = flow.get_dataset_requirements()
print(f"Required: {reqs.required_columns}")
print(f"Default model: {flow.get_default_model()}")

# 4. Configure
flow.set_model_config(model="...", api_key="...")

# 5. Prepare and validate
df = pd.DataFrame({...})
errors = flow.validate_dataset(df)

# 6. Dry run
dry = flow.dry_run(df, sample_size=2)

# 7. Generate
result = flow.generate(df, checkpoint_dir="./ckpt", save_freq=100)

# 8. Save
result.to_parquet("output.parquet")
```
