# Common Flow Patterns

## Pattern 1: LLM Chain (Prompt → Generate → Parse)

Most common pattern for LLM-based generation.

```yaml
blocks:
  # Step 1: Build prompt from template
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_prompt"
      input_cols: ["document"]
      output_cols: "messages"
      prompt_config_path: "prompt.yaml"

  # Step 2: Call LLM
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate"
      input_cols: "messages"
      output_cols: "raw_response"
      temperature: 0.7
      async_mode: true

  # Step 3: Parse response
  - block_type: "TextParserBlock"
    block_config:
      block_name: "parse"
      input_cols: "raw_response"
      output_cols: ["field1", "field2"]
      pattern: "Field1:\\s*(.+?)\\s*Field2:\\s*(.+)"
```

**Test each step:**

```python
# play.py - Test chain step by step
from sdg_hub import Flow
import pandas as pd

flow = Flow.from_yaml("flow.yaml")
flow.set_model_config(model="openai/gpt-4", api_key="...")

df = pd.DataFrame({"document": ["Test document"]})

# Test with 1 sample
result = flow.generate(df)
print(result.columns)
print(result[["document", "field1", "field2"]])
```

---

## Pattern 2: Quality Filtering

Generate → Evaluate → Filter low quality.

```yaml
blocks:
  # Generate
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate"
      input_cols: "prompt"
      output_cols: "response"

  # Build evaluation prompt
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_eval_prompt"
      input_cols: ["response"]
      output_cols: "eval_prompt"
      prompt_config_path: "eval_prompt.yaml"

  # Evaluate quality
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "evaluate"
      input_cols: "eval_prompt"
      output_cols: "eval_response"
      temperature: 0.0   # Deterministic for evaluation

  # Parse score
  - block_type: "TextParserBlock"
    block_config:
      block_name: "parse_score"
      input_cols: "eval_response"
      output_cols: "score"
      pattern: "Score:\\s*(\\d+)"

  # Filter low quality
  - block_type: "ColumnValueFilterBlock"
    block_config:
      block_name: "filter_low_quality"
      input_cols: "score"
      filter_value: [4, 5]         # Keep only scores 4-5
      operation: "in"
      convert_dtype: "int"
```

**Evaluation prompt template:**

```yaml
# eval_prompt.yaml
system: |
  You evaluate response quality on a scale of 1-5.

user: |
  Response: {response}

  Rate the quality (1=poor, 5=excellent).
  Output: Score: <number>
```

---

## Pattern 3: Parallel Paths with Melt

Process same input multiple ways, then combine.

```yaml
blocks:
  # Duplicate source for parallel processing
  - block_type: "DuplicateColumnsBlock"
    block_config:
      block_name: "duplicate"
      input_cols: "document"
      output_cols: "base_document"

  # Path A: Detailed summary
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_detailed_prompt"
      input_cols:
        document: base_document
      output_cols: "detailed_prompt"
      prompt_config_path: "detailed_summary.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate_detailed"
      input_cols: "detailed_prompt"
      output_cols: "detailed_summary"

  # Path B: Brief summary
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_brief_prompt"
      input_cols:
        document: base_document
      output_cols: "brief_prompt"
      prompt_config_path: "brief_summary.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate_brief"
      input_cols: "brief_prompt"
      output_cols: "brief_summary"

  # Combine: Melt both summaries into rows
  - block_type: "MeltColumnsBlock"
    block_config:
      block_name: "melt_summaries"
      input_cols:
        - "detailed_summary"
        - "brief_summary"
      output_cols: "summary"
      id_vars:
        - "base_document"
```

**Result:** Each document becomes 2 rows (one per summary type).

---

## Pattern 4: Column Renaming Pipeline

Rename columns for clean output.

```yaml
blocks:
  # ... processing blocks ...

  # Final rename for output
  - block_type: "RenameColumnsBlock"
    block_config:
      block_name: "rename_output"
      input_cols:
        question: generated_question
        answer: generated_answer
        context: source_document
```

---

## Pattern 5: Multi-Step Extraction

Extract structured data in multiple passes.

```yaml
blocks:
  # First pass: Extract entities
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_entity_prompt"
      input_cols: ["text"]
      output_cols: "entity_prompt"
      prompt_config_path: "extract_entities.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "extract_entities"
      input_cols: "entity_prompt"
      output_cols: "entities_raw"

  - block_type: "LLMParserBlock"
    block_config:
      block_name: "parse_entities"
      input_cols: "entities_raw"
      field_prefix: "entity_"
      extract_content: true

  # Second pass: Extract relationships using entities
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_relation_prompt"
      input_cols: ["text", "entity_names"]   # Use extracted entities
      output_cols: "relation_prompt"
      prompt_config_path: "extract_relations.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "extract_relations"
      input_cols: "relation_prompt"
      output_cols: "relations_raw"
```

---

## Pattern 6: Conditional Processing

Use filtering to create conditional branches.

```yaml
blocks:
  # Classify first
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "classify"
      input_cols: "classify_prompt"
      output_cols: "category"

  # Filter for specific category
  - block_type: "ColumnValueFilterBlock"
    block_config:
      block_name: "filter_category_a"
      input_cols: "category"
      filter_value: ["category_a"]
      operation: "eq"

  # Process only category_a items
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "process_category_a"
      input_cols: "process_prompt"
      output_cols: "result"
```

**Note:** This filters out other categories. For true branching, use separate flows or Python logic.

---

## Testing Patterns

```python
# play.py - Test any pattern
import pandas as pd
from sdg_hub import Flow

flow = Flow.from_yaml("flow.yaml")
flow.set_model_config(model="openai/gpt-4", api_key="...")

# Small test data
df = pd.DataFrame({
    "document": ["Test document 1", "Test document 2"]
})

# Always dry run first
dry = flow.dry_run(df, sample_size=2)
print(f"Success: {dry['execution_successful']}")

for b in dry['blocks_executed']:
    print(f"  {b['block_name']}: {b['input_rows']} -> {b['output_rows']} rows")

# Full run if successful
if dry['execution_successful']:
    result = flow.generate(df)
    print("\nOutput columns:", list(result.columns))
    print(result.head())
```
