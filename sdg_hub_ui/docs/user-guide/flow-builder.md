# User Guide: Flow Builder

The Flow Builder lets you create custom data generation pipelines by visually composing blocks.

## Interface Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ ← Back to Template Selection          Flow Builder              │
├──────────────────────────────────────────┬──────────────────────┤
│                                          │                      │
│           Flow Blocks (8)                │   Bundles            │
│                                          │   ┌────────────────┐ │
│   ┌───────────────────────────────────┐  │   │ QA Generation  │ │
│   │ ≡ 1 │ generate_questions │ LLM   │  │   │ Bundle         │ │
│   │     │ ChatCompletionBlock        │  │   └────────────────┘ │
│   │     │                    [↑][↓][✎][🗑]│ │   ┌────────────────┐ │
│   └───────────────────────────────────┘  │   │ Text Processing│ │
│   ┌───────────────────────────────────┐  │   │ Bundle         │ │
│   │ ≡ 2 │ evaluate_answers │ Eval   │   │   └────────────────┘ │
│   │     │ FaithfulnessBlock          │  │                      │
│   └───────────────────────────────────┘  │   Block Library      │
│                                          │   ┌────────────────┐ │
│   [Empty state when no blocks]           │   │ ChatCompletion │ │
│                                          │   │ TextParser     │ │
│                                          │   │ ColumnMapper   │ │
│                                          │   │ ...            │ │
│                                          │   └────────────────┘ │
└──────────────────────────────────────────┴──────────────────────┘
```

### Left Panel: Flow Blocks

Your current pipeline. Blocks execute in order from top to bottom.

- **Drag handle (≡)** — Reorder blocks via drag-and-drop
- **Position badge** — Shows execution order
- **Block name** — Your custom name for this block
- **Block type** — The underlying SDG Hub block class
- **Actions** — Move up/down, edit, delete

### Right Panel: Block Library

Add blocks from:

1. **Bundles** — Pre-configured block combinations
2. **Block Library** — Individual blocks by category

## Adding Blocks

### Using Bundles

Bundles are pre-configured sets of blocks for common patterns:

| Bundle | Blocks Included | Use Case |
|--------|-----------------|----------|
| **QA Generation** | PromptBuilder → ChatCompletion → TextParser | Generate Q&A pairs |
| **Document Summary** | PromptBuilder → ChatCompletion → TextParser | Summarize documents |
| **Evaluation** | FaithfulnessEval → RelevancyEval | Score generated content |

Click a bundle to add all its blocks at once.

### Using Individual Blocks

Browse blocks by category:

| Category | Block Types |
|----------|-------------|
| **LLM** | ChatCompletionBlock, PromptBuilderBlock |
| **Transform** | ColumnMapperBlock, TextParserBlock, JsonExtractorBlock |
| **Filtering** | QualityFilterBlock, DeduplicationBlock |
| **Evaluation** | FaithfulnessBlock, RelevancyBlock |

Click any block to open the configuration modal.

## Block Configuration

When adding or editing a block, the configuration modal appears:

### Basic Settings

| Field | Description |
|-------|-------------|
| **Block Name** | Unique identifier for this block instance |
| **Description** | Optional notes about this block's purpose |

### Block-Specific Configuration

Each block type has its own settings. Common examples:

**ChatCompletionBlock:**

```yaml
input_cols:
  - prompt
output_cols:
  - response
model_settings:
  temperature: 0.7
  max_tokens: 2048
```

**PromptBuilderBlock:**

```yaml
template_path: prompts/my_prompt.yaml
input_cols:
  - document
  - query
output_cols:
  - prompt
```

**ColumnMapperBlock:**

```yaml
mappings:
  old_column: new_column
  source: target
```

### Prompt Editing

For blocks that use prompt templates, click **Edit Prompt** to open the prompt editor:

```
┌─────────────────────────────────────────────────────────────┐
│ Edit Prompt: generate_questions                              │
├─────────────────────────────────────────────────────────────┤
│ Template (Jinja2):                                          │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ You are a helpful assistant that generates questions.   │ │
│ │                                                         │ │
│ │ Given the following document:                           │ │
│ │ {{ document }}                                          │ │
│ │                                                         │ │
│ │ Generate {{ num_questions }} thoughtful questions.      │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Variables Available:                                        │
│ • document (from dataset)                                   │
│ • num_questions (parameter)                                 │
│                                                             │
│                              [Cancel] [Save Prompt]         │
└─────────────────────────────────────────────────────────────┘
```

Prompts use **Jinja2 templating**:

- `{{ variable }}` — Insert variable value
- `{% if condition %}...{% endif %}` — Conditional content
- `{% for item in list %}...{% endfor %}` — Loops

## Reordering Blocks

Data flows through blocks sequentially. Reorder as needed:

### Drag and Drop

Click the grip handle (≡) and drag to new position.

### Arrow Buttons

Use ↑ and ↓ buttons to move one position at a time.

### Execution Order

```
Dataset → Block 1 → Block 2 → Block 3 → Output

Each block:
1. Receives output columns from previous blocks
2. Processes data according to its configuration
3. Adds its output columns for next blocks
```

## Saving Your Flow

### Save Flow Button

When you have at least one block, click **Save Flow** in the wizard footer.

### Metadata Form

Provide flow metadata:

| Field | Required | Description |
|-------|----------|-------------|
| **Name** | Yes | Flow display name |
| **Description** | No | What this flow does |
| **Version** | No | Semantic version (e.g., 1.0.0) |
| **Author** | No | Creator name |
| **Tags** | No | Searchable keywords |
| **Required Columns** | No | Dataset columns this flow needs |

### After Saving

Your custom flow:

1. Appears in the flow list with "(Custom)" suffix
2. Can be selected like any SDG Hub flow
3. Can be edited, cloned, or deleted
4. Is saved to `backend/custom_flows/[flow_name]/`

## Flow Files

Custom flows are saved as YAML:

```yaml
# custom_flows/my_flow/flow.yaml
metadata:
  name: My Custom Flow
  description: Generates Q&A pairs from documents
  version: 1.0.0
  author: Your Name
  tags:
    - question-generation
    - custom
  required_columns:
    - document
    - domain

blocks:
  - block_type: PromptBuilderBlock
    block_config:
      block_name: build_prompt
      template_path: prompts/qa_prompt.yaml
      input_cols:
        - document
      output_cols:
        - prompt

  - block_type: ChatCompletionBlock
    block_config:
      block_name: generate_qa
      input_cols:
        - prompt
      output_cols:
        - response
```

## Tips for Building Flows

### Design Principles

1. **Single Responsibility** — Each block does one thing well
2. **Clear Data Flow** — Output of one block feeds into next
3. **Meaningful Names** — Use descriptive block names
4. **Validate Early** — Add validation blocks before expensive LLM calls

### Common Patterns

**Generate and Evaluate:**

```
PromptBuilder → ChatCompletion → TextParser → Evaluator
```

**Multi-Stage Generation:**

```
Summary → QA Generation → Answer Verification → Scoring
```

**Filter and Transform:**

```
QualityFilter → ColumnMapper → Deduplication → Output
```

### Debugging Tips

1. **Start with bundles** — They're pre-tested
2. **Add blocks incrementally** — Test after each addition
3. **Use dry runs** — Validate with small samples
4. **Check column names** — Most errors are column mismatches
5. **Review prompts** — Ensure templates reference correct variables

## Next Steps

- [Model Configuration](model-configuration.md) — Configure your LLM
- [Dataset Configuration](dataset-configuration.md) — Set up your data
- [Running Generation](generation.md) — Execute your custom flow

