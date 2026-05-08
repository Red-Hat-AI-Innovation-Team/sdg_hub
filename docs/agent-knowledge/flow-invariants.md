# Flow Invariants

Rules every flow YAML in SDG Hub must follow. A flow is a YAML-defined
pipeline that chains blocks into a data generation workflow.

## Required Top-Level Keys

Every flow YAML must have exactly two top-level keys:

```yaml
metadata:
  # ... flow metadata
blocks:
  # ... ordered list of block configurations
```

## Required Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Human-readable flow name |
| `author` | string | Who created or maintains this flow |
| `description` | string | What the flow does and when to use it |
| `tags` | list[string] | At least one tag for discoverability |

Optional but recommended: `version`, `recommended_models`, `license`,
`dataset_requirements`.

## Metadata Example

```yaml
metadata:
  name: "My Data Generation Flow"
  author: "SDG Hub Contributors"
  description: >-
    Generates question-answer pairs from source documents using
    multi-step summarization and LLM-powered extraction.
  version: "1.0.0"
  tags:
    - "knowledge-infusion"
    - "qa-generation"
```

## Flow Location

Flows live in `src/sdg_hub/flows/{category}/`. Each flow gets its own
subdirectory within a category.

### Flow Categories

| Category | Directory | Purpose |
|----------|-----------|---------|
| `agentic` | `src/sdg_hub/flows/agentic/` | Agent-based data generation (MCP distillation, etc.) |
| `evaluation` | `src/sdg_hub/flows/evaluation/` | RAG evaluation, MCP eval benchmarks |
| `knowledge_infusion` | `src/sdg_hub/flows/knowledge_infusion/` | QA generation from documents |
| `red_team` | `src/sdg_hub/flows/red_team/` | Adversarial prompt generation |
| `text_analysis` | `src/sdg_hub/flows/text_analysis/` | Structured text analysis pipelines |
| `code_evaluation` | `src/sdg_hub/flows/code_evaluation/` | Code benchmark generation |

## Block References

Every `block_type` value in the `blocks` section must reference a registered
block name. Use `BlockRegistry.discover_blocks()` to list all available blocks.

## Validation Checklist

Before merging a new flow:

- [ ] `metadata` key present with `name`, `author`, `description`, `tags`
- [ ] `tags` contains at least one entry
- [ ] `blocks` key present with at least one block
- [ ] Every `block_type` value is a registered block name
- [ ] `FlowRegistry.discover_flows()` finds the flow
- [ ] `flow.dry_run(dataset)` succeeds with a representative dataset
- [ ] Flow placed in the correct category directory
