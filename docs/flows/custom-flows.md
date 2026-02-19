# Adding Custom Flows

Want to create a new flow? Follow these conventions:

### Directory Structure

```
flows/
└── {category}/              # e.g., knowledge_infusion, text_analysis
    └── {flow_name}/         # e.g., my_new_flow
        ├── flow.yaml             # Main flow definition (required)
        ├── README.md             # Flow-specific documentation
        └── prompts/              # Prompt configurations
            ├── prompt1.yaml
            └── prompt2.yaml
```

### Flow Metadata Requirements

Every `flow.yaml` must include:

```yaml
metadata:
  name: "Descriptive Flow Name"
  description: "Clear description of what this flow does..."
  version: "1.0.0"
  author: "Your Name"

  recommended_models:
    default: "model/name"
    compatible: ["alt1", "alt2"]

  tags:
    - "category-tag"
    - "purpose-tag"
    - "technique-tag"

  license: "Apache-2.0"

  dataset_requirements:
    required_columns:
      - "column1"
      - "column2"
    description: "Input dataset requirements..."

blocks:
  - block_type: "BlockName"
    block_config:
      block_name: "unique_name"
      # ... configuration
```

### Integration with Discovery

Flows are automatically discovered if:
1. Located in `src/sdg_hub/flows/`
2. Directory contains `flow.yaml`
3. Contains valid metadata section

No manual registration needed!