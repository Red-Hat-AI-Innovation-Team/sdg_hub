# InstructLab Knowledge Q&A Generation

Generates high-quality question-answer training data for [InstructLab](https://instructlab.ai/) knowledge contributions from document chunks.

## Overview

This flow takes pre-chunked documents with taxonomy metadata and produces diverse, grounded Q&A pairs suitable for InstructLab's `qna.yaml` schema. Each document chunk yields multiple Q&A pairs that are independently validated for faithfulness to the source material.

```
document_text + taxonomy_path + domain
        │
        ▼
   Generate Questions ──── 5 diverse questions per chunk
        │
        ▼
   Generate Answers ────── grounded in source document
        │
        ▼
   Evaluate Faithfulness ─ LLM judges answer quality
        │
        ▼
   Filter Unfaithful ───── keep only faithful Q&A pairs
```

## Input Dataset

| Column | Type | Description |
|--------|------|-------------|
| `document_text` | str | Pre-chunked document text (300-500 words recommended) |
| `taxonomy_path` | str | InstructLab taxonomy path (e.g., `knowledge/science/physics`) |
| `domain` | str | Knowledge domain name (e.g., `physics`) |

## Output Columns

| Column | Description |
|--------|-------------|
| `question` | Generated question |
| `answer` | Grounded answer from document context |
| `document_text` | Source document chunk (preserved for reference) |
| `domain` | Knowledge domain |
| `taxonomy_path` | InstructLab taxonomy placement |
| `faithfulness_explanation` | LLM explanation of faithfulness judgment |
| `faithfulness_judgment` | `YES` (all unfaithful pairs are filtered out) |

## Usage

```python
from sdg_hub import Flow

# Load the flow
flow = Flow.from_registry("bright-coral-421")

# Or load from YAML
flow = Flow.from_yaml("src/sdg_hub/flows/knowledge_infusion/instructlab_qna/flow.yaml")

# Configure your model
flow.set_model("openai/gpt-oss-120b")

# Run on your dataset
output_df = flow.generate(input_df)
```

## Post-Processing to qna.yaml

The output DataFrame can be grouped by `taxonomy_path` and formatted into InstructLab's `qna.yaml` schema:

```python
import yaml

for path, group in output_df.groupby("taxonomy_path"):
    qna_yaml = {
        "version": 3,
        "domain": group["domain"].iloc[0],
        "task_description": f"Teach the model about {group['domain'].iloc[0]}",
        "created_by": "sdg-hub",
        "seed_examples": [
            {"question": row["question"], "answer": row["answer"], "context": row["document_text"]}
            for _, row in group.iterrows()
        ],
    }
    print(yaml.dump(qna_yaml, default_flow_style=False))
```

## Question Categories

The question generation prompt produces questions across five categories:

- **Definitional** — What is X? How would you define X?
- **Procedural** — What are the steps for X? How do you perform X?
- **Troubleshooting** — What are common problems with X?
- **Comparative** — How does X compare to alternatives?
- **Best practices** — What do experts recommend for X?

## Quality Assurance

Every generated Q&A pair goes through a faithfulness evaluation step where an LLM judges whether the answer is supported by the source document. Only pairs judged as faithful (`YES`) are kept in the final output.
