---
name: data-generation
description: "Use when the user wants to generate synthetic data, create training datasets, run data generation pipelines, build custom flows, or produce synthetic data from documents. Applies to tasks like: QA generation, knowledge infusion, red-teaming, RAG evaluation, MCP distillation, code evaluation."
---

# Synthetic Data Generation

Help the user generate synthetic data using sdg_hub flows.

## Detection

First, check the environment:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh"
```

## Routing

Based on detection results:

### Nothing available (`library=missing`, `config=missing`)
Invoke the `setup-guide` skill to walk through installation and configuration.

### Config missing but library installed (`library=installed`, `config=missing`)
Ask the user to run `/sdg-setup` to configure, or invoke the `setup-guide` skill.

### Ready (`library=installed`, `config=found`)
Proceed based on the user's intent.

## Approach Selection

Help the user choose the right approach:

| User wants | Approach | Route to |
|---|---|---|
| Use an existing pipeline | Pre-built flow | `/sdg-flows` then `/sdg-generate` |
| Build a custom pipeline in Python | Custom Python | Provide code example using `BaseBlock` and `Flow` |
| Define a pipeline in YAML | Custom YAML flow | Help author YAML, then `/sdg-generate` |
| Agent-based generation (Langflow, LangGraph) | Agent pipeline | Help integrate with agent frameworks |
| MCP tool-use distillation | MCP pipeline | Help set up MCP agent block |

## Flow Detection

If the user mentions a specific use case, suggest the matching flow:

| Use case | Flow category |
|---|---|
| "knowledge", "QA", "question answering" | Knowledge generation flows |
| "red team", "safety", "adversarial" | Red-teaming flows |
| "RAG", "retrieval", "evaluation" | RAG evaluation flows |
| "code", "programming", "evaluation" | Code evaluation flows |
| "MCP", "tool calling", "distillation" | MCP distillation flows |

Route to `/sdg-flows search <category>` to find specific flows.

## Generation Execution

For generation requests with a known flow and input file, route to `/sdg-generate`.

Recommend starting with `--sample 2` for a dry run: "I suggest running a dry-run with 2 samples first to verify the flow works before processing the full dataset."

## Custom Flow Authoring

If the user wants a custom flow, help them:
1. Identify the blocks they need from the BlockRegistry
2. Chain blocks in a YAML pipeline definition
3. Validate with `FlowValidator`
4. Test with `--sample 2` before full generation
