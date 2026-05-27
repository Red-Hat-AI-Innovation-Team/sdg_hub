---
name: data-generation
description: "Use when the user wants to generate synthetic data, create training datasets, run data generation pipelines, build custom flows, or produce synthetic data from documents. Applies to tasks like: QA generation, knowledge infusion, red-teaming, RAG evaluation, MCP distillation, code evaluation."
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_generate.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_flows.sh:*)"]
---

# Synthetic Data Generation

Help the user generate synthetic data using sdg_hub flows.

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh"
```

### If not ready

- `library=missing` and `config=missing`: invoke the `setup-guide` skill.
- `library=installed` and `config=missing`: tell the user to run the `setup-guide` skill to configure.

### If ready (`library=installed`, `config=found`)

Proceed to Step 2.

## Step 2: Approach Selection

Help the user choose the right approach:

| User wants | Approach | Next step |
|---|---|---|
| Use an existing pipeline | Pre-built flow | Invoke `flow-browser` skill, then run generation |
| Build a custom pipeline in Python | Custom Python | Provide code example using `BaseBlock` and `Flow` |
| Define a pipeline in YAML | Custom YAML flow | Help author YAML, then run generation |
| Agent-based generation (Langflow, LangGraph) | Agent pipeline | Help integrate with agent frameworks |
| MCP tool-use distillation | MCP pipeline | Help set up MCP agent block |

## Step 3: Flow Detection

If the user mentions a specific use case, suggest the matching flow:

| Use case | Flow category |
|---|---|
| "knowledge", "QA", "question answering" | Knowledge generation flows |
| "red team", "safety", "adversarial" | Red-teaming flows |
| "RAG", "retrieval", "evaluation" | RAG evaluation flows |
| "code", "programming", "evaluation" | Code evaluation flows |
| "MCP", "tool calling", "distillation" | MCP distillation flows |

Search for matching flows:
```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_flows.sh" search "<category>"
```

## Step 4: Execute Generation

If the user doesn't specify a flow, suggest they invoke the `flow-browser` skill first.

Recommend starting with `--sample 2` for a dry run: "I suggest running a dry-run with 2 samples first to verify the flow works before processing the full dataset."

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_generate.sh" $ARGUMENTS
```

## Step 5: Present Results

1. **Generation status** — Whether generation completed successfully
2. **Row counts** — Input rows processed and output rows generated
3. **Output location** — Path to the generated dataset
4. **Errors** — If any rows failed, report the count and suggest checking the output for error entries

If generation failed, suggest troubleshooting:
- Missing columns: Check that the input dataset has the columns required by the flow
- API errors: Verify API key and endpoint in `.sdg-hub/config.json`
- Rate limiting: Reduce concurrency or add delay

## Custom Flow Authoring

If the user wants a custom flow, help them:
1. Identify the blocks they need from the BlockRegistry
2. Chain blocks in a YAML pipeline definition
3. Validate with `FlowValidator`
4. Test with `--sample 2` before full generation
