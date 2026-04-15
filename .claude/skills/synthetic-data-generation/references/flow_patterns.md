# Common Flow Patterns

Reusable composition patterns for building flows. Each pattern includes the YAML and a test script.

## Pattern 1: LLM Chain (Prompt -> Generate -> Parse)

The most common pattern. Build a prompt, call an LLM, parse the response.

```yaml
blocks:
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_prompt"
      input_cols: ["document"]
      output_cols: "messages"
      prompt_config_path: "prompt.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate"
      input_cols: "messages"
      output_cols: "raw_response"
      temperature: 0.7
      async_mode: true

  - block_type: "TagParserBlock"
    block_config:
      block_name: "parse"
      input_cols: "raw_response"
      output_cols: ["question", "answer"]
      start_tags: ["<question>", "<answer>"]
      end_tags: ["</question>", "</answer>"]
```

---

## Pattern 2: Quality Filtering

Generate -> Evaluate -> Filter low quality.

```yaml
blocks:
  # Generate content
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate"
      input_cols: "messages"
      output_cols: "response"

  # Build evaluation prompt
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_eval"
      input_cols: ["response"]
      output_cols: "eval_messages"
      prompt_config_path: "eval_prompt.yaml"

  # Score quality (use temperature=0 for deterministic evaluation)
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "evaluate"
      input_cols: "eval_messages"
      output_cols: "eval_response"
      temperature: 0.0

  # Parse score
  - block_type: "RegexParserBlock"
    block_config:
      block_name: "parse_score"
      input_cols: "eval_response"
      output_cols: "score"
      pattern: "Score:\\s*(\\d+)"

  # Keep only high quality (4-5)
  - block_type: "ColumnValueFilterBlock"
    block_config:
      block_name: "filter"
      input_cols: "score"
      filter_value: [4, 5]
      operation: "in"
      convert_dtype: "int"
```

---

## Pattern 3: Parallel Paths with Melt

Process the same input multiple ways, then combine into rows.

```yaml
blocks:
  # Keep original for reference
  - block_type: "DuplicateColumnsBlock"
    block_config:
      block_name: "dup"
      input_cols: "document"
      output_cols: "base_document"

  # Path A: Detailed summary
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "prompt_detailed"
      input_cols:
        document: base_document
      output_cols: "detailed_msgs"
      prompt_config_path: "detailed.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "gen_detailed"
      input_cols: "detailed_msgs"
      output_cols: "detailed_summary"

  # Path B: Brief summary
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "prompt_brief"
      input_cols:
        document: base_document
      output_cols: "brief_msgs"
      prompt_config_path: "brief.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "gen_brief"
      input_cols: "brief_msgs"
      output_cols: "brief_summary"

  # Combine: each doc becomes 2 rows
  - block_type: "MeltColumnsBlock"
    block_config:
      block_name: "melt"
      input_cols: ["detailed_summary", "brief_summary"]
      output_cols: "summary"
      id_vars: ["base_document"]
```

---

## Pattern 4: Multi-Step Extraction

Extract structured data in multiple LLM passes, where later passes use earlier results.

```yaml
blocks:
  # Pass 1: Extract entities
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "entity_prompt"
      input_cols: ["text"]
      output_cols: "entity_msgs"
      prompt_config_path: "extract_entities.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "extract_entities"
      input_cols: "entity_msgs"
      output_cols: "entities_raw"

  - block_type: "JSONParserBlock"
    block_config:
      block_name: "parse_entities"
      input_cols: "entities_raw"
      output_cols: ["entity_names", "entity_types"]

  # Pass 2: Extract relationships using entities from pass 1
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "relation_prompt"
      input_cols: ["text", "entity_names"]
      output_cols: "relation_msgs"
      prompt_config_path: "extract_relations.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "extract_relations"
      input_cols: "relation_msgs"
      output_cols: "relations_raw"
```

---

## Pattern 5: Agent Integration

Use an external agent framework as a pipeline step.

```yaml
blocks:
  # Prepare input for agent
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_query"
      input_cols: ["topic"]
      output_cols: "question"
      prompt_config_path: "query_template.yaml"

  # Call agent framework
  - block_type: "AgentBlock"
    block_config:
      block_name: "agent_call"
      input_cols: ["question"]
      output_cols: ["agent_response"]
      # agent_framework, agent_url set via flow.set_agent_config()
      extract_response: true

  # Extract structured data from agent response
  - block_type: "AgentResponseExtractorBlock"
    block_config:
      block_name: "extract"
      input_cols: "agent_response"
      output_cols: ["text_content", "tool_trace"]
      extract_tool_trace: true
```

---

## Pattern 6: MCP Tool-Use Distillation

Generate training data from MCP tool interactions.

```yaml
blocks:
  # Build task prompts
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_task"
      input_cols: ["task_description"]
      output_cols: "messages"
      prompt_config_path: "task_prompt.yaml"

  # LLM explores MCP tools in agentic loop
  - block_type: "MCPAgentBlock"
    block_config:
      block_name: "mcp_explore"
      input_cols: "messages"
      output_cols: "agent_trace"
      mcp_server_url: "http://localhost:3000/mcp"
      max_iterations: 10
```

---

## Testing Any Pattern

```python
# play.py
from sdg_hub import Flow
import pandas as pd

flow = Flow.from_yaml("flow.yaml")
flow.set_model_config(model="openai/gpt-4o-mini", api_key="sk-...")

df = pd.DataFrame({"document": ["Test document"]})

# Dry run -- check block-by-block execution
dry = flow.dry_run(df, sample_size=2)
print(f"Success: {dry['execution_successful']}")
for b in dry['blocks_executed']:
    print(f"  {b['block_name']}: {b['input_rows']} -> {b['output_rows']} rows, {b['execution_time_seconds']:.2f}s")

# Full run if successful
if dry['execution_successful']:
    result = flow.generate(df)
    print("Output columns:", list(result.columns))
    print(result.head())
```
