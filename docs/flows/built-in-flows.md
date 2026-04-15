# Built-in Flows

SDG Hub ships with built-in flows in `src/sdg_hub/flows/`. They are
automatically discovered by `FlowRegistry` and can be loaded by id or name.

```python
from sdg_hub import Flow
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()
flow_path = FlowRegistry.get_flow_path_safe("epic-jade-656")
flow = Flow.from_yaml(flow_path)
```

---

## Quick Reference

| Category | Flow Name | ID | Default Model | Required Columns |
|----------|-----------|------|---------------|------------------|
| knowledge_infusion | Extractive Summary Knowledge Tuning | `epic-jade-656` | openai/gpt-oss-120b | document, document_outline, domain, icl_document, icl_query_1-3 |
| knowledge_infusion | Detailed Summary Knowledge Tuning | `mild-thunder-748` | openai/gpt-oss-120b | document, document_outline, domain, icl_document, icl_query_1-3 |
| knowledge_infusion | Key Facts Knowledge Tuning | `heavy-heart-77` | openai/gpt-oss-120b | document, document_outline, domain |
| knowledge_infusion | Document Based Knowledge Tuning | `stellar-peak-605` | openai/gpt-oss-120b | document, document_outline, domain, icl_document, icl_query_1-3 |
| knowledge_infusion | Japanese Multi-Summary QA | `clean-shadow-397` | microsoft/phi-4 | document, document_outline, domain, icl_document, icl_query_1-3, icl_response_1-3 |
| knowledge_infusion | Enhanced Multi-Summary QA (Spanish) | 4 flows (es variants) | openai/gpt-oss-120b | same as English variants |
| text_analysis | Structured Text Insights Extraction | `green-clay-812` | openai/gpt-oss-120b | text |
| red_team | Red Teaming Prompt Generation | `major-sage-742` | IlyaGusev/gemma-2-9b-it-abliterated | policy_concept, concept_definition |
| agentic | MCP Server Distillation | `new-night-835` | openai/gpt-5.2 | tool_list, mcp_server_name, mcp_server_description |
| code_evaluation | Domain Code Evaluation Benchmark Generator | `domain-code-eval` | gpt-5.1-codex-mini | domain, function_spec, difficulty, time_complexity |
| evaluation | RAG Evaluation Dataset | `loud-dawn-245` | openai/gpt-oss-120b | document, document_outline |
| evaluation | Agent Tool-Use Evaluation | `eager-path-837` | openai/gpt-4o | question, expert_answer_truncated, expert_trace_formatted, model_answer, model_trace_formatted |

---

## Knowledge Infusion Flows

### Enhanced Multi-Summary QA (4 variants)

Location: `src/sdg_hub/flows/knowledge_infusion/enhanced_multi_summary_qa/`

These four flows share a common pipeline pattern and produce knowledge tuning
datasets. Each variant creates a different type of document augmentation before
generating QA pairs.

Common input columns: `document`, `document_outline`, `domain`,
`icl_document`, `icl_query_1`, `icl_query_2`, `icl_query_3` (except Key Facts,
which does not require `icl_*` columns).

Common output columns: `question`, `response`, `document`,
`faithfulness_explanation`, `faithfulness_judgment` (plus variant-specific
columns).

#### Extractive Summary (epic-jade-656)

Pipeline: Document --> Extractive Summary (n=50) --> Question List --> Answers --> Faithfulness Check --> Filtered QA

Extracts 2-4 key passages per document section and annotates each with context
markers, relevance ratings, and relationships to other extracts. Output includes
`raw_document`.

#### Detailed Summary (mild-thunder-748)

Pipeline: Document --> Detailed Summary (n=50) --> Question List --> Answers --> Faithfulness Check --> Filtered QA

Generates high-level summaries focusing on overarching themes, main arguments,
and core principles. Abstractive rather than extractive. Output includes
`raw_document`.

#### Key Facts (heavy-heart-77)

Pipeline: Document --> Atomic Facts Extraction --> Fact Parsing --> Multi-QA Generation (5 per fact)

Breaks documents into atomic facts and generates 5 QA pairs per fact. Does not
require `icl_*` columns (only `document`, `document_outline`, `domain`). No
faithfulness evaluation. Output columns: `key_fact`, `question`, `response`,
`raw_key_fact_qa`.

#### Document Based (stellar-peak-605)

Pipeline: Document --> Question List --> Answers --> Faithfulness Check --> Filtered QA

Directly generates QA pairs from the raw document without intermediate
summarization. Fastest of the four variants.

#### Usage Example

```python
from datasets import Dataset
from sdg_hub import Flow
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()

# Load any of the four variants by id
flow = Flow.from_yaml(
    FlowRegistry.get_flow_path_safe("epic-jade-656")  # extractive summary
)

flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="your-key",
)

dataset = Dataset.from_dict({
    "document": ["Your document content..."],
    "document_outline": ["Document Title"],
    "domain": ["articles/essays"],
    "icl_document": ["Example document..."],
    "icl_query_1": ["Example question 1?"],
    "icl_query_2": ["Example question 2?"],
    "icl_query_3": ["Example question 3?"],
})

result = flow.generate(dataset, max_concurrency=50)
```

---

### Japanese Multi-Summary QA (clean-shadow-397)

Location: `src/sdg_hub/flows/knowledge_infusion/japanese_multi_summary_qa/`

A comprehensive Japanese-language flow that generates three summary types
(detailed, extractive, atomic facts), melts them into a unified dataset, then
generates QA pairs with three-stage evaluation: faithfulness, relevancy, and
question verification.

Default model: `microsoft/phi-4`

Required columns: `document`, `document_outline`, `domain`, `icl_document`,
`icl_query_1-3`, `icl_response_1-3`

Tags: `question-generation`, `knowledge-extraction`, `qa-pairs`,
`multilingual`, `japanese`

```python
from sdg_hub import Flow
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()
flow = Flow.from_yaml(
    FlowRegistry.get_flow_path_safe("clean-shadow-397")
)
flow.set_model_config(
    model="microsoft/phi-4",
    api_key="your-key",
)

result = flow.generate(japanese_dataset, max_concurrency=10)
```

---

### Enhanced Multi-Summary QA (Spanish)

Location: `src/sdg_hub/flows/knowledge_infusion/enhanced_multi_summary_qa_es/`

Spanish-language translations of the four Enhanced Multi-Summary QA flows
(detailed summary, extractive summary, key facts, doc direct QA). Same pipeline
structure as the English originals with translated prompt templates. Generated
using the `translate_flow()` utility.

---

## Text Analysis Flows

### Structured Text Insights Extraction (green-clay-812)

Location: `src/sdg_hub/flows/text_analysis/structured_insights/`

Multi-step pipeline that extracts four types of analysis from text: summary,
keywords, entities, and sentiment. All extractions use async LLM calls with low
temperature for deterministic output. Results are combined into a single
`structured_insights` JSON column using `JSONStructureBlock`.

Default model: `openai/gpt-oss-120b`

Required columns: `text` (minimum 50 words recommended)

Output columns: `summary`, `keywords`, `entities`, `sentiment`,
`structured_insights`

Tags: `text-analysis`, `summarization`, `nlp`, `structured-output`,
`sentiment-analysis`, `entity-extraction`, `keyword-extraction`

```python
from datasets import Dataset
from sdg_hub import Flow
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()
flow = Flow.from_yaml(
    FlowRegistry.get_flow_path_safe("green-clay-812")
)
flow.set_model_config(
    model="openai/gpt-oss-120b",
    api_key="your-key",
)

articles = Dataset.from_dict({
    "text": [
        "Your article content with at least 50 words...",
    ],
})
result = flow.generate(articles, max_concurrency=20)
```

---

## Red Team Flows

### Red Teaming Prompt Generation (major-sage-742)

Location: `src/sdg_hub/flows/red_team/prompt_generation/`

Generates adversarial prompts for safety testing by combining policy concepts
with multi-dimensional sampling across demographics, expertise, geography,
language style, exploit stage, medium, temporal context, and trust signals. Uses
`RowMultiplierBlock` to replicate rows and `SamplerBlock` for each dimension.
Output is structured JSON parsed by `JSONParserBlock`.

Default model: `IlyaGusev/gemma-2-9b-it-abliterated`

Required columns: `policy_concept`, `concept_definition`

Optional columns: `demographics_pool`, `expertise_pool`, `geography_pool`,
`language_styles_pool`, `exploit_stages_pool`, `task_medium_pool`,
`temporal_pool`, `trust_signals_pool`

Output columns: `prompt`, `why_prompt_harmful`,
`why_prompt_has_temporal_relevance`, `why_prompt_fits_exploit_stage`,
and additional `why_prompt_*` rationale fields.

Tags: `red-team`, `adversarial`, `prompt-generation`, `safety-testing`,
`security`

```python
from datasets import Dataset
from sdg_hub import Flow
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()
flow = Flow.from_yaml(
    FlowRegistry.get_flow_path_safe("major-sage-742")
)
flow.set_model_config(
    model="IlyaGusev/gemma-2-9b-it-abliterated",
    api_key="your-key",
)

dataset = Dataset.from_dict({
    "policy_concept": ["hate speech"],
    "concept_definition": ["Content that promotes hatred against groups..."],
})
result = flow.generate(dataset)
```

---

## Agentic Flows

### MCP Server Distillation (new-night-835)

Location: `src/sdg_hub/flows/agentic/mcp_distillation/`

Generates tool-use training data through expert distillation. The flow explores
an MCP server to understand tool behavior, generates grounded multi-tool
questions, runs expert trajectories through an agent connector, and filters for
strong completions. This flow uses both LLM and agent blocks, so both
`set_model_config()` and `set_agent_config()` must be called before
`generate()`.

Default model: `openai/gpt-5.2`

Required columns: `tool_list`, `mcp_server_name`, `mcp_server_description`

Tags: `agentic`, `tool-use`, `data-generation`, `mcp`, `distillation`,
`exploration`

```python
from datasets import Dataset
from sdg_hub import Flow
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()
flow = Flow.from_yaml(
    FlowRegistry.get_flow_path_safe("new-night-835")
)

# Configure both LLM and agent blocks
flow.set_model_config(
    model="openai/gpt-5.2",
    api_key="your-llm-key",
)
flow.set_agent_config(
    agent_framework="langflow",
    agent_url="http://localhost:7860/api/v1/run/default",
    agent_api_key="your-langflow-key",
)

seed = Dataset.from_dict({
    "mcp_server_name": ["ecommerce_analytics"],
    "mcp_server_description": ["Analytics server for products and orders"],
    "tool_list": [[
        {
            "name": "search_products",
            "description": "Search products by keyword",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]],
})

result = flow.generate(seed, max_concurrency=10)
```

---

## Code Evaluation Flows

### Domain Code Evaluation Benchmark Generator (domain-code-eval)

Location: `src/sdg_hub/flows/code_evaluation/domain_code_eval/`

Generates execution-verified coding benchmarks for custom domains. Inspired by
[AutoCodeBench](https://arxiv.org/abs/2503.08013) (Tencent, 2025) and
[CRUXEval](https://arxiv.org/abs/2401.03065) (Meta, 2024), this flow generates
domain-specific Python functions, verifies them via sandboxed execution, and
reverse-generates problem descriptions. The output is a ready-to-use evaluation
benchmark where every solution and test suite is execution-verified.

!!! note "Optional dependency"
    This flow uses `PythonInterpreterBlock`, which requires the `code` optional
    dependency group: `uv pip install sdg-hub[code]` or `pip install sdg-hub[code]`

Default model: `gpt-5.1-codex-mini`

Compatible models: `openai/gpt-5.1-codex-mini`,
`meta-llama/Llama-3.3-70B-Instruct`, `Qwen/Qwen2.5-Coder-32B-Instruct`

Required columns: `domain`, `function_spec`, `difficulty`, `time_complexity`

Output columns: `function_code`, `test_code`, `input_generator`,
`problem_description`, `execution_result`, `execution_result_success`

Tags: `code-evaluation`, `benchmark-generation`, `code-verification`,
`domain-specific`, `synthetic-data`

#### 4-Phase Pipeline

```
Phase 1: Generate Function
  PromptBuilderBlock → LLMChatBlock → LLMResponseExtractorBlock → TagParserBlock
  Input: domain, function_spec, difficulty, time_complexity
  Output: function_code

Phase 2: Generate Test Suite
  PromptBuilderBlock → LLMChatBlock → LLMResponseExtractorBlock → TagParserBlock × 2
  Input: function_code, domain
  Output: test_code, input_generator

Phase 3: Verify via Execution
  TextConcatBlock → PythonInterpreterBlock → ColumnValueFilterBlock
  Input: function_code + test_code → executable_code
  Output: execution_result (rows where execution failed are filtered out)

Phase 4: Reverse Problem Generation
  PromptBuilderBlock → LLMChatBlock → LLMResponseExtractorBlock → TagParserBlock
  Input: function_code, test_code, domain, time_complexity
  Output: problem_description
```

**Phase 1** generates a domain-specific Python function based on the spec and
constraints. The prompt enforces sandbox-safe code: no imports, no I/O, no
randomness, JSON-serializable return values.

**Phase 2** generates a test suite (`test()` function with assertions) and an
input generator (`generate_input(n)` function) for the generated function.

**Phase 3** combines the function and tests into a single executable script,
runs it in the Monty sandbox, and filters out rows where execution failed.
This ensures every benchmark entry in the output is verified.

**Phase 4** reverse-generates a problem description from the verified function
and tests, producing a natural-language problem statement that does not reveal
the implementation.

#### Usage Example

```python
from datasets import Dataset
from sdg_hub import Flow
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()
flow = Flow.from_yaml(
    FlowRegistry.get_flow_path_safe("domain-code-eval")
)

flow.set_model_config(
    model="openai/gpt-5.1-codex-mini",
    api_key="your-key",
)

dataset = Dataset.from_dict({
    "domain": ["financial calculations"],
    "function_spec": ["Calculate compound interest given principal, rate, and time"],
    "difficulty": ["intermediate"],
    "time_complexity": ["O(1)"],
})

result = flow.generate(dataset, max_concurrency=5)
# result contains only rows where execution verification passed
```

#### Input Column Requirements

| Column | Description | Example |
|--------|-------------|---------|
| `domain` | Problem domain | `"financial calculations"` |
| `function_spec` | What the function should do | `"Calculate compound interest"` |
| `difficulty` | Difficulty level | `"beginner"`, `"intermediate"`, `"advanced"` |
| `time_complexity` | Expected time complexity | `"O(1)"`, `"O(n)"`, `"O(n log n)"` |

#### Example Notebooks

Two example notebooks are available under `examples/code_interpreter/`:

- `domain_code_eval.ipynb` -- end-to-end benchmark generation
- `model_evaluation.ipynb` -- LeetCode-style model evaluation with timing
  verification

---

## Evaluation Flows

### RAG Evaluation Dataset (loud-dawn-245)

Location: `src/sdg_hub/flows/evaluation/rag_evaluation/`

Generates QA pairs for RAG evaluation. The pipeline extracts topics from
documents, generates conceptual questions, evolves them for complexity, produces
answers, scores groundedness with a critic, filters low-scoring pairs, and
extracts supporting context passages.

Default model: `openai/gpt-oss-120b`

Required columns: `document`, `document_outline`

Tags: `rag-evaluation`, `qa-pairs`

```python
from datasets import Dataset
from sdg_hub import Flow
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()
flow = Flow.from_yaml(
    FlowRegistry.get_flow_path_safe("loud-dawn-245")
)
flow.set_model_config(
    model="openai/gpt-oss-120b",
    api_key="your-key",
)

dataset = Dataset.from_dict({
    "document": ["Your document text..."],
    "document_outline": ["Document Title"],
})
result = flow.generate(dataset, max_concurrency=20)
```

### Agent Tool-Use Evaluation (eager-path-837)

Location: `src/sdg_hub/flows/evaluation/mcp_eval_benchmark/model_evaluation/`

Scores an agent's tool-use traces against expert gold-standard traces using an
LLM-as-judge. Evaluates 6 dimensions: task fulfillment, grounding, tool
appropriateness, parameter accuracy, dependency awareness, and
parallelism/efficiency. Works with any agent framework that produces tool call
traces.

Default model: `openai/gpt-4o`

Required columns: `question`, `expert_answer_truncated`,
`expert_trace_formatted`, `model_answer`, `model_trace_formatted`

Output columns: `task_fulfillment`, `grounding`, `tool_appropriateness`,
`parameter_accuracy`, `dependency_awareness`, `parallelism_and_efficiency`,
`rationale`

Tags: `evaluation`, `mcp`, `benchmark`, `model-evaluation`, `llm-as-judge`

```python
from sdg_hub import Flow
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()
flow = Flow.from_yaml(
    FlowRegistry.get_flow_path_safe("eager-path-837")
)
flow.set_model_config(
    model="openai/gpt-4o",
    api_key="your-key",
)

result = flow.generate(evaluation_dataset, max_concurrency=10)
```
