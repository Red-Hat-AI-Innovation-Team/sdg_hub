# RAG Evaluation ICL Dataset Flow

Generates realistic Q&A pairs for RAG (Retrieval-Augmented Generation) evaluation using In-Context Learning (ICL) with real user question examples.

## What It Does

Uses example questions from real users to guide question generation style, producing realistic, user-like questions instead of textbook-style ones:

1. Takes example questions alongside the document they relate to as style references
2. Generates a series of realistic questions per document that match the style and tone of the examples
3. Produces extractive answers grounded in the document context
4. Evaluates answer groundedness on a 1-5 scale
5. Filters out poorly grounded Q&A pairs (keeps only scores 4-5)
6. Extracts ground truth context sentences from the document

## Pipeline

```
Document + ICL Examples → ICL Question Generation → Tag Parsing (row expansion) →
Answer Generation → Groundedness Scoring → Filter (4-5) → Context Extraction → Final QA Pairs
```

## Input Requirements

| Column | Description | Required |
|--------|-------------|----------|
| `document` | Full document text to generate questions about | Yes |
| `document_outline` | Document title or structural outline | Yes |
| `icl_document` | Example document used as style reference | Yes |
| `icl_query_1` | First example question (real user style) | Yes |
| `icl_query_2` | Second example question (real user style) | Yes |
| `icl_query_3` | Third example question (real user style) | Yes |

The `icl_*` columns provide style guidance. The `icl_document` is a separate document with `icl_query_1/2/3` being example questions that were asked about it. The LLM studies the style, tone, and structure of these examples, then generates similar-style questions for the target `document`.

## Output Columns

| Column | Description |
|--------|-------------|
| `question` | Generated realistic question |
| `response` | Extractive answer grounded in the document |
| `ground_truth_context` | Exact sentences from the document that answer the question |

## Key Parameters

```python
runtime_params = {
    "gen_icl_questions": {
        "max_tokens": 256,
        "temperature": 0.7    # Higher for question diversity
    },
    "gen_answer": {
        "max_tokens": 4096,
        "temperature": 0.2    # Lower for factual answers
    },
    "gen_critic_score": {
        "max_tokens": 512,
        "temperature": 0.0    # Deterministic scoring
    }
}
```

## When to Use

- Evaluating RAG systems with realistic user-style questions
- Need questions that reflect how real users ask (first-person, scenario-based, troubleshooting)
- Have example questions from real users to use as style references
- Want multiple questions per document with groundedness filtering

For textbook-style questions without ICL examples, use the base `rag_evaluation` flow instead.

## Example Usage

```python
from datasets import Dataset
from sdg_hub.core.flow import Flow, FlowRegistry

# Load flow
FlowRegistry.discover_flows()
flow_path = FlowRegistry.get_flow_path("RAG Evaluation ICL Dataset Flow")
flow = Flow.from_yaml(flow_path)

# Configure model
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="your_key"
)

# Prepare input data with ICL examples
dataset = Dataset.from_dict({
    "document": ["Your target document content..."],
    "document_outline": ["Document Title; Section 1; Section 2"],
    "icl_document": ["Example document that the example questions are about..."],
    "icl_query_1": ["I'm trying to configure X but getting timeout errors - is there a max retry setting?"],
    "icl_query_2": ["We set up a pipeline with custom tasks and the labels seem to get reused - is that expected?"],
    "icl_query_3": ["How do I debug failed builds when the logs only show the last step?"]
})

# Generate
result = flow.generate(dataset, max_concurrency=50)
print(f"Generated {len(result)} QA pairs")
```

## Example Output

```json
{
  "question": "I configured the webhook trigger but it doesn't fire on push events to feature branches - do I need to set a specific branch filter pattern?",
  "response": "According to the documentation, webhook triggers require an explicit branch filter configuration...",
  "ground_truth_context": "Webhook triggers support glob patterns for branch filtering. By default, only the main branch is matched unless a custom pattern is specified."
}
```

## Comparison with Base RAG Evaluation Flow

| Aspect | `rag_evaluation` | `rag_evaluation_icl` |
|--------|------------------|----------------------|
| Question style | Textbook-like, indirect | Realistic, user-like |
| ICL examples required | No | Yes |
| Questions per document | 1 | Multiple |
| Question generation | 3 stages (topic, conceptual, evolution) | 1 stage (ICL-driven) |
| Answer generation | Identical | Identical |
| Groundedness scoring | Identical (1-5 scale) | Identical (1-5 scale) |
| Output columns | Same | Same |
