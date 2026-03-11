# RAG Evaluation ICL Dataset Flow

Generates realistic Q&A pairs for RAG (Retrieval-Augmented Generation) evaluation using the existing 3-stage question generation pipeline with ICL-driven question generation.

## What It Does

Uses the same 3-stage pipeline as the base RAG evaluation flow, but adds ICL examples to the question generation step so questions are generated in a realistic user style from the start:

1. Extracts a topic from the document (diversity control)
2. Generates a realistic question about that topic using ICL examples as style references (has full document context)
3. Evolves the question to be more indirect and compressed
4. Produces extractive answers grounded in the document context
5. Evaluates answer groundedness on a 1-5 scale
6. Filters out poorly grounded Q&A pairs (keeps only scores 4-5)
7. Extracts ground truth context sentences from the document

## Pipeline

```
Document → Topic Extraction → ICL Question Generation → Evolution →
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

The `icl_*` columns provide style guidance for the question generation step. The `icl_document` is a separate document with `icl_query_1/2/3` being example questions that were asked about it. The LLM studies the style, tone, and structure of these examples, then generates a realistic question about the extracted topic.

## Output Columns

| Column | Description |
|--------|-------------|
| `question` | Generated realistic question |
| `response` | Extractive answer grounded in the document |
| `ground_truth_context` | Exact sentences from the document that answer the question |

## Key Parameters

```python
runtime_params = {
    "gen_topic": {
        "max_tokens": 2048,
        "temperature": 0.7
    },
    "gen_conceptual_question": {
        "max_tokens": 2048,
        "temperature": 0.7
    },
    "evolve_question": {
        "max_tokens": 4096,
        "temperature": 0.7
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
- Want topic-focused diversity control from the 3-stage pipeline

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
  "question": "I'm trying to set up a distributed application where I need each individual Pod to be directly discoverable by its peers - when would I specifically choose a Headless Service, and how does it change DNS resolution compared to a standard Service?",
  "response": "You would choose a Headless Service when each Pod needs to be individually addressable. A Headless Service is created by setting clusterIP to None, and instead of allocating a cluster IP, it returns the Pod IPs directly through DNS...",
  "ground_truth_context": "Headless Services are created by setting clusterIP to None. They don't allocate a cluster IP and instead return the Pod IPs directly through DNS. This is useful for StatefulSets where each Pod needs to be individually addressable."
}
```

## Comparison with Base RAG Evaluation Flow

| Aspect | `rag_evaluation` | `rag_evaluation_icl` |
|--------|------------------|----------------------|
| Question style | Textbook-like, indirect | Realistic, user-like |
| ICL examples required | No | Yes |
| Questions per document | 1 | 1 |
| Question generation | 3 stages (topic, conceptual, evolution) | 3 stages (topic, ICL conceptual, evolution) |
| Answer generation | Identical | Identical |
| Groundedness scoring | Identical (1-5 scale) | Identical (1-5 scale) |
| Output columns | Same | Same |
