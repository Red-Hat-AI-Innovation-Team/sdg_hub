# RAG Evaluation ICL Persona Dataset Flow

Generates realistic Q&A pairs for RAG (Retrieval-Augmented Generation) evaluation using the existing 3-stage question generation pipeline with ICL-driven question generation and persona-aware answer generation.

## What It Does

Combines two capabilities: realistic question generation (from ICL examples) and persona-aware answer generation (from a configurable system prompt):

1. Extracts a topic from the document (diversity control)
2. Generates a realistic question about that topic using ICL examples as style references (has full document context)
3. Evolves the question to be more indirect and compressed
4. Produces answers grounded in the document context, following the chatbot's persona (tone, formatting, response structure)
5. Evaluates answer groundedness on a 1-5 scale
6. Filters out poorly grounded Q&A pairs (keeps only scores 4-5)
7. Extracts ground truth context sentences from the document

## Pipeline

```
Document → Topic Extraction → ICL Question Generation → Evolution →
Persona-Aware Answer Generation → Groundedness Scoring → Filter (4-5) → Context Extraction → Final QA Pairs
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
| `system_prompt` | Chatbot persona defining tone, formatting, and response structure | Yes |

The `system_prompt` column defines how the chatbot should respond. The answer generation step uses this as the system message alongside grounding rules, producing answers that match the chatbot's persona while staying grounded in the document context.

## Output Columns

| Column | Description |
|--------|-------------|
| `question` | Generated realistic question |
| `response` | Persona-aware answer grounded in the document |
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

- Evaluating a specific chatbot with questions and answers that match its persona
- Need evaluation datasets where answers reflect how the actual chatbot would respond
- Have a chatbot system prompt and want to generate ground truth in that style
- Want realistic user-style questions combined with persona-aware answers

For generic extractive answers without persona, use `rag_evaluation_icl` instead. For textbook-style questions, use the base `rag_evaluation` flow.

## Example Usage

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

# Load flow
FlowRegistry.discover_flows()
flow_path = FlowRegistry.get_flow_path("RAG Evaluation ICL Persona Dataset Flow")
flow = Flow.from_yaml(flow_path)

# Configure model
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="your_key"
)

# Define your chatbot's persona
persona = """You are a Senior Platform Engineer. Your mission: help engineers
fix CI/CD issues quickly and accurately. Be conversational and direct.
Explain WHY before WHAT. Use plain text formatting, no markdown."""

# Prepare input data
dataset = Dataset.from_dict({
    "document": ["Your target document content..."],
    "document_outline": ["Document Title; Section 1; Section 2"],
    "icl_document": ["Example document that the example questions are about..."],
    "icl_query_1": ["I'm trying to configure X but getting timeout errors - is there a max retry setting?"],
    "icl_query_2": ["We set up a pipeline with custom tasks and the labels seem to get reused - is that expected?"],
    "icl_query_3": ["How do I debug failed builds when the logs only show the last step?"],
    "system_prompt": [persona]
})

# Generate
result = flow.generate(dataset, max_concurrency=50)
print(f"Generated {len(result)} QA pairs")
```

## Example Output

```json
{
  "question": "I'm trying to access individual pods within a StatefulSet directly, bypassing the main service - which Service type is typically used for this, and how does it work?",
  "response": "Good question - this is a common pattern for stateful apps. The reason you need a Headless Service here is that it bypasses the normal load balancing. When you set clusterIP to None, DNS returns the individual Pod IPs directly instead of a single cluster IP...",
  "ground_truth_context": "Headless Services are created by setting clusterIP to None. They don't allocate a cluster IP and instead return the Pod IPs directly through DNS. This is useful for StatefulSets where each Pod needs to be individually addressable."
}
```

## Comparison with Other RAG Evaluation Flows

| Aspect | `rag_evaluation` | `rag_evaluation_icl` | `rag_evaluation_icl_persona` |
|--------|------------------|----------------------|------------------------------|
| Question style | Textbook-like | Realistic, user-like | Realistic, user-like |
| Answer style | Generic extractive | Generic extractive | Persona-aware |
| ICL examples | No | Yes | Yes |
| System prompt | No | No | Yes |
| Questions per document | 1 | 1 | 1 |
| Question generation | 3 stages | 3 stages (ICL) | 3 stages (ICL) |
| Groundedness scoring | 1-5 scale | 1-5 scale | 1-5 scale |
| Output columns | Same | Same | Same |
