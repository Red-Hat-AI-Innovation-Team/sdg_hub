# Flow Catalog

Comprehensive guide to all available flows in SDG Hub for synthetic data generation.

## Overview

SDG Hub provides a rich ecosystem of pre-built flows for various data generation and analysis tasks. Each flow is a carefully orchestrated pipeline of blocks designed to solve specific problems:

- **QA Generation Flows** - Create training datasets with question-answer pairs for knowledge tuning
- **Text Analysis Flows** - Extract structured insights from unstructured text content
- **Multilingual Flows** - Localized variants for non-English data generation

All flows support:
- Automatic discovery and registration
- Runtime model configuration
- Async processing for efficiency
- Quality evaluation and filtering
- Checkpointing and concurrency control

## Quick Reference

| Flow Category | Flow Count | Primary Use Case | Tags |
|---------------|------------|------------------|------|
| [Enhanced Multi-Summary QA](#enhanced-multi-summary-qa-flows) | 4 | Knowledge tuning dataset generation | `knowledge-tuning`, `document-internalization` |
| [InstructLab QA](#instructlab-multi-summary-qa-flow) | 1 | High-quality QA with extensive evaluation | `question-generation`, `educational` |
| [Multilingual QA](#japanese-multilingual-multi-summary-qa-flow) | 1 | Japanese language QA generation | `multilingual`, `japanese` |
| [Text Analysis](#structured-text-insights-extraction-flow) | 1 | NLP insights extraction | `text-analysis`, `nlp` |

## Flow Discovery

All flows are automatically discovered from `src/sdg_hub/flows/`:

```python
from sdg_hub.core.flow import FlowRegistry, Flow

# Auto-discover all available flows
FlowRegistry.discover_flows()

# List all flows
all_flows = FlowRegistry.list_flows()
print(f"Found {len(all_flows)} flows")

# Search by tag
qa_flows = FlowRegistry.search_flows(tag="question-generation")
knowledge_flows = FlowRegistry.search_flows(tag="knowledge-tuning")
analysis_flows = FlowRegistry.search_flows(tag="text-analysis")

# Get flow information
flow_name = "Extractive Summary Knowledge Tuning Dataset Generation Flow"
metadata = FlowRegistry.get_flow_metadata(flow_name)
print(f"Flow: {metadata.name}")
print(f"Version: {metadata.version}")
print(f"Tags: {', '.join(metadata.tags)}")

# Load and use a flow
flow_path = FlowRegistry.get_flow_path(flow_name)
flow = Flow.from_yaml(flow_path)
```

---

## Enhanced Multi-Summary QA Flows

**Purpose:** Generate high-quality knowledge tuning datasets by creating multiple document augmentations and corresponding question-answer pairs.

**Architecture Pattern:**
```
Document → Summary/Extraction Generation → Question Generation → Answer Generation → Faithfulness Evaluation → Filtered QA Pairs
```

**Common Characteristics:**
- Designed for knowledge internalization and model training
- Include faithfulness evaluation to ensure answer quality
- Support high-volume generation with configurable `n` parameter
- Async processing for efficiency
- All tagged with `knowledge-tuning`, `document-internalization`, `question-generation`

**Location:** `src/sdg_hub/flows/qa_generation/document_grounded_qa/enhanced_multi_summary_qa/`

### 2.1 Extractive Summary Knowledge Tuning Flow

**Name:** `Extractive Summary Knowledge Tuning Dataset Generation Flow`

**What It Does:**

Creates enhanced extractive summaries with rich contextual annotations:
1. Extracts 2-4 key passages from each document section
2. Annotates each extract with:
   - **Context Marker**: Where it fits in the document narrative
   - **Relevance**: Importance rating (Low, Medium, High, Very High)
   - **Relationship**: Connections to other extracts
3. Generates questions from annotated summaries
4. Produces answers with faithfulness evaluation

**Pipeline:**
```yaml
Document → Extractive Summary (n=50) → Question List → Answers → Faithfulness Check → Filtered QA
```

**Input Requirements:**

| Column | Description | Required |
|--------|-------------|----------|
| `document` | Full document text | Yes |
| `document_outline` | Document title/outline | Yes |
| `domain` | Content domain (e.g., "articles/essays") | Yes |
| `icl_document` | In-context learning example document | Yes |
| `icl_query_1`, `icl_query_2`, `icl_query_3` | Example questions | Yes |

**Output Columns:**
- `summary` - The extractive summary with annotations
- `question` - Generated question
- `response` - Generated answer
- `raw_document` - Original document (preserved)
- `faithfulness_explanation` - Evaluation explanation
- `faithfulness_judgment` - "YES" or "NO"

**Key Parameters:**

```python
runtime_params = {
    "gen_extractive_summary": {
        "n": 50,              # Generate 50 summaries per document
        "max_tokens": 4096,
        "temperature": 0.7
    },
    "question_generation": {
        "max_tokens": 256,
        "temperature": 0.7,
        "n": 1
    },
    "answer_generation": {
        "max_tokens": 4096,
        "temperature": 0.7
    }
}
```

**When to Use:**
- Need detailed knowledge extraction with contextual understanding
- Want to teach models about information relationships
- Working with complex documents where context matters
- Prefer quality summaries with semantic annotations

**Example Usage:**

```python
from datasets import Dataset
from sdg_hub.core.flow import Flow, FlowRegistry
import os

# Discover and load flow
FlowRegistry.discover_flows()
flow_path = FlowRegistry.get_flow_path(
    "Extractive Summary Knowledge Tuning Dataset Generation Flow"
)
flow = Flow.from_yaml(flow_path)

# Configure model
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="your_key"
)

# Prepare input data
dataset = Dataset.from_dict({
    "document": ["Your document content..."],
    "document_outline": ["Document Title"],
    "domain": ["articles/essays"],
    "icl_document": ["Example document..."],
    "icl_query_1": ["Example question 1?"],
    "icl_query_2": ["Example question 2?"],
    "icl_query_3": ["Example question 3?"]
})

# Generate with custom parameters
result = flow.generate(
    dataset,
    runtime_params={
        "gen_extractive_summary": {"n": 30},  # Generate 30 summaries
    },
    max_concurrency=50
)

# Save output
result.to_json("extractive_summary/gen.jsonl", orient="records", lines=True)
print(f"Generated {len(result)} QA pairs")
```

**Example Output:**

```json
{
  "summary": "### Extract 1\n> \"Remote work has grown by over 150% since 2020.\"\n\n**Context Marker**: Opening factual statement providing temporal context\n**Relevance**: Very High – Quantifies the transformation scale\n**Relationship**: Establishes cause for changes in Extracts 2 and 3",
  "question": "How has remote work adoption changed since the pandemic?",
  "response": "Remote work has grown by over 150% since 2020 due to the pandemic...",
  "faithfulness_judgment": "YES"
}
```

---

### 2.2 Detailed Summary Knowledge Tuning Flow

**Name:** `Detailed Summary Knowledge Tuning Dataset Generation Flow`

**What It Does:**

Generates high-level summaries focusing on overarching themes and core principles:
1. Creates comprehensive summaries emphasizing main arguments
2. Focuses on "big picture" understanding rather than specific details
3. Generates thoughtful questions about themes and principles
4. Produces answers that demonstrate conceptual understanding

**Pipeline:**
```yaml
Document → Detailed Summary (n=50) → Question List → Answers → Faithfulness Check → Filtered QA
```

**Input Requirements:**

Same as Extractive Summary Flow (see above).

**Output Columns:**

Same structure as Extractive Summary Flow:
- `summary`, `question`, `response`, `raw_document`, `faithfulness_explanation`, `faithfulness_judgment`

**Key Parameters:**

```python
runtime_params = {
    "gen_detailed_summary": {
        "n": 50,              # Generate 50 summaries per document
        "max_tokens": 4096,
        "temperature": 0.7
    },
    "question_generation": {
        "max_tokens": 256,
        "temperature": 0.7
    }
}
```

**When to Use:**
- Teaching models about overarching themes and arguments
- Need conceptual understanding over factual details
- Working with analytical or argumentative content
- Want summaries that capture author's main points

**Differences from Extractive:**
- Abstractive vs extractive summarization
- Focuses on themes vs specific passages
- Better for concept learning vs fact learning
- More interpretive, less literal

**Example Usage:**

```python
# Load flow
flow_path = FlowRegistry.get_flow_path(
    "Detailed Summary Knowledge Tuning Dataset Generation Flow"
)
flow = Flow.from_yaml(flow_path)

# Configure and generate
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1"
)

result = flow.generate(
    dataset,
    runtime_params={
        "gen_detailed_summary": {"n": 25}
    },
    max_concurrency=50
)
```

**Example Output:**

```json
{
  "summary": "The document explores the transformation of work practices during the pandemic, examining both the benefits and challenges of remote work adoption. It argues that hybrid models represent an optimal balance between flexibility and collaboration.",
  "question": "What central argument does the document make about the future of work?",
  "response": "The document argues that hybrid models represent the optimal balance...",
  "faithfulness_judgment": "YES"
}
```

---

### 2.3 Key Facts Knowledge Tuning Flow

**Name:** `Key Facts Knowledge Tuning Dataset Generation Flow`

**What It Does:**

Extracts atomic facts and generates multiple QA pairs for each:
1. Breaks document into discrete, atomic facts
2. Lists key facts with contextual information
3. Generates **5 QA pairs per atomic fact** (highest volume output)
4. No faithfulness evaluation (assumes fact-based answers are faithful)

**Pipeline:**
```yaml
Document → Atomic Facts Extraction → Individual Fact Parsing → Multi-QA Generation (5 per fact)
```

**Input Requirements:**

| Column | Description | Required |
|--------|-------------|----------|
| `document` | Full document text | Yes |
| `document_outline` | Document title/outline | Yes |
| `domain` | Content domain | Yes |

Note: Does NOT require `icl_*` fields (simpler input)

**Output Columns:**
- `key_fact` - The extracted atomic fact
- `question` - Generated question
- `response` - Generated answer
- `raw_key_fact_qa` - Raw model output

**Key Parameters:**

```python
runtime_params = {
    "gen_atomic_facts": {
        "max_tokens": 4096,
        "temperature": 0.7,
        "n": 1  # One atomic facts list per document
    },
    "generate_key_fact_qa": {
        "max_tokens": 4096,
        "temperature": 0.7,
        "n": 1  # Generates 5 QA pairs internally
    }
}
```

**When to Use:**
- Need maximum QA pair volume (5 per fact × many facts)
- Working with fact-dense documents (scientific, technical)
- Training models on factual recall
- Want fast generation without evaluation overhead

**Output Volume:**
If a document yields 20 atomic facts, you get **100 QA pairs** (20 × 5)

**Example Usage:**

```python
# Load flow
flow_path = FlowRegistry.get_flow_path(
    "Key Facts Knowledge Tuning Dataset Generation Flow"
)
flow = Flow.from_yaml(flow_path)

flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1"
)

# Simpler input (no icl_* fields needed)
dataset = Dataset.from_dict({
    "document": ["Your document content..."],
    "document_outline": ["Document Title"],
    "domain": ["scientific"]
})

result = flow.generate(dataset, max_concurrency=50)
print(f"Generated {len(result)} QA pairs")
```

**Example Output:**

```json
{
  "key_fact": "Remote work adoption increased by 150% between 2020 and 2023.",
  "question": "By what percentage did remote work adoption increase during the pandemic?",
  "response": "Remote work adoption increased by 150% between 2020 and 2023."
}
```

---

### 2.4 Document-Based QA Flow

**Name:** `Document Based Knowledge Tuning Dataset Generation Flow`

**What It Does:**

Directly generates QA pairs from raw documents without intermediate summarization:
1. Takes original document as-is
2. Generates questions directly from full content
3. Produces comprehensive answers
4. Includes faithfulness evaluation

**Pipeline:**
```yaml
Document → Question List → Answers → Faithfulness Check → Filtered QA
```

**Input Requirements:**

Same as Extractive/Detailed flows (includes `icl_*` fields).

**Output Columns:**
- `question` - Generated question
- `response` - Generated answer
- `raw_document` - Original document (preserved)
- `faithfulness_explanation` - Evaluation explanation
- `faithfulness_judgment` - "YES" or "NO"

Note: No `summary` column (direct from document)

**Key Parameters:**

```python
runtime_params = {
    "question_generation": {
        "max_tokens": 256,
        "temperature": 1.0,  # Higher temperature for diversity
        "n": 1
    },
    "answer_generation": {
        "max_tokens": 4096,
        "temperature": 1.0
    }
}
```

**When to Use:**
- Need quick QA generation without augmentation overhead
- Document content is already well-structured
- Want QAs grounded in full original text
- Don't need multiple augmentation variants

**Performance:**
- Fastest of the 4 flows (no summary generation)
- Lower output volume (no n parameter for summaries)
- Still includes quality filtering

**Example Usage:**

```python
# Load flow
flow_path = FlowRegistry.get_flow_path(
    "Document Based Knowledge Tuning Dataset Generation Flow"
)
flow = Flow.from_yaml(flow_path)

flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1"
)

result = flow.generate(dataset, max_concurrency=50)
```

**Example Output:**

```json
{
  "question": "What are the main challenges companies faced with remote work?",
  "response": "Companies faced several challenges including communication gaps, team cohesion issues, and difficulties maintaining company culture...",
  "raw_document": "[Full original document]",
  "faithfulness_judgment": "YES"
}
```

---

### Enhanced Flows Comparison

| Feature | Extractive | Detailed | Key Facts | Document-Based |
|---------|-----------|----------|-----------|----------------|
| **Summary Type** | Annotated extracts | Thematic overview | Atomic facts | None |
| **n Parameter** | 50 (default) | 50 (default) | N/A | N/A |
| **QA per Document** | ~50 | ~50 | ~100+ | ~1-3 |
| **Input Complexity** | High (icl_* required) | High (icl_* required) | Low (no icl_*) | High (icl_* required) |
| **Processing Time** | High | High | Medium | Low |
| **Best For** | Context-rich learning | Conceptual learning | Factual recall | Quick QA generation |
| **Quality Filter** | Faithfulness | Faithfulness | None | Faithfulness |
| **Output Volume** | High | High | Very High | Low |

### Complete Workflow Example

Generate all 4 flow types for comprehensive knowledge tuning:

```python
from datasets import load_dataset
from sdg_hub.core.flow import Flow, FlowRegistry
import os

# Setup
FlowRegistry.discover_flows()

def set_model_config(flow):
    flow.set_model_config(
        model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
        api_base=os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
        api_key=os.getenv("VLLM_API_KEY", "EMPTY")
    )
    return flow

# Load seed data
seed_data = load_dataset("json", data_files="seed_data.jsonl", split="train")

# 1. Extractive Summary
print("Generating extractive summaries...")
flow = Flow.from_yaml(FlowRegistry.get_flow_path(
    "Extractive Summary Knowledge Tuning Dataset Generation Flow"
))
flow = set_model_config(flow)
extractive_data = flow.generate(
    seed_data,
    runtime_params={"gen_extractive_summary": {"n": 50}},
    max_concurrency=50
)
extractive_data.to_json("output/extractive_summary/gen.jsonl", orient="records", lines=True)

# 2. Detailed Summary
print("Generating detailed summaries...")
flow = Flow.from_yaml(FlowRegistry.get_flow_path(
    "Detailed Summary Knowledge Tuning Dataset Generation Flow"
))
flow = set_model_config(flow)
detailed_data = flow.generate(
    seed_data,
    runtime_params={"gen_detailed_summary": {"n": 50}},
    max_concurrency=50
)
detailed_data.to_json("output/detailed_summary/gen.jsonl", orient="records", lines=True)

# 3. Key Facts
print("Generating key facts...")
flow = Flow.from_yaml(FlowRegistry.get_flow_path(
    "Key Facts Knowledge Tuning Dataset Generation Flow"
))
flow = set_model_config(flow)
key_facts_data = flow.generate(seed_data, max_concurrency=50)
key_facts_data.to_json("output/key_facts/gen.jsonl", orient="records", lines=True)

# 4. Document-Based
print("Generating document-based QA...")
flow = Flow.from_yaml(FlowRegistry.get_flow_path(
    "Document Based Knowledge Tuning Dataset Generation Flow"
))
flow = set_model_config(flow)
doc_qa_data = flow.generate(seed_data, max_concurrency=50)
doc_qa_data.to_json("output/document_qa/gen.jsonl", orient="records", lines=True)

print(f"""
Generation Complete:
  Extractive: {len(extractive_data)} QA pairs
  Detailed: {len(detailed_data)} QA pairs
  Key Facts: {len(key_facts_data)} QA pairs
  Document-Based: {len(doc_qa_data)} QA pairs
  Total: {len(extractive_data) + len(detailed_data) + len(key_facts_data) + len(doc_qa_data)} QA pairs
""")
```

---

## InstructLab Multi-Summary QA Flow

**Name:** `Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning`

**Purpose:** Generate highest-quality QA pairs with comprehensive three-stage evaluation (faithfulness, relevancy, question verification).

**Location:** `src/sdg_hub/flows/qa_generation/document_grounded_qa/multi_summary_qa/instructlab/`

### Architecture

**Multi-Stage Pipeline:**

```yaml
Document → 3 Summary Types (detailed, extractive, atomic) →
Melt to Unified Dataset → QA Generation →
Triple Evaluation (faithfulness, relevancy, verification) →
Filtered High-Quality QA
```

**Key Differences from Enhanced Flows:**

1. **Combined Approach**: Generates all 3 summary types in one flow
2. **Triple Evaluation**:
   - Faithfulness: Is answer grounded in document?
   - Relevancy: Does answer address the question? (score ≥ 2.0)
   - Verification: Is question well-formed? (rating ≥ 1.0)
3. **Lower n Parameter**: `n=2` for detailed summaries (quality over quantity)
4. **MeltColumnsBlock**: Combines summary types into unified dataset

### Input Requirements

| Column | Description | Required |
|--------|-------------|----------|
| `document` | Full document text | Yes |
| `document_outline` | Document title/outline | Yes |
| `domain` | Content domain | Yes |
| `icl_document` | Example document for in-context learning | Yes |
| `icl_query_1-3` | Example questions | Yes |
| `icl_response_1-3` | Example responses | Yes |

Note: Requires example **responses** (not just queries like enhanced flows)

### Output Columns

- `question` - Generated question
- `response` - Generated answer
- `raw_document` - Original document
- `dataset_type` - Source summary type (detailed/extractive/atomic/document)
- `faithfulness_explanation`, `faithfulness_judgment`
- `relevancy_explanation`, `relevancy_score`
- `verification_explanation`, `verification_rating`

### Key Parameters

```python
runtime_params = {
    "gen_detailed_summary": {
        "n": 2,              # Only 2 detailed summaries (vs 50 in enhanced)
        "max_tokens": 2048
    },
    "knowledge_generation": {
        "temperature": 0.0,  # Deterministic for consistency
        "max_tokens": 2048
    }
}
```

### When to Use
- Need high-volume generation (50+ QAs per document)
- Want specific augmentation types separately

### Performance Characteristics

| Metric | InstructLab | Enhanced Flows |
|--------|-------------|----------------|
| QA per Document | ~10-20 | ~50-100+ |
| Evaluation Stages | 3 | 1 |
| Processing Time | High | Medium |
| Output Quality | Highest | High |
| Failure Rate | Higher (stricter) | Lower |

### Example Usage

```python
from sdg_hub.core.flow import Flow, FlowRegistry
from datasets import Dataset

# Load flow
FlowRegistry.discover_flows()
flow_path = FlowRegistry.get_flow_path(
    "Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning"
)
flow = Flow.from_yaml(flow_path)

# Configure model
flow.set_model_config(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key="your_key"
)

# Prepare input (note: includes icl_response fields)
dataset = Dataset.from_dict({
    "document": ["Your document..."],
    "document_outline": ["Title"],
    "domain": ["educational"],
    "icl_document": ["Example doc..."],
    "icl_query_1": ["Example question 1?"],
    "icl_response_1": ["Example answer 1"],
    "icl_query_2": ["Example question 2?"],
    "icl_response_2": ["Example answer 2"],
    "icl_query_3": ["Example question 3?"],
    "icl_response_3": ["Example answer 3"]
})

# Generate with triple evaluation
result = flow.generate(dataset, max_concurrency=10)

# Filter only highest quality
high_quality = result.filter(
    lambda x: (x['faithfulness_judgment'] == 'YES' and
               x['relevancy_score'] >= 2.0 and
               x['verification_rating'] >= 1.0)
)

print(f"Generated {len(result)} total, {len(high_quality)} high-quality QA pairs")
```

### Evaluation Details

**1. Faithfulness Evaluation:**
```yaml
Prompt: "Is this answer faithful to the document?"
Output: [Start of Explanation]...[End of Explanation]
        [Start of Answer]YES/NO[End of Answer]
Filter: Keep only "YES"
```

**2. Relevancy Evaluation:**
```yaml
Prompt: "Rate how well the answer addresses the question (0.0-2.0)"
Output: [Start of Feedback]...[End of Feedback]
        [Start of Score]2.0[End of Score]
Filter: Keep score ≥ 2.0
```

**3. Question Verification:**
```yaml
Prompt: "Rate question quality (-1.0 to 1.0)"
Output: [Start of Explanation]...[End of Explanation]
        [Start of Rating]1.0[End of Rating]
Filter: Keep rating ≥ 1.0
```

---

## Japanese Multilingual Multi-Summary QA Flow

**Name:** `Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning` (Japanese)

**Purpose:** Localized version of InstructLab flow for Japanese language training data generation.

**Location:** `src/sdg_hub/flows/qa_generation/document_grounded_qa/multi_summary_qa/multilingual/japanese/`

### Architecture

Same as InstructLab flow but with:
- All prompts translated to Japanese
- Japanese prompt YAML files (suffixed with `_ja.yaml`)
- Same block structure and evaluation stages

### Files

```
japanese/
├── flow.yaml                        # Main flow (identical blocks structure)
├── atomic_facts_ja.yaml            # Japanese atomic facts prompt
├── detailed_summary_ja.yaml        # Japanese detailed summary prompt
├── extractive_summary_ja.yaml      # Japanese extractive summary prompt
└── generate_questions_responses_ja.yaml  # Japanese QA generation prompt
```

### Input Requirements

Same structure as InstructLab, but with **Japanese text** in document fields:

```python
dataset = Dataset.from_dict({
    "document": ["日本語の文書内容..."],
    "document_outline": ["文書のタイトル"],
    "domain": ["記事/エッセイ"],
    "icl_document": ["日本語の例..."],
    "icl_query_1": ["質問の例1?"],
    "icl_response_1": ["回答の例1"],
    # ... etc
})
```

### Output Columns

Same as InstructLab flow:
- Japanese question and response
- Evaluation metrics (faithfulness, relevancy, verification)

### When to Use

- Generating Japanese training data for multilingual models
- Fine-tuning Japanese language models
- Creating Japanese knowledge tuning datasets
- Supporting Japanese-speaking users

### Example Usage

```python
from sdg_hub.core.flow import Flow, FlowRegistry

# Load Japanese flow
FlowRegistry.discover_flows()
flow_path = FlowRegistry.get_flow_path(
    "Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning"
)

# Note: Disambiguate if needed by checking metadata
flows = FlowRegistry.list_flows()
for fname in flows:
    metadata = FlowRegistry.get_flow_metadata(fname)
    if metadata and "japanese" in metadata.tags:
        flow_path = FlowRegistry.get_flow_path(fname)
        break

flow = Flow.from_yaml(flow_path)

# Configure model (use model with Japanese support)
flow.set_model_config(
    model="meta-llama/Llama-3.3-70B-Instruct",  # Supports Japanese
    api_key="your_key"
)

# Generate Japanese QA pairs
result = flow.generate(japanese_dataset, max_concurrency=10)
```

### Extending to Other Languages

To create a new language variant:

1. **Create directory structure:**
   ```
   multilingual/
   └── {language}/
       ├── flow.yaml
       ├── atomic_facts_{lang}.yaml
       ├── detailed_summary_{lang}.yaml
       ├── extractive_summary_{lang}.yaml
       └── generate_questions_responses_{lang}.yaml
   ```

2. **Copy and translate prompts:**
   - Start from `instructlab/*.yaml` or `japanese/*_ja.yaml`
   - Translate system and user messages
   - Preserve formatting tags and structure

3. **Update flow.yaml metadata:**
   ```yaml
   metadata:
     tags:
       - "multilingual"
       - "{language}"  # e.g., "spanish", "french"
     dataset_requirements:
       description: "Input dataset with {language} text..."
   ```

4. **Update prompt paths in flow.yaml:**
   ```yaml
   prompt_config_path: detailed_summary_{lang}.yaml
   ```

5. **Test with native speakers** to ensure quality

---

## Structured Text Insights Extraction Flow

**Name:** `Structured Text Insights Extraction Flow`

**Purpose:** Extract structured NLP insights (summary, keywords, entities, sentiment) for content analysis and metadata generation.

**Category:** Text Analysis (not QA generation)

**Location:** `src/sdg_hub/flows/text_analysis/structured_insights/`

### Architecture

**Parallel Extraction Pipeline:**

```yaml
Text → ┌─ Summary Extraction ─┐
       ├─ Keywords Extraction ─┤
       ├─ Entities Extraction ─┤  → JSON Structure → Structured Insights
       └─ Sentiment Analysis ──┘
```

All extractions run in **parallel** (async mode) for efficiency.

### What It Does

Performs 4 parallel LLM-powered analyses:

1. **Summary**: Concise overview of content (max 1024 tokens)
2. **Keywords**: Key terms and phrases (max 512 tokens)
3. **Entities**: Named entities (people, places, organizations)
4. **Sentiment**: Sentiment classification with justification

Then combines into structured JSON output via `JSONStructureBlock`.

### Input Requirements

| Column | Description | Minimum |
|--------|-------------|---------|
| `text` | Content to analyze | 50 words |

Suitable for:
- News articles
- Blog posts
- Product reviews
- Social media content
- Customer feedback

### Output Columns

- `summary` - Text summary
- `keywords` - Extracted keywords
- `entities` - Named entities
- `sentiment` - Sentiment analysis
- `structured_insights` - JSON combining all above

### Key Parameters

```python
runtime_params = {
    "generate_summary": {
        "max_tokens": 1024,
        "temperature": 0.3  # Low temperature for factual extraction
    },
    "generate_keywords": {
        "max_tokens": 512,
        "temperature": 0.3
    },
    "generate_entities": {
        "max_tokens": 1024,
        "temperature": 0.3
    },
    "generate_sentiment": {
        "max_tokens": 256,
        "temperature": 0.1  # Very low for consistent classification
    }
}
```

### When to Use

✅ **Use Structured Insights Flow For:**
- Content categorization and tagging
- Metadata extraction for search/indexing
- Sentiment monitoring and analysis
- Entity extraction for knowledge graphs
- Quick content analysis at scale

❌ **Don't Use For:**
- Training data generation (use QA flows instead)
- Question-answer pairs
- Knowledge tuning datasets
- Document augmentation

### Performance Characteristics

- **Fast**: All extractions run in parallel
- **Efficient**: Lower token limits (256-1024 vs 2048-4096)
- **Deterministic**: Low temperature settings
- **Scalable**: Designed for high-volume content processing

### Example Usage

```python
from datasets import Dataset
from sdg_hub.core.flow import Flow, FlowRegistry

# Load flow
FlowRegistry.discover_flows()
flow_path = FlowRegistry.get_flow_path("Structured Text Insights Extraction Flow")
flow = Flow.from_yaml(flow_path)

# Configure model
flow.set_model_config(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key="your_key"
)

# Prepare content
articles = Dataset.from_dict({
    "text": [
        "Article 1 content with at least 50 words...",
        "Article 2 content with at least 50 words...",
        # ... more articles
    ]
})

# Extract insights
result = flow.generate(articles, max_concurrency=20)

# Access structured output
for row in result:
    print(f"Summary: {row['summary']}")
    print(f"Keywords: {row['keywords']}")
    print(f"Entities: {row['entities']}")
    print(f"Sentiment: {row['sentiment']}")
    print(f"JSON: {row['structured_insights']}")
    print("---")
```

### Example Output

```json
{
  "text": "The new AI-powered feature received overwhelmingly positive feedback...",
  "summary": "[SUMMARY]The announcement of an AI feature garnered positive user response...[/SUMMARY]",
  "keywords": "[KEYWORDS]AI-powered, feature, positive feedback, user response[/KEYWORDS]",
  "entities": "[ENTITIES]Tech Company (ORG), Product Team (ORG)[/ENTITIES]",
  "sentiment": "[SENTIMENT]Positive - Users expressed enthusiasm and satisfaction...[/SENTIMENT]",
  "structured_insights": {
    "summary": "The announcement of an AI feature garnered positive user response...",
    "keywords": ["AI-powered", "feature", "positive feedback"],
    "entities": ["Tech Company", "Product Team"],
    "sentiment": "Positive"
  }
}
```