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

### Use Cases

**Content Management:**
```python
# Tag blog posts automatically
blog_posts = load_blog_posts()
insights = flow.generate(blog_posts)
for post, insight in zip(blog_posts, insights):
    post.tags = insight['keywords']
    post.summary = insight['summary']
```

**Sentiment Monitoring:**
```python
# Track customer feedback sentiment
reviews = load_customer_reviews()
insights = flow.generate(reviews)
positive = insights.filter(lambda x: 'positive' in x['sentiment'].lower())
print(f"Positive reviews: {len(positive)}/{len(insights)}")
```

**Search Indexing:**
```python
# Build search index with entities and keywords
documents = load_documents()
insights = flow.generate(documents)
for doc, insight in zip(documents, insights):
    search_index.add(
        doc_id=doc.id,
        entities=insight['entities'],
        keywords=insight['keywords'],
        summary=insight['summary']
    )
```

---

## Flow Comparison Matrix

Comprehensive comparison of all available flows:

### By Category and Purpose

| Flow Name | Category | Primary Output | Volume | Quality | Speed | Use Case |
|-----------|----------|----------------|--------|---------|-------|----------|
| **Extractive Summary** | Knowledge Tuning | QA pairs | High (50+) | High | Medium | Context-rich knowledge |
| **Detailed Summary** | Knowledge Tuning | QA pairs | High (50+) | High | Medium | Conceptual learning |
| **Key Facts** | Knowledge Tuning | QA pairs | Very High (100+) | Medium | Fast | Factual recall |
| **Document-Based QA** | Knowledge Tuning | QA pairs | Low (1-3) | High | Fast | Quick QA generation |
| **InstructLab Multi-Summary** | Knowledge Tuning | QA pairs | Medium (10-20) | Highest | Slow | Production training |
| **Japanese Multi-Summary** | Knowledge Tuning | QA pairs (JP) | Medium (10-20) | Highest | Slow | Japanese training |
| **Structured Insights** | Text Analysis | NLP metadata | N/A | N/A | Fast | Content analysis |

### By Input Requirements

| Flow | document | document_outline | domain | icl_document | icl_query_* | icl_response_* | text |
|------|----------|------------------|--------|--------------|-------------|----------------|------|
| Extractive Summary | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Detailed Summary | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Key Facts | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Document-Based QA | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| InstructLab | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Japanese | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Structured Insights | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

### By Evaluation and Filtering

| Flow | Faithfulness | Relevancy | Verification | Filter Rate | Final Quality |
|------|--------------|-----------|--------------|-------------|---------------|
| Extractive Summary | ✅ | ❌ | ❌ | ~10-20% | High |
| Detailed Summary | ✅ | ❌ | ❌ | ~10-20% | High |
| Key Facts | ❌ | ❌ | ❌ | 0% | Medium |
| Document-Based QA | ✅ | ❌ | ❌ | ~10-20% | High |
| InstructLab | ✅ | ✅ | ✅ | ~40-60% | Highest |
| Japanese | ✅ | ✅ | ✅ | ~40-60% | Highest |
| Structured Insights | ❌ | ❌ | ❌ | 0% | N/A |

---

## Common Usage Patterns

### Pattern 1: Flow Discovery and Selection

```python
from sdg_hub.core.flow import FlowRegistry

# Discover all flows
FlowRegistry.discover_flows()

# List all available
all_flows = FlowRegistry.list_flows()
print(f"Total flows: {len(all_flows)}")

# Search by purpose
qa_flows = FlowRegistry.search_flows(tag="question-generation")
print(f"QA Generation flows: {len(qa_flows)}")

knowledge_flows = FlowRegistry.search_flows(tag="knowledge-tuning")
print(f"Knowledge Tuning flows: {len(knowledge_flows)}")

# Get detailed information
for flow_name in qa_flows:
    metadata = FlowRegistry.get_flow_metadata(flow_name)
    if metadata:
        print(f"\n{metadata.name}")
        print(f"  Version: {metadata.version}")
        print(f"  Tags: {', '.join(metadata.tags)}")
        print(f"  Required columns: {metadata.dataset_requirements.required_columns}")
        print(f"  Default model: {metadata.recommended_models.get('default', 'N/A')}")
```

### Pattern 2: Model Configuration for Different Providers

```python
from sdg_hub.core.flow import Flow
import os

def configure_hosted_vllm(flow):
    """Configure for self-hosted vLLM server"""
    flow.set_model_config(
        model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
        api_base=os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
        api_key=os.getenv("VLLM_API_KEY", "EMPTY")
    )
    return flow

def configure_openai(flow):
    """Configure for OpenAI API"""
    flow.set_model_config(
        model="openai/gpt-4",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return flow

def configure_ollama(flow):
    """Configure for local Ollama"""
    flow.set_model_config(
        model="ollama/gemma2",
        api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    )
    return flow

def configure_maas(flow):
    """Configure for Model-as-a-Service"""
    flow.set_model_config(
        model=os.getenv("MAAS_MODEL"),
        api_base=os.getenv("MAAS_API_BASE"),
        api_key=os.getenv("MAAS_API_KEY")
    )
    return flow

# Usage
flow = Flow.from_yaml("path/to/flow.yaml")
flow = configure_hosted_vllm(flow)  # Choose your provider
```

### Pattern 3: Runtime Parameter Customization

```python
# Adjust generation parameters per block
runtime_params = {
    # Control summary generation
    "gen_extractive_summary": {
        "n": 30,              # Generate 30 instead of default 50
        "max_tokens": 6000,   # Longer summaries
        "temperature": 0.8    # More creative
    },

    # Control question generation
    "question_generation": {
        "temperature": 0.9,   # Very creative questions
        "max_tokens": 512
    },

    # Control answer generation
    "answer_generation": {
        "temperature": 0.5,   # More focused answers
        "max_tokens": 2048
    }
}

result = flow.generate(dataset, runtime_params=runtime_params)
```

### Pattern 4: Batch Processing with Checkpointing

```python
from datasets import load_dataset
import os

def process_large_dataset(flow, dataset, output_dir):
    """Process large dataset with checkpointing"""

    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Process with checkpointing
    result = flow.generate(
        dataset,
        checkpoint_dir=checkpoint_dir,
        save_freq=100,        # Save every 100 samples
        max_concurrency=50,   # Control API load
        runtime_params={
            "gen_extractive_summary": {"n": 30}
        }
    )

    # Save final output
    output_path = os.path.join(output_dir, "final_output.jsonl")
    result.to_json(output_path, orient="records", lines=True)

    return result

# Usage
large_dataset = load_dataset("json", data_files="large_seed_data.jsonl", split="train")
flow = Flow.from_yaml(FlowRegistry.get_flow_path("Extractive Summary..."))
flow = configure_hosted_vllm(flow)

result = process_large_dataset(flow, large_dataset, "./output/extractive")
print(f"Processed {len(result)} samples")
```

### Pattern 5: Dry Run and Time Estimation

```python
from datasets import Dataset

# Load small sample
sample_data = dataset.select(range(2))

# Dry run with time estimation
dry_result = flow.dry_run(
    sample_data,
    sample_size=2,
    enable_time_estimation=True,
    max_concurrency=50
)

# Automatically shows Rich table with estimates:
# - Estimated total time
# - Per-block timing
# - API request counts

print(f"Dry run successful!")
print(f"Sample output columns: {dry_result['final_dataset']['columns']}")

# Now run on full dataset with confidence
full_result = flow.generate(dataset, max_concurrency=50)
```

### Pattern 6: Multi-Flow Pipeline

```python
def generate_comprehensive_training_data(seed_data, output_base_dir):
    """Generate all flow types for comprehensive coverage"""

    FlowRegistry.discover_flows()

    flows_to_run = [
        ("Extractive Summary Knowledge Tuning Dataset Generation Flow",
         "extractive", {"gen_extractive_summary": {"n": 50}}),
        ("Detailed Summary Knowledge Tuning Dataset Generation Flow",
         "detailed", {"gen_detailed_summary": {"n": 50}}),
        ("Key Facts Knowledge Tuning Dataset Generation Flow",
         "key_facts", {}),
        ("Document Based Knowledge Tuning Dataset Generation Flow",
         "doc_qa", {})
    ]

    results = {}

    for flow_name, output_name, runtime_params in flows_to_run:
        print(f"\n{'='*60}")
        print(f"Running: {flow_name}")
        print(f"{'='*60}")

        # Load and configure flow
        flow = Flow.from_yaml(FlowRegistry.get_flow_path(flow_name))
        flow = configure_hosted_vllm(flow)

        # Generate with checkpointing
        output_dir = os.path.join(output_base_dir, output_name)
        os.makedirs(output_dir, exist_ok=True)

        result = flow.generate(
            seed_data,
            runtime_params=runtime_params,
            checkpoint_dir=os.path.join(output_dir, "checkpoints"),
            save_freq=100,
            max_concurrency=50
        )

        # Save output
        result.to_json(
            os.path.join(output_dir, "gen.jsonl"),
            orient="records",
            lines=True
        )

        results[output_name] = result
        print(f"✓ Generated {len(result)} samples")

    # Print summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    total = sum(len(r) for r in results.values())
    for name, result in results.items():
        print(f"  {name:20s}: {len(result):6d} QA pairs")
    print(f"  {'TOTAL':20s}: {total:6d} QA pairs")

    return results

# Usage
seed_data = load_dataset("json", data_files="seed_data.jsonl", split="train")
results = generate_comprehensive_training_data(seed_data, "./training_data")
```

---

## Best Practices

### Choosing the Right Flow

**For Knowledge Tuning:**

1. **Start with Key Facts Flow** if:
   - Need maximum QA pair volume quickly
   - Documents are fact-dense (scientific, technical)
   - Budget for API calls is limited
   - Quality filtering not critical

2. **Use Extractive + Detailed** if:
   - Need both factual and conceptual learning
   - Want diverse augmentation types
   - Have compute budget for high-volume generation
   - Training general-purpose models

3. **Use InstructLab Flow** if:
   - Quality is paramount (production models)
   - Can afford extensive evaluation overhead
   - Need rigorous filtering (faithfulness + relevancy + verification)
   - Training InstructLab or similar frameworks

4. **Use Document-Based** if:
   - Need quick QA generation
   - Documents already well-structured
   - Don't need augmentation diversity
   - Testing pipelines before full generation

**For Content Analysis:**

Use **Structured Insights Flow** for:
- Content categorization and tagging
- Metadata extraction
- Sentiment monitoring
- Entity recognition
- **NOT for training data generation**

### Quality vs Quantity Tradeoffs

| Approach | Volume | Quality | Cost | Best For |
|----------|--------|---------|------|----------|
| Key Facts only | Highest | Medium | Low | Rapid prototyping |
| Extractive or Detailed | High | High | Medium | General training |
| All Enhanced Flows | Very High | High | High | Comprehensive datasets |
| InstructLab | Medium | Highest | Very High | Production models |

### Performance Optimization

**1. Use Appropriate Concurrency:**
```python
# Conservative (production, rate-limited APIs)
result = flow.generate(dataset, max_concurrency=5)

# Moderate (development, self-hosted)
result = flow.generate(dataset, max_concurrency=20)

# Aggressive (robust APIs, small datasets)
result = flow.generate(dataset, max_concurrency=50)
```

**2. Always Checkpoint Large Runs:**
```python
result = flow.generate(
    dataset,
    checkpoint_dir="./checkpoints",
    save_freq=100  # Save every 100 samples
)
```

**3. Tune Token Limits:**
```python
# Reduce tokens if getting truncated or timeout
runtime_params = {
    "gen_extractive_summary": {
        "max_tokens": 3000  # Lower than default 4096
    }
}
```

**4. Use Dry Runs:**
```python
# Always dry run before full generation
dry_result = flow.dry_run(
    dataset.select(range(2)),
    enable_time_estimation=True
)
```

### Data Mixing Strategies

After generating with multiple flows:

**Option 1: Separate Training**
```python
# Train on each augmentation type separately
train_on_extractive()
train_on_detailed()
train_on_key_facts()
```

**Option 2: Uniform Mixing**
```python
# Combine all outputs equally
combined = pd.concat([
    extractive_data,
    detailed_data,
    key_facts_data
])
train(combined.sample(frac=1))  # Shuffle
```

**Option 3: Weighted Mixing**
```python
# Weight by quality or diversity needs
mixed = pd.concat([
    extractive_data.sample(n=1000),
    detailed_data.sample(n=1000),
    key_facts_data.sample(n=2000)  # More key facts
])
```

See [knowledge_mixing.ipynb](../../examples/knowledge_tuning/enhanced_summary_knowledge_tuning/knowledge_mixing.ipynb) for complete mixing guide.

---

## Adding New Flows

Want to create a new flow? Follow these conventions:

### Directory Structure

```
flows/
└── {category}/              # e.g., qa_generation, text_analysis
    └── {subcategory}/       # e.g., document_grounded_qa
        └── {flow_name}/     # e.g., my_new_flow
            ├── __init__.py
            ├── flow.yaml         # Main flow definition (required)
            ├── prompt1.yaml      # Prompt configurations
            ├── prompt2.yaml
            └── README.md         # Flow-specific documentation
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

### Tag Conventions

Use consistent tags for discoverability:

**Category Tags:**
- `question-generation` - QA pair generation
- `knowledge-tuning` - Knowledge internalization
- `text-analysis` - NLP analysis
- `summarization` - Text summarization
- `evaluation` - Quality evaluation

**Technique Tags:**
- `document-internalization` - Document-to-QA
- `extractive-summaries` - Extractive summarization
- `detailed-summaries` - Abstractive summarization
- `key-facts` - Atomic fact extraction
- `structured-output` - JSON/structured outputs

**Language Tags:**
- `multilingual` - Non-English support
- `japanese`, `spanish`, etc. - Specific language

### Integration with Discovery

Flows are automatically discovered if:
1. Located in `src/sdg_hub/flows/`
2. Directory contains `flow.yaml`
3. Contains valid metadata section
4. Has `__init__.py` in directory

No manual registration needed!

### Documentation Template

Add to this file (`enhance_qa.md`):

```markdown
## Your New Flow Name

**Name:** `Full Flow Name from Metadata`

**Purpose:** One-sentence description

**Location:** `src/sdg_hub/flows/path/to/flow/`

### Architecture

Describe pipeline: Block1 → Block2 → Block3

### Input Requirements

| Column | Description | Required |
|--------|-------------|----------|
| col1 | Description | Yes |

### Output Columns

- `output1` - Description
- `output2` - Description

### Key Parameters

```python
runtime_params = {
    "block_name": {
        "param": value
    }
}
```

### When to Use

- Use case 1
- Use case 2

### Example Usage

```python
# Code example
```

### Example Output

```json
{
  "example": "output"
}
```
```

---

## Next Steps

### For Knowledge Tuning Workflows

1. **Generate Data** (you are here)
   - Use this guide to select and run appropriate flows
   - Generate QA pairs with multiple augmentation types

2. **Mix and Curate Data**
   - See [knowledge_mixing.ipynb](../../examples/knowledge_tuning/enhanced_summary_knowledge_tuning/knowledge_mixing.ipynb)
   - Combine flow outputs
   - Convert to training format (messages)
   - Balance and filter datasets

3. **Train Models**
   - Use mixed datasets for fine-tuning
   - Follow InstructLab or your framework's training process

### Learning More

- **[Flow System Overview](overview.md)** - Deep dive into flow architecture
- **[Flow Discovery](discovery.md)** - Advanced discovery and organization
- **[Block Documentation](../blocks/)** - Learn about individual blocks
- **[Example Notebooks](../../examples/)** - Complete end-to-end examples

### Example References

- **Enhanced QA Generation:** [knowledge_generation.ipynb](../../examples/knowledge_tuning/enhanced_summary_knowledge_tuning/knowledge_generation.ipynb)
- **Data Mixing:** [knowledge_mixing.ipynb](../../examples/knowledge_tuning/enhanced_summary_knowledge_tuning/knowledge_mixing.ipynb)
- **Custom Flow Creation:** See CLAUDE.md and existing flows

### Getting Help

- Check flow metadata: `FlowRegistry.get_flow_metadata(flow_name)`
- Review dataset requirements before running
- Always dry run first: `flow.dry_run(sample_data)`
- Use checkpointing for long runs
- Monitor metrics output for bottlenecks

---

**Last Updated:** 2025-01-30
**SDG Hub Version:** 0.2.0+
