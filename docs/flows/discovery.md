# Flow Discovery

Learn how to discover, organize, and manage flows in SDG Hub. The discovery system automatically finds and registers flows, making them available for use without manual configuration.

## 🔍 Auto-Discovery System

SDG Hub automatically discovers flows in the `src/sdg_hub/flows/` directory using a hierarchical organization system.

### Discovery Process

```python
from sdg_hub.core.flow import FlowRegistry

# Auto-discover all flows in the system
FlowRegistry.discover_flows()

# This scans the flows directory and registers all valid flows
# No manual registration required!
```

### Discovery Locations

The discovery system searches these locations:

```
src/sdg_hub/flows/                    # Built-in flows
├── qa_generation/                    # Question-answer generation
│   ├── document_grounded_qa/
│   │   ├── enhanced_multi_summary_qa/
│   │   │   └── flow.yaml
│   │   └── multi_summary_qa/
│   │       └── multilingual/
│   │           └── japanese/
│   │               └── flow.yaml
│   └── simple_qa/
│       └── flow.yaml
├── text_processing/                  # Text manipulation flows
│   ├── summarization/
│   └── classification/
└── evaluation/                       # Quality assessment flows
    ├── quality_assessment/
    └── bias_detection/

```

## 📋 Flow Registry Operations

### Listing Available Flows

```python
from sdg_hub.core.flow import FlowRegistry

# Discover flows first
FlowRegistry.discover_flows()

# List all available flows
all_flows = FlowRegistry.list_flows()
print(f"Found {len(all_flows)} flows:")
for flow_name in all_flows:
    print(f"  • {flow_name}")

# Output:
# Found 3 flows:
#   • Extractive Summary Knowledge Tuning Dataset Generation Flow
#   • Simple QA Generation Flow
#   • Document Summarization Flow
```

### Getting Flow Information

Access detailed flow metadata and configuration:

```python
from sdg_hub.core.flow import FlowRegistry, Flow

# Get metadata for a specific flow
flow_name = "Extractive Summary Knowledge Tuning Dataset Generation Flow"
metadata = FlowRegistry.get_flow_metadata(flow_name)

if metadata:
    print(f"Flow: {metadata.name}")
    print(f"Version: {metadata.version}")
    print(f"Author: {metadata.author}")
    print(f"Description: {metadata.description}")
    print(f"Tags: {', '.join(metadata.tags)}")
    print(f"Recommended model: {metadata.recommended_models.get('default', 'Not specified')}")

# Load flow and get detailed information
flow_path = FlowRegistry.get_flow_path(flow_name)
flow = Flow.from_yaml(flow_path)

# Get comprehensive flow info
info = flow.get_info()
print(f"Total blocks: {info['total_blocks']}")
print(f"Block sequence: {', '.join(info['block_names'])}")

# Get dataset requirements
requirements = flow.get_dataset_requirements()
if requirements:
    print(f"Required columns: {requirements.required_columns}")
    print(f"Description: {requirements.description}")
    print(f"Min samples: {requirements.min_samples}")

# Get model recommendations
recommendations = flow.get_model_recommendations()
print(f"Default model: {recommendations.get('default')}")
print(f"Compatible models: {recommendations.get('compatible', [])}")
```

### Getting Flow Paths

```python
# Get the file path for a flow
flow_path = FlowRegistry.get_flow_path(flow_name)
print(f"Flow located at: {flow_path}")

# Use the path to load the flow
from sdg_hub.core.flow import Flow
flow = Flow.from_yaml(flow_path)
```

## 🔎 Searching and Filtering Flows

### Search by Tags

```python
# Search for flows with specific tags
qa_flows = FlowRegistry.search_flows(tag="question-generation")
print(f"Q&A Generation flows: {qa_flows}")

educational_flows = FlowRegistry.search_flows(tag="educational")
print(f"Educational flows: {educational_flows}")

document_flows = FlowRegistry.search_flows(tag="document-processing")
print(f"Document processing flows: {document_flows}")

# Get all flows organized by category
flows_by_category = FlowRegistry.get_flows_by_category()
# Returns: {"knowledge-tuning": [{"id": "...", "name": "..."}, ...], "text-analysis": [...], ...}
```


## 📊 Flow Organization Patterns

### Hierarchical Organization

Flows are organized in a logical hierarchy:

```
flows/
├── domain/              # By problem domain
│   ├── qa_generation/
│   ├── text_processing/
│   ├── evaluation/
│   └── data_preparation/
├── use_case/            # By specific use case
│   ├── document_grounded_qa/
│   ├── conversational_qa/
│   └── multi_turn_dialogue/
└── variant/             # By implementation variant
    ├── simple/          # Simplified version
    ├── advanced/        # Feature-rich version
    └── multilingual/    # Language-specific variants
```

### Flow Naming Conventions

Follow consistent naming patterns:

```yaml
# Good naming examples
metadata:
  name: "Extractive Summary Knowledge Tuning Dataset Generation Flow"
  name: "Simple Text Summarization Flow"
  name: "Multi-Turn Dialogue Generation with Context Tracking"

```

### Directory Structure Guidelines

Organize flows logically:

```
qa_generation/                           # Primary domain
├── document_grounded_qa/                # Specific approach
│   ├── enhanced_multi_summary_qa/       # Enhanced implementation
│   │   ├── flow.yaml                   # Main flow definition
│   │   └── README.md                   # Flow documentation
│   └── multi_summary_qa/               # Multi-summary variant
│       └── multilingual/               # Language-specific
│           └── japanese/               # Japanese language support
│               ├── flow.yaml
│               ├── atomic_facts_ja.yaml
│               └── detailed_summary_ja.yaml
```

## 🏷️ Flow Categorization and Tagging

### Why Tags Matter

Tags make flows discoverable and help users quickly identify the right tool for their task. When creating or using flows, proper tagging ensures:

- **Fast Discovery**: Users can find flows by searching for purpose, output type, or domain
- **Clear Intent**: Tags communicate what a flow does at a glance
- **Organized Ecosystem**: Consistent tagging builds a searchable, maintainable flow library

**Key Principle**: Tags are automatically converted to lowercase, so use lowercase hyphenated phrases (e.g., `knowledge-tuning`, not `KnowledgeTuning`).

---

### How to Tag Your Flow

When creating a new flow, select tags from these **four categories**. Aim for **3-7 total tags** combining different categories:

#### 1. Purpose Tags (Choose 1-2)

**What the flow does** - Select the main goal(s) of your flow:

| Tag | When to Use |
|-----|-------------|
| `question-generation` | Your flow creates questions from documents |
| `knowledge-extraction` | Your flow pulls knowledge or facts from documents |
| `knowledge-tuning` | Your flow generates training data for knowledge tuning/internalization |
| `document-internalization` | Your flow helps models learn and internalize document content |
| `text-analysis` | Your flow analyzes text for insights, patterns, or characteristics |
| `summarization` | Your flow creates summaries (any type) |
| `sentiment-analysis` | Your flow determines sentiment or emotional tone |
| `entity-extraction` | Your flow identifies entities (people, places, organizations) |
| `keyword-extraction` | Your flow identifies key terms or phrases |

**Example:**
```yaml
# A flow that generates QA pairs for knowledge tuning
tags:
  - "knowledge-tuning"        # Main purpose
  - "question-generation"     # Method used
```

---

#### 2. Output Type Tags (Choose 1-2)

**What the flow produces** - Describe the primary output format:

| Tag | When to Use |
|-----|-------------|
| `qa-pairs` | Your flow produces question-answer pairs |
| `extractive-summaries` | Your flow creates extractive summaries (selected passages from source) |
| `detailed-summaries` | Your flow creates detailed/abstractive summaries |
| `key-facts` | Your flow produces atomic facts or key facts |
| `structured-output` | Your flow outputs structured JSON data |
| `insights` | Your flow generates analytical insights or metadata |

**Example:**
```yaml
# A flow that creates extractive summaries then QA pairs
tags:
  - "qa-pairs"               # Final output
  - "extractive-summaries"   # Intermediate output
```

---

#### 3. Domain/Application Tags (Choose 0-1)

**Where it's used** - Add a domain tag if your flow targets a specific use case:

| Tag | When to Use |
|-----|-------------|
| `educational` | Your flow is designed for educational content or training materials |
| `document-processing` | Your flow is part of document transformation/processing pipelines |
| `nlp` | Your flow performs general NLP or text mining tasks |

**Example:**
```yaml
# An educational QA generation flow
tags:
  - "knowledge-tuning"
  - "qa-pairs"
  - "educational"            # Targets education domain
```

---

#### 4. Technical/Method Tags (Choose 0-2)

**Special characteristics** - Add technical tags for unique features or methods:

| Tag | When to Use |
|-----|-------------|
| `knowledge-extractive-summary` | Your flow uses extractive summarization specifically for knowledge extraction |
| `multilingual` | Your flow supports non-English languages |
| `japanese` / `spanish` / etc. | Your flow is localized for a specific language |

**Example:**
```yaml
# A Japanese multilingual knowledge tuning flow
tags:
  - "knowledge-tuning"
  - "qa-pairs"
  - "multilingual"           # Supports non-English
  - "japanese"               # Specific language
```

---

### Tagging Best Practices

#### ✅ DO: Aim for 3-7 Tags Total

Balance discoverability with clarity:

```yaml
# ✅ GOOD - Clear and discoverable (5 tags)
tags:
  - "knowledge-tuning"           # Purpose
  - "question-generation"        # Purpose
  - "qa-pairs"                   # Output
  - "extractive-summaries"       # Output
  - "educational"                # Domain

# ❌ TOO FEW - Hard to discover (1 tag)
tags:
  - "qa-pairs"

# ❌ TOO MANY - Unclear focus (10 tags)
tags:
  - "knowledge-tuning"
  - "document-internalization"
  - "question-generation"
  - "knowledge-extraction"
  - "qa-pairs"
  - "extractive-summaries"
  - "educational"
  - "document-processing"
  - "text-analysis"
  - "summarization"
```

#### ✅ DO: Mix Tag Categories

Combine tags from different categories for best results:

```yaml
# ✅ GOOD - Purpose + Output + Domain
tags:
  - "knowledge-tuning"        # Purpose (category 1)
  - "question-generation"     # Purpose (category 1)
  - "qa-pairs"               # Output (category 2)
  - "educational"            # Domain (category 3)
```

#### ✅ DO: Be Specific

Choose the most precise tag available:

```yaml
# ✅ GOOD - Specific tag
tags:
  - "extractive-summaries"

# ❌ BAD - Too generic
tags:
  - "summaries"
```

#### ✅ DO: Order by Importance

List primary tags first:

```yaml
# ✅ GOOD - Most important tags first
tags:
  - "knowledge-tuning"              # Primary purpose (most important)
  - "document-internalization"      # Primary purpose
  - "question-generation"           # Method used
  - "qa-pairs"                      # Output produced
  - "key-facts"                     # Specific output type
```

---

### Using Tags to Find Flows

Once you understand tagging conventions, use tags to quickly find the right flow:

```python
from sdg_hub.core.flow import FlowRegistry

# Discover all flows
FlowRegistry.discover_flows()

# Find flows by what they do (purpose)
qa_flows = FlowRegistry.search_flows(tag="question-generation")
knowledge_flows = FlowRegistry.search_flows(tag="knowledge-tuning")
analysis_flows = FlowRegistry.search_flows(tag="text-analysis")

# Find flows by what they produce (output)
extractive_flows = FlowRegistry.search_flows(tag="extractive-summaries")
fact_flows = FlowRegistry.search_flows(tag="key-facts")
structured_flows = FlowRegistry.search_flows(tag="structured-output")

# Find flows by domain
educational_flows = FlowRegistry.search_flows(tag="educational")
nlp_flows = FlowRegistry.search_flows(tag="nlp")

# Find flows by language
multilingual_flows = FlowRegistry.search_flows(tag="multilingual")
japanese_flows = FlowRegistry.search_flows(tag="japanese")

# Example: Find all knowledge tuning flows
print(f"Found {len(knowledge_flows)} knowledge tuning flows:")
for flow_name in knowledge_flows:
    metadata = FlowRegistry.get_flow_metadata(flow_name)
    print(f"  - {metadata.name}")
    print(f"    Tags: {', '.join(metadata.tags)}")
```

---

### Creating New Tags

If your flow has unique functionality not covered by existing tags:

#### Step 1: Check Existing Tags First

```python
# See all flows and their tags
FlowRegistry.discover_flows()
all_flows = FlowRegistry.list_flows()

for flow_name in all_flows:
    metadata = FlowRegistry.get_flow_metadata(flow_name)
    print(f"{metadata.name}: {metadata.tags}")
```

#### Step 2: Follow Naming Conventions

**DO:**
- ✅ Use lowercase: `knowledge-tuning`
- ✅ Use hyphens: `sentiment-analysis`
- ✅ Be descriptive: `extractive-summaries`
- ✅ Use full words: `multilingual`

**DON'T:**
- ❌ Use CamelCase: `KnowledgeTuning`
- ❌ Use underscores: `knowledge_tuning`
- ❌ Be too generic: `summaries`
- ❌ Use abbreviations: `ml`

#### Step 3: Add Your New Tag

```yaml
metadata:
  name: "Document Classification Flow"
  tags:
    - "text-analysis"           # Existing purpose tag
    - "classification"          # NEW: Specific purpose
    - "multi-label"            # NEW: Technical method
    - "structured-output"       # Existing output tag
```

#### Step 4: Document New Tags (Optional)

If your new tag represents a common pattern others might use, consider updating the tag tables in this documentation via a pull request.


## 🚀 Next Steps

Master flow discovery and organization:

- **[Custom Flows](custom-flows.md)** - Build and organize your own flows
- **[Flow Overview](overview.md)** - YAML structure and configuration
- **[Development Guide](../development.md)** - Contribute flows to the ecosystem
- **[API Reference](../api-reference.md)** - Complete technical documentation