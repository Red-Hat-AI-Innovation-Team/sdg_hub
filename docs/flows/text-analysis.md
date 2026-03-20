# Text Analysis Flows

Text analysis flows extract structured NLP insights from unstructured text content. Unlike QA generation flows designed for training data, these flows focus on content understanding, categorization, and metadata extraction.

## Structured Text Insights Extraction Flow

**Name:** `Structured Text Insights Extraction Flow`

**Location:** `src/sdg_hub/flows/text_analysis/structured_insights/`

**Tags:** `text-analysis`, `summarization`, `nlp`, `structured-output`, `insights`, `sentiment-analysis`, `entity-extraction`, `keyword-extraction`

### Architecture

The flow runs 4 LLM-powered analyses sequentially on each input text, then consolidates the results into a single JSON column:

```yaml
Text → Summary Extraction → Keywords Extraction → Entities Extraction → Sentiment Analysis → JSONStructureBlock → Structured Insights
```

Each extraction follows the same 4-block pattern:

1. **PromptBuilderBlock** — Constructs the analysis prompt from the input text
2. **LLMChatBlock** — Generates the analysis via LLM (async mode)
3. **LLMResponseExtractorBlock** — Extracts the response content
4. **TagParserBlock** — Parses tagged output (e.g., `[SUMMARY]...[/SUMMARY]`)

The final **JSONStructureBlock** combines all 4 parsed outputs into a single `structured_insights` JSON column.

### Pipeline Detail

| Stage | Block Name | Block Type | Output Column |
|-------|-----------|------------|---------------|
| Summary | `build_summary_prompt` | PromptBuilderBlock | `summary_prompt` |
| | `generate_summary` | LLMChatBlock | `raw_summary` |
| | `extract_summary` | LLMResponseExtractorBlock | `extract_summary_content` |
| | `parse_summary` | TagParserBlock | `summary` |
| Keywords | `build_keywords_prompt` | PromptBuilderBlock | `keywords_prompt` |
| | `generate_keywords` | LLMChatBlock | `raw_keywords` |
| | `extract_keywords` | LLMResponseExtractorBlock | `extract_keywords_content` |
| | `parse_keywords` | TagParserBlock | `keywords` |
| Entities | `build_entities_prompt` | PromptBuilderBlock | `entities_prompt` |
| | `generate_entities` | LLMChatBlock | `raw_entities` |
| | `extract_entities` | LLMResponseExtractorBlock | `extract_entities_content` |
| | `parse_entities` | TagParserBlock | `entities` |
| Sentiment | `build_sentiment_prompt` | PromptBuilderBlock | `sentiment_prompt` |
| | `generate_sentiment` | LLMChatBlock | `raw_sentiment` |
| | `extract_sentiment` | LLMResponseExtractorBlock | `extract_sentiment_content` |
| | `parse_sentiment` | TagParserBlock | `sentiment` |
| Combine | `create_structured_insights` | JSONStructureBlock | `structured_insights` |

### Input Requirements

| Column | Description | Minimum |
|--------|-------------|---------|
| `text` | Content to analyze | 50 words recommended |

Suitable content types: news articles, blog posts, product reviews, social media content, customer feedback.

### Output Columns

| Column | Description |
|--------|-------------|
| `summary` | 2-3 sentence concise summary |
| `keywords` | 10 key terms and phrases, comma-separated |
| `entities` | Named entities as JSON with `people`, `organizations`, `locations` |
| `sentiment` | Sentiment classification: `positive`, `negative`, or `neutral` |
| `structured_insights` | JSON object combining all four outputs above |

### Recommended Models

| Tier | Models |
|------|--------|
| Default | `openai/gpt-oss-120b` |
| Compatible | `meta-llama/Llama-3.3-70B-Instruct`, `microsoft/phi-4`, `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| Experimental | `gpt-4o` |

### Key Parameters

Each LLM block uses low temperature settings for consistent, factual output:

```python
runtime_params = {
    "generate_summary": {
        "max_tokens": 1024,
        "temperature": 0.3
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

### Prompt Details

Each extraction uses a dedicated prompt template in `prompts/`:

**`summarize.yaml`** — Asks for a 2-3 sentence summary capturing the most important information and key points. Output wrapped in `[SUMMARY]...[/SUMMARY]` tags.

**`extract_keywords.yaml`** — Extracts exactly 10 keywords or key phrases (1-3 words each) that are representative, searchable, and a mix of specific and broad terms. Output wrapped in `[KEYWORDS]...[/KEYWORDS]` tags.

**`extract_entities.yaml`** — Identifies named entities categorized as `people`, `organizations`, and `locations`. Returns structured JSON (3-8 entities per category max). Output wrapped in `[ENTITIES]...[/ENTITIES]` tags.

**`analyze_sentiment.yaml`** — Classifies overall sentiment as `positive`, `negative`, or `neutral` based on emotional tone, intensity, context, and balance. Output wrapped in `[SENTIMENT]...[/SENTIMENT]` tags.

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
        "The Federal Reserve announced a quarter-point interest rate cut on Wednesday, "
        "marking the third consecutive reduction this year. Fed Chair Jerome Powell cited "
        "slowing inflation and a cooling labor market as key factors in the decision. "
        "Markets responded positively, with the S&P 500 rising 1.2% in after-hours trading. "
        "Economists broadly welcomed the move, though some warned that further cuts could "
        "reignite inflationary pressures in the housing market.",
    ]
})

# Extract insights
result = flow.generate(articles, max_concurrency=20)

# Access individual fields
for _, row in result.iterrows():
    print(f"Summary: {row['summary']}")
    print(f"Keywords: {row['keywords']}")
    print(f"Entities: {row['entities']}")
    print(f"Sentiment: {row['sentiment']}")
    print(f"JSON: {row['structured_insights']}")
```

### Example Output

```json
{
  "summary": "The Federal Reserve cut interest rates by a quarter point for the third time this year, citing slowing inflation and a cooling labor market. Markets reacted positively with the S&P 500 rising 1.2%.",
  "keywords": "Federal Reserve, interest rate cut, inflation, labor market, Jerome Powell, S&P 500, monetary policy, housing market, economists, after-hours trading",
  "entities": {
    "people": ["Jerome Powell"],
    "organizations": ["Federal Reserve", "S&P 500"],
    "locations": []
  },
  "sentiment": "positive"
}
```

### When to Use

**Use this flow for:**

- Content categorization and tagging at scale
- Metadata extraction for search and indexing
- Sentiment monitoring across large content collections
- Entity extraction for knowledge graphs
- Quick structured analysis of unstructured text

**Don't use this flow for:**

- Training data generation — use [QA generation flows](available-flows.md#enhanced-multi-summary-qa-flows) instead
- Question-answer pair creation
- Knowledge tuning datasets
- Document augmentation

### Performance Characteristics

- **Fast** — Each extraction uses async mode for concurrent LLM calls across dataset rows
- **Efficient** — Lower token limits (256-1024) compared to QA flows (2048-4096)
- **Deterministic** — Low temperature settings produce consistent results
- **Scalable** — Designed for high-volume content processing

### Extending the Flow

The flow can be extended at runtime by adding custom blocks. For example, the [examples/text_analysis](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/tree/main/examples/text_analysis) directory demonstrates adding a stock ticker extraction block for financial content:

```python
from sdg_hub.core.blocks.llm import PromptBuilderBlock, LLMChatBlock
from sdg_hub.core.blocks.parsing import TextParserBlock

# Add a custom extraction step after loading the flow
ticker_prompt = PromptBuilderBlock(
    block_name="build_ticker_prompt",
    input_cols=["text"],
    output_cols=["ticker_prompt"],
    prompt_config_path="extract_stock_tickers.yaml",
)

ticker_gen = LLMChatBlock(
    block_name="generate_tickers",
    input_cols="ticker_prompt",
    output_cols="raw_tickers",
    max_tokens=256,
    temperature=0.1,
    async_mode=True,
)

# Chain the new blocks onto the flow
flow = flow.add_block(ticker_prompt)
flow = flow.add_block(ticker_gen)
```

See the `structured_insights_demo.ipynb` notebook for a complete walkthrough using the Bloomberg Financial News dataset (447k articles).
