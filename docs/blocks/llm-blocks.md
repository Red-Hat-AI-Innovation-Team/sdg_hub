# LLM Blocks

LLM (Large Language Model) blocks provide AI-powered text generation capabilities. These blocks integrate with 100+ language model providers through LiteLLM, offering unified interfaces for chat, prompt building, and text parsing operations.

## 🧠 Available LLM Blocks

### LLMChatBlock
The core block for direct language model interaction, supporting chat completions across all major providers.

### PromptBuilderBlock  
Constructs structured prompts from templates and data, with support for complex formatting and validation.

### TextParserBlock
Extracts structured data from LLM responses using patterns, schemas, or parsing rules.

## 🚀 LLMChatBlock

The unified chat block that replaces provider-specific implementations with a single, powerful interface.

### Supported Providers

**Cloud Providers:**
- **OpenAI** - GPT-3.5, GPT-4, GPT-4o
- **Anthropic** - Claude 3 Haiku, Sonnet, Opus
- **Google** - Gemini Pro, Gemini Ultra, PaLM
- **Azure OpenAI** - All OpenAI models via Azure

**Local/Self-Hosted:**
- **vLLM** - High-performance local inference
- **Ollama** - Local model serving
- **HuggingFace** - Transformers integration
- **LM Studio** - Local GUI-based serving

### Basic Usage

```python
from sdg_hub.core.blocks import LLMChatBlock
from datasets import Dataset

# Configure for OpenAI
chat_block = LLMChatBlock(
    block_name="question_answerer",
    input_cols=["messages"],
    output_cols=["response"],
    model="openai/gpt-4o",
    api_key="your-openai-key",
    temperature=0.7,
    max_tokens=150
)

# Create dataset with messages
dataset = Dataset.from_dict({
    "messages": [
        [{"role": "user", "content": "What is machine learning?"}],
        [{"role": "user", "content": "Explain neural networks"}]
    ]
})

# Generate responses
result = chat_block.generate(dataset)
print(result["response"])  # Generated answers
```

### Provider-Specific Examples

#### Local vLLM Server
```python
chat_block = LLMChatBlock(
    block_name="local_llama",
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct", 
    api_base="http://localhost:8000/v1",
    api_key="your_key",
    input_cols=["messages"],
    output_cols=["response"]
)
```

#### Anthropic Claude
```python
chat_block = LLMChatBlock(
    block_name="claude_chat",
    model="anthropic/claude-3-sonnet-20240229",
    api_key="your-anthropic-key",
    input_cols=["messages"], 
    output_cols=["response"],
    max_tokens=1000
)
```

#### Google Gemini
```python
chat_block = LLMChatBlock(
    block_name="gemini_chat",
    model="google/gemini-pro",
    api_key="your-google-key",
    input_cols=["messages"],
    output_cols=["response"]
)
```

### Advanced Configuration

#### Multiple Completions
```python
# Generate 3 alternative responses per input
chat_block = LLMChatBlock(
    block_name="multi_response",
    model="openai/gpt-4o",
    n=3,  # Generate 3 completions
    input_cols=["messages"],
    output_cols=["responses"]  # Will contain list of 3 responses
)
```

#### Structured Output (JSON Mode)
```python
chat_block = LLMChatBlock(
    block_name="structured_chat",
    model="openai/gpt-4o",
    response_format={"type": "json_object"},
    input_cols=["messages"],
    output_cols=["json_response"]
)

# Ensure your prompt requests JSON format
dataset = Dataset.from_dict({
    "messages": [
        [{"role": "user", "content": "Return a JSON object with 'topic' and 'summary' fields about machine learning"}]
    ]
})
```

#### Async Processing & Concurrency Control
```python
chat_block = LLMChatBlock(
    block_name="async_chat",
    model="openai/gpt-4o",
    async_mode=True,  # Enable async processing
    input_cols=["messages"],
    output_cols=["response"]
)

# Automatically handles concurrent API calls for better throughput
result = chat_block.generate(large_dataset)
```

**Flow-Level Concurrency Control:**

When using LLM blocks within flows, you can control concurrency to prevent overwhelming API servers or hitting rate limits:

```python
from sdg_hub import Flow

# Load a flow with LLM blocks
flow = Flow.from_yaml("path/to/your/flow.yaml")
flow.set_model_config(model="openai/gpt-4o", api_key="your-key")

# Control concurrency for each LLM block in the flow
result = flow.generate(
    dataset, 
    max_concurrency=5  # Max 5 concurrent requests at any time
)
```

**Benefits of Concurrency Control:**
- **Rate Limit Management** - Prevent API throttling by limiting concurrent requests
- **Resource Control** - Manage memory and network usage for large datasets  
- **Provider-Friendly** - Respect API provider recommendations for concurrent requests
- **Automatic Scaling** - No concurrency limit = maximum parallelism for fastest processing

**How It Works:**

The unified async system automatically detects whether you're processing single or multiple messages and applies concurrency control appropriately:

```python
# Single message - processed immediately
single_message = [{"role": "user", "content": "Hello"}]

# Multiple messages - concurrency controlled via semaphore
batch_messages = [
    [{"role": "user", "content": "Question 1"}],
    [{"role": "user", "content": "Question 2"}],
    [{"role": "user", "content": "Question 3"}],
    # ... up to thousands of messages
]

# Both cases use the same unified API under the hood
# Concurrency is managed transparently
```

**Performance Guidelines:**
- **Small datasets (<100 samples)**: No concurrency limit needed
- **Medium datasets (100-1000 samples)**: `max_concurrency=10-20`
- **Large datasets (1000+ samples)**: `max_concurrency=5-10` (respect API limits)
- **Production workloads**: Start conservative and tune based on error rates

### Message Format

LLMChatBlock expects messages in OpenAI chat format:

```python
# Single conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "What are its benefits?"}
]

# Dataset with multiple conversations
dataset = Dataset.from_dict({
    "messages": [
        [{"role": "user", "content": "Question 1"}],
        [{"role": "user", "content": "Question 2"}],
        [
            {"role": "system", "content": "You are an expert."},
            {"role": "user", "content": "Complex question with context"}
        ]
    ]
})
```

## 🏗️ PromptBuilderBlock

Constructs prompts from templates and data with validation and formatting support.

### Basic Template Usage

#TODO: Add prompt builder block example

## 🔍 TextParserBlock

Extracts structured data from LLM responses using tag-based parsing or custom regex patterns. Essential for parsing LLM outputs into structured fields.

### Basic Tag-Based Parsing

Extract content between start and end tags:

```python
from sdg_hub.core.blocks import TextParserBlock
from datasets import Dataset

# Single field extraction
parser = TextParserBlock(
    block_name="extract_answer",
    input_cols=["llm_response"],
    output_cols=["answer"],
    start_tags=["<answer>"],
    end_tags=["</answer>"]
)

dataset = Dataset.from_dict({
    "llm_response": [
        "Question analysis: ...\n<answer>Machine learning is a subset of AI.</answer>",
        "Let me think...\n<answer>Neural networks process data in layers.</answer>"
    ]
})

result = parser.generate(dataset)
print(result["answer"])
# ['Machine learning is a subset of AI.', 'Neural networks process data in layers.']
```

### Multiple Field Extraction

Extract multiple structured fields from a single response:

```python
# Extract multiple fields with tag pairs
parser = TextParserBlock(
    block_name="extract_qa",
    input_cols=["llm_response"],
    output_cols=["question", "answer", "confidence"],
    start_tags=["<question>", "<answer>", "<confidence>"],
    end_tags=["</question>", "</answer>", "</confidence>"]
)

dataset = Dataset.from_dict({
    "llm_response": [
        """
        <question>What is Python?</question>
        <answer>Python is a high-level programming language.</answer>
        <confidence>0.95</confidence>
        """
    ]
})

result = parser.generate(dataset)
print(result["question"])     # ['What is Python?']
print(result["answer"])       # ['Python is a high-level programming language.']
print(result["confidence"])   # ['0.95']
```

### Custom Regex Parsing

Use regex patterns for flexible extraction:

```python
# Extract using regex pattern
parser = TextParserBlock(
    block_name="regex_parser",
    input_cols=["llm_response"],
    output_cols=["answer"],
    parsing_pattern=r"Answer:\s*(.+?)(?:\n|$)"
)

dataset = Dataset.from_dict({
    "llm_response": [
        "Question: What is AI?\nAnswer: Artificial Intelligence is...\n",
        "Let me answer:\nAnswer: Machine learning enables..."
    ]
})

result = parser.generate(dataset)
print(result["answer"])
# ['Artificial Intelligence is...', 'Machine learning enables...']
```

### Tag Cleanup

Remove unwanted tags from extracted content:

```python
# Clean up markdown and code tags
parser = TextParserBlock(
    block_name="clean_parser",
    input_cols=["llm_response"],
    output_cols=["clean_answer"],
    start_tags=["<answer>"],
    end_tags=["</answer>"],
    parser_cleanup_tags=["```", "###", "**"]
)

dataset = Dataset.from_dict({
    "llm_response": [
        "<answer>Here's the code: ```python\nprint('hello')```</answer>",
        "<answer>**Important**: This is the ### answer</answer>"
    ]
})

result = parser.generate(dataset)
print(result["clean_answer"])
# ['Here\'s the code: python\nprint(\'hello\')', 'Important: This is the  answer']
```

### Handling Multiple Matches

Extract all occurrences of a pattern:

```python
parser = TextParserBlock(
    block_name="multi_extract",
    input_cols=["llm_response"],
    output_cols=["keywords"],
    start_tags=["[KEY]"],
    end_tags=["[/KEY]"]
)

dataset = Dataset.from_dict({
    "llm_response": [
        "Important terms: [KEY]machine learning[/KEY], [KEY]neural networks[/KEY], [KEY]deep learning[/KEY]"
    ]
})

result = parser.generate(dataset)
print(result["keywords"])
# [['machine learning', 'neural networks', 'deep learning']]
```

### Practical Example: Evaluation Response Parsing

Common pattern for parsing LLM evaluation responses:

```python
# Parse structured evaluation output
evaluation_parser = TextParserBlock(
    block_name="parse_evaluation",
    input_cols=["evaluation_response"],
    output_cols=["explanation", "judgment"],
    start_tags=["[Start of Explanation]", "[Start of Answer]"],
    end_tags=["[End of Explanation]", "[End of Answer]"],
    parser_cleanup_tags=["```", "###"]
)

dataset = Dataset.from_dict({
    "evaluation_response": [
        """
        [Start of Explanation]
        The response accurately reflects the information in the document.
        No hallucinations or contradictions were found.
        [End of Explanation]

        [Start of Answer]
        YES
        [End of Answer]
        """
    ]
})

result = evaluation_parser.generate(dataset)
print(result["explanation"])  # ['The response accurately reflects...']
print(result["judgment"])     # ['YES']
```

### Integration with LLMChatBlock

TextParserBlock is commonly used after LLMChatBlock to structure responses:

```python
from sdg_hub.core.blocks import LLMChatBlock, LLMParserBlock, TextParserBlock

# Step 1: Generate LLM response
chat_block = LLMChatBlock(
    block_name="evaluator",
    model="openai/gpt-4o",
    input_cols=["messages"],
    output_cols=["eval_response"]
)

# Step 2: Extract content from response object
# Use field_prefix="" to get cleaner column names
llm_parser = LLMParserBlock(
    block_name="extract_eval",
    input_cols=["eval_response"],
    extract_content=True,
    field_prefix="eval_"  # Results in "eval_content" instead of "extract_content"
)

# Step 3: Parse structured fields from text
text_parser = TextParserBlock(
    block_name="parse_fields",
    input_cols=["eval_content"], 
    output_cols=["score", "reasoning"],
    start_tags=["[SCORE]", "[REASONING]"],
    end_tags=["[/SCORE]", "[/REASONING]"]
)

# Execute in sequence (or use a Flow)
dataset = Dataset.from_dict({
    "messages": [[{"role": "user", "content": "Evaluate this text..."}]]
})

result = chat_block.generate(dataset)
result = llm_parser.generate(result)
result = text_parser.generate(result)

print(result["score"])      # Extracted score
print(result["reasoning"])  # Extracted reasoning
```

### Configuration Reference

**Required Parameters:**
- `block_name` - Unique identifier for the block
- `input_cols` - Single column containing text to parse
- `output_cols` - List of field names for extracted content

**Parsing Methods (choose one):**
- **Tag-based**: `start_tags` + `end_tags` (must have same length as `output_cols`)
- **Regex**: `parsing_pattern` (single regex with capture groups)

**Optional Parameters:**
- `parser_cleanup_tags` - List of tags to remove from extracted text
- `expand_lists` - Whether to expand list inputs into rows (default: `True`)

**Tag Parsing Rules:**
- Number of tag pairs must match number of output columns
- Each tag pair extracts all matches for that field
- Tags can be any string (XML-style, markdown-style, custom)
- Missing tags result in empty lists for that field

## 🚀 Next Steps

- **[Transform Blocks](transform-blocks.md)** - Data manipulation and reshaping
- **[Filtering Blocks](filtering-blocks.md)** - Quality control and validation
- **[Flow Integration](../flows/overview.md)** - Combine LLM blocks into complete pipelines