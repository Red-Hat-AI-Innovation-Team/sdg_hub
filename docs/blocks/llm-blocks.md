# LLM Blocks

LLM blocks handle language model interaction through LiteLLM, supporting 100+ providers including OpenAI, Anthropic, Google, and local models such as vLLM and Ollama. This page covers three blocks: `LLMChatBlock` for chat completions, `PromptBuilderBlock` for Jinja2 template rendering into structured messages, and `LLMResponseExtractorBlock` for extracting fields from LLM response objects.

All blocks operate on pandas DataFrames. The model is set at runtime via `flow.set_model_config()` or directly in the constructor.

---

## LLMChatBlock

Unified LLM chat block supporting all providers via LiteLLM. Sends messages to a language model and returns raw response objects. Accepts any LiteLLM completion parameter (temperature, max_tokens, top_p, etc.) as extra keyword arguments.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `str \| list[str]` | required | Single input column containing the messages list |
| `output_cols` | `str \| list[str]` | required | Single output column for the response |
| `model` | `Optional[str]` | `None` | Model identifier in LiteLLM format (e.g. `"openai/gpt-4"`, `"anthropic/claude-3-sonnet-20240229"`) |
| `api_key` | `Optional[SecretStr]` | `None` | API key for the provider. Falls back to environment variables. Automatically redacted in logs. |
| `api_base` | `Optional[str]` | `None` | Base URL for the API. Required for local models. |
| `async_mode` | `bool` | `False` | Whether to use async processing |
| `timeout` | `float` | `120.0` | Request timeout in seconds |
| `num_retries` | `int` | `6` | Number of retry attempts using LiteLLM built-in retry |
| `drop_params` | `bool` | `True` | Whether to drop unsupported parameters to prevent API errors |
| `**kwargs` | `Any` | -- | Any LiteLLM completion parameter (temperature, max_tokens, top_p, response_format, seed, n, etc.) |

The `model`, `api_key`, `api_base`, `async_mode`, `timeout`, and `num_retries` fields are excluded from YAML serialization. They are set at runtime through `flow.set_model_config()` or the constructor.

Exactly one input column and one output column are required. The input column must contain lists of message dicts in OpenAI chat format. The output column receives lists of response message dicts.

### Message Format

LLMChatBlock expects messages in OpenAI chat completion format. Each message is a dict with `role` and `content` keys:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]
```

Each row in the input column must be a non-empty list of such dicts. The block validates this format before calling the model.

### Python Example

```python
from sdg_hub.core.blocks import LLMChatBlock
import pandas as pd

block = LLMChatBlock(
    block_name="qa_generator",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=1000,
)

dataset = pd.DataFrame({
    "messages": [
        [{"role": "user", "content": "What is machine learning?"}],
        [{"role": "user", "content": "Explain neural networks."}],
    ]
})

result = block.generate(dataset)
# result["response"] contains lists of response message dicts
```

### Provider Examples

**OpenAI:**

```python
from sdg_hub.core.blocks import LLMChatBlock

block = LLMChatBlock(
    block_name="openai_chat",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=1000,
)
```

**Local vLLM server:**

```python
from sdg_hub.core.blocks import LLMChatBlock

block = LLMChatBlock(
    block_name="local_llama",
    input_cols="messages",
    output_cols="response",
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="token-abc123",
    temperature=0.7,
)
```

**Anthropic Claude (via LiteLLM):**

```python
from sdg_hub.core.blocks import LLMChatBlock

block = LLMChatBlock(
    block_name="claude_chat",
    input_cols="messages",
    output_cols="response",
    model="anthropic/claude-3-sonnet-20240229",
    api_key="sk-ant-...",
    max_tokens=1000,
)
```

### Async Mode and Concurrency

Set `async_mode=True` to process all rows concurrently. When running inside a flow, pass `max_concurrency` to `flow.generate()` to limit concurrent requests:

```python
from sdg_hub.core.blocks import LLMChatBlock

block = LLMChatBlock(
    block_name="async_chat",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o",
    async_mode=True,
)

# Standalone usage runs all requests concurrently
result = block.generate(dataset)

# Within a flow, control concurrency via flow.generate()
from sdg_hub import Flow

flow = Flow.from_yaml("my_flow.yaml")
flow.set_model_config(model="openai/gpt-4o", api_key="sk-...")
result = flow.generate(dataset, max_concurrency=10)
```

When `n > 1` is set (multiple completions per request), the effective concurrency is automatically adjusted by dividing `max_concurrency` by `n`.

### Structured Output (JSON Mode)

```python
from sdg_hub.core.blocks import LLMChatBlock

block = LLMChatBlock(
    block_name="json_chat",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o",
    response_format={"type": "json_object"},
)
```

### YAML Example

```yaml
blocks:
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "qa_generator"
      input_cols: "messages"
      output_cols: "response"
      temperature: 0.7
      max_tokens: 1000
      drop_params: true
```

The `model`, `api_key`, and `api_base` fields are set at runtime via `flow.set_model_config()` and are not included in the YAML.

### Response Format

The output column contains a `list[dict]` for each row. Each dict is a response
message with at minimum a `content` key holding the assistant's text reply. When
`n=1` (the default), the list has a single element. When `n > 1`, the list
contains one dict per completion choice.

```python
# Single completion (n=1, the default):
[{"content": "Machine learning is a subset of AI..."}]

# Multiple completions (n=3):
[
    {"content": "Machine learning is..."},
    {"content": "ML refers to..."},
    {"content": "In computer science, machine learning..."},
]
```

Use `LLMResponseExtractorBlock` after `LLMChatBlock` to extract the `content`
(or other fields like `reasoning_content`, `tool_calls`) into flat columns.

---

## PromptBuilderBlock

Formats prompts into structured chat messages or plain text using Jinja2 templates. Takes input from dataset columns, applies templates from a YAML config file, and outputs either a list of message dicts or a concatenated string.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `str \| list[str] \| dict[str, str]` | required | Input columns. Use a dict to map dataset column names to template variable names. |
| `output_cols` | `str` | required | Single output column name for formatted content |
| `prompt_config_path` | `str` | required | Path to YAML file containing the Jinja2 template configuration |
| `format_as_messages` | `bool` | `True` | If True, output is a list of dicts with `role` and `content` keys. If False, output is a concatenated string with role prefixes. |

The YAML template file must be a list of message objects. Each message requires `role` (`system`, `user`, `assistant`, or `tool`) and `content` fields. At least one message must have `role: user`, and the final message must also have `role: user`.

Template variables use Jinja2 `{{ variable }}` syntax. Variables are resolved from dataset columns based on `input_cols`.

### Python Example

Template file (`qa_prompt.yaml`):

```yaml
- role: system
  content: "You are an expert {{ domain }} assistant."

- role: user
  content: |
    Context: {{ context }}
    Question: {{ question }}

    Provide a clear answer.
```

Block usage:

```python
from sdg_hub.core.blocks import PromptBuilderBlock
import pandas as pd

builder = PromptBuilderBlock(
    block_name="qa_prompter",
    input_cols=["domain", "context", "question"],
    output_cols="messages",
    prompt_config_path="qa_prompt.yaml",
    format_as_messages=True,
)

dataset = pd.DataFrame([{
    "domain": "physics",
    "context": "Newton's laws describe motion.",
    "question": "What is Newton's first law?",
}])

result = builder.generate(dataset)
# result["messages"][0] is a list of dicts:
# [
#   {"role": "system", "content": "You are an expert physics assistant."},
#   {"role": "user", "content": "Context: Newton's laws...\nQuestion: What is..."}
# ]
```

### Column Mapping with Dict

When dataset column names differ from template variable names, use a dict for `input_cols`:

```python
from sdg_hub.core.blocks import PromptBuilderBlock

builder = PromptBuilderBlock(
    block_name="mapped_prompter",
    input_cols={
        "article_text": "context",   # dataset column -> template variable
        "user_query": "question",
        "subject": "domain",
    },
    output_cols="messages",
    prompt_config_path="qa_prompt.yaml",
)
```

### Plain Text Output

Set `format_as_messages=False` to get concatenated text with role prefixes instead of structured message lists:

```python
from sdg_hub.core.blocks import PromptBuilderBlock

builder = PromptBuilderBlock(
    block_name="text_prompter",
    input_cols=["document", "response"],
    output_cols="formatted_prompt",
    prompt_config_path="eval_prompt.yaml",
    format_as_messages=False,
)

# Output is a string like:
# "system: You are an evaluator.\n\nuser: Document: ...\nResponse: ..."
```

### YAML Example

```yaml
blocks:
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "qa_prompter"
      input_cols:
        - "domain"
        - "context"
        - "question"
      output_cols: "messages"
      prompt_config_path: "prompts/qa_prompt.yaml"
      format_as_messages: true
```

---

## LLMResponseExtractorBlock

Extracts specified fields from LLM response objects (dicts or lists of dicts). Use this block after LLMChatBlock to pull out `content`, `reasoning_content`, or `tool_calls` from the raw response into separate columns.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `str \| list[str]` | required | Single input column containing response objects (dict or list of dicts) |
| `extract_content` | `bool` | `True` | Whether to extract the `content` field from responses |
| `extract_reasoning_content` | `bool` | `False` | Whether to extract the `reasoning_content` field from responses |
| `extract_tool_calls` | `bool` | `False` | Whether to extract the `tool_calls` field from responses |
| `expand_lists` | `bool` | `True` | Whether to expand list inputs into individual rows (True) or preserve lists (False) |
| `field_prefix` | `str` | `""` | Prefix for output field names. If empty, defaults to `"{block_name}_"`. Example: `"llm_"` produces columns `llm_content`, `llm_reasoning_content`. |

At least one of `extract_content`, `extract_reasoning_content`, or `extract_tool_calls` must be enabled.

The `output_cols` field is computed automatically from the extraction settings and prefix. You do not set it manually.

### Python Example

```python
from sdg_hub.core.blocks import LLMChatBlock, LLMResponseExtractorBlock
import pandas as pd

# After running LLMChatBlock, extract content from responses
extractor = LLMResponseExtractorBlock(
    block_name="extract_content",
    input_cols="response",
    extract_content=True,
    extract_reasoning_content=False,
    extract_tool_calls=False,
    field_prefix="llm_",
)

# Assuming 'result' has a "response" column from LLMChatBlock
extracted = extractor.generate(result)
# extracted["llm_content"] contains the text content from each response
```

### Extracting Multiple Fields

```python
from sdg_hub.core.blocks import LLMResponseExtractorBlock

extractor = LLMResponseExtractorBlock(
    block_name="full_extract",
    input_cols="response",
    extract_content=True,
    extract_reasoning_content=True,
    extract_tool_calls=True,
    field_prefix="llm_",
)

# Produces columns: llm_content, llm_reasoning_content, llm_tool_calls
```

### Handling Multiple Completions

When `LLMChatBlock` is configured with `n > 1`, the response column contains a list of response dicts. `expand_lists=True` (default) creates a separate row for each completion. Set `expand_lists=False` to keep lists as column values.

```python
from sdg_hub.core.blocks import LLMResponseExtractorBlock

# Expand each completion into its own row
extractor = LLMResponseExtractorBlock(
    block_name="expand_responses",
    input_cols="response",
    extract_content=True,
    expand_lists=True,
    field_prefix="gen_",
)

# Or preserve list structure
extractor_flat = LLMResponseExtractorBlock(
    block_name="keep_lists",
    input_cols="response",
    extract_content=True,
    expand_lists=False,
    field_prefix="gen_",
)
```

### YAML Example

```yaml
blocks:
  - block_type: "LLMResponseExtractorBlock"
    block_config:
      block_name: "extract_content"
      input_cols: "response"
      extract_content: true
      extract_reasoning_content: false
      extract_tool_calls: false
      expand_lists: true
      field_prefix: "llm_"
```

---

## Common Pipeline Pattern

A typical LLM pipeline chains these three blocks: build prompts, call the model, and extract the text content.

```python
from sdg_hub.core.blocks import (
    LLMChatBlock,
    LLMResponseExtractorBlock,
    PromptBuilderBlock,
)
import pandas as pd

# Step 1: Build messages from dataset columns using a template
prompt_builder = PromptBuilderBlock(
    block_name="build_prompt",
    input_cols=["context", "question"],
    output_cols="messages",
    prompt_config_path="prompts/qa.yaml",
)

# Step 2: Send messages to the LLM
chat = LLMChatBlock(
    block_name="call_llm",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o",
    api_key="sk-...",
    temperature=0.7,
)

# Step 3: Extract the text content from the response
extractor = LLMResponseExtractorBlock(
    block_name="extract",
    input_cols="response",
    extract_content=True,
    field_prefix="llm_",
)

# Run the pipeline
dataset = pd.DataFrame([{
    "context": "Python is a programming language.",
    "question": "What is Python used for?",
}])

result = prompt_builder.generate(dataset)
result = chat.generate(result)
result = extractor.generate(result)
# result["llm_content"] contains the extracted answer text
```

This same pipeline can be defined in YAML and executed as a flow. See the [Flow documentation](../flows/index.md) for details.

---

## Next Steps

- [Parsing Blocks](parsing-blocks.md) -- extract structured data from LLM text output
- [Transform Blocks](transform-blocks.md) -- data manipulation and column operations
- [Filtering Blocks](filtering-blocks.md) -- quality control and row filtering
