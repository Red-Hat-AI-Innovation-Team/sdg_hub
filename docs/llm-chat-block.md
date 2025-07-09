# LLMChatBlock Documentation

The `LLMChatBlock` is the primary interface for interacting with language models in SDG Hub. It provides a unified way to call 100+ LLM providers through a single, consistent API.

## Quick Start

```yaml
- block_type: LLMChatBlock
  block_config:
    block_name: my_llm_block
    input_cols: messages
    output_cols: response
    model: openai/gpt-4
    temperature: 0.7
    max_tokens: 1000
```

## Overview

`LLMChatBlock` replaces all provider-specific LLM blocks (like OpenAIChatBlock) with a single, unified interface that supports:

- **100+ Providers**: OpenAI, Anthropic, Google, local models (vLLM, Ollama), and more
- **Async Processing**: Optional asynchronous mode for better performance
- **Robust Error Handling**: Automatic retries with exponential backoff
- **Runtime Overrides**: Change parameters at generation time
- **Comprehensive Logging**: Detailed monitoring and debugging information

## Supported Providers

The block supports all providers available through [LiteLLM](https://litellm.vercel.app/docs/providers), including:

### Cloud Providers
- **OpenAI**: GPT-4, GPT-3.5, GPT-4 Turbo, etc.
- **Anthropic**: Claude 3 (Sonnet, Haiku, Opus)
- **Google**: Gemini Pro, PaLM 2
- **Azure OpenAI**: All OpenAI models via Azure
- **AWS Bedrock**: Claude, Titan, Jurassic models
- **Cohere**: Command models
- **Mistral AI**: Mistral models
- **Together AI**: Various open-source models

### Local Deployments
- **vLLM**: Self-hosted models
- **Ollama**: Local model serving
- **OpenAI-compatible APIs**: Any OpenAI-compatible endpoint

## Configuration

### Required Parameters

```yaml
block_config:
  block_name: str           # Unique name for this block
  input_cols: str          # Column containing message lists
  output_cols: str         # Column to store responses
  model: str               # Model in "provider/model" format
```

### Model Format

All models use the format `provider/model-name`:

```yaml
# OpenAI models
model: openai/gpt-4
model: openai/gpt-3.5-turbo
model: openai/gpt-4-turbo

# Anthropic models
model: anthropic/claude-3-sonnet-20240229
model: anthropic/claude-3-haiku-20240307
model: anthropic/claude-3-opus-20240229

# Google models
model: google/gemini-pro
model: google/gemini-pro-vision

# Local vLLM deployment
model: hosted_vllm/meta-llama/Llama-2-7b-chat-hf

# Ollama local model
model: ollama/llama2
```

### Authentication

API keys are automatically resolved from environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="..."

# And many more...
```

You can also specify the API key directly (not recommended for production):

```yaml
block_config:
  api_key: "your-api-key-here"  # Use environment variables instead
```

### Optional Parameters

```yaml
block_config:
  # API Configuration
  api_base: str            # Custom API endpoint (required for local models)
  timeout: float           # Request timeout in seconds (default: 120.0)
  max_retries: int         # Maximum retry attempts (default: 6)
  async_mode: bool         # Enable async processing (default: false)
  
  # Generation Parameters
  temperature: float       # Sampling temperature 0.0-2.0
  max_tokens: int         # Maximum tokens to generate
  top_p: float            # Nucleus sampling 0.0-1.0
  frequency_penalty: float # Frequency penalty -2.0-2.0
  presence_penalty: float # Presence penalty -2.0-2.0
  stop: str|list          # Stop sequences
  seed: int               # Random seed for reproducibility
  response_format: dict   # Response format (e.g., {"type": "json_object"})
  n: int                  # Number of completions to generate
```

## Usage Examples

### Basic OpenAI GPT-4

```yaml
- block_type: LLMChatBlock
  block_config:
    block_name: gpt4_chat
    input_cols: messages
    output_cols: response
    model: openai/gpt-4
    temperature: 0.7
    max_tokens: 1000
```

### Anthropic Claude with Custom Settings

```yaml
- block_type: LLMChatBlock
  block_config:
    block_name: claude_chat
    input_cols: messages
    output_cols: response
    model: anthropic/claude-3-sonnet-20240229
    temperature: 0.3
    max_tokens: 2000
    top_p: 0.95
```

### Local vLLM Model

```yaml
- block_type: LLMChatBlock
  block_config:
    block_name: local_llama
    input_cols: messages
    output_cols: response
    model: hosted_vllm/meta-llama/Llama-2-7b-chat-hf
    api_base: http://localhost:8000/v1
    temperature: 0.7
```

### Async Processing for Large Batches

```yaml
- block_type: LLMChatBlock
  block_config:
    block_name: async_chat
    input_cols: messages
    output_cols: response
    model: openai/gpt-3.5-turbo
    async_mode: true
    temperature: 0.7
```

### JSON Mode Response

```yaml
- block_type: LLMChatBlock
  block_config:
    block_name: json_chat
    input_cols: messages
    output_cols: response
    model: openai/gpt-4
    response_format:
      type: json_object
    temperature: 0.1
```

## Input Format

The input column must contain a list of messages in OpenAI chat format:

```python
# Example dataset
{
  "messages": [
    [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    [
      {"role": "user", "content": "Write a haiku about programming."}
    ]
  ]
}
```

### Message Roles

- `system`: System instructions that guide the model's behavior
- `user`: User messages/questions
- `assistant`: Previous assistant responses (for conversation context)

## Runtime Parameter Overrides

You can override any generation parameter at runtime using `gen_kwargs`:

```yaml
- block_type: LLMChatBlock
  block_config:
    block_name: flexible_chat
    input_cols: messages
    output_cols: response
    model: openai/gpt-4
    temperature: 0.7  # Default temperature
  gen_kwargs:
    temperature: 0.9  # Override temperature for this run
    max_tokens: 500   # Override max tokens
    top_p: 0.95      # Add additional parameter
```

## Complete Flow Example

```yaml
# flows/chat_example.yaml
- block_type: LLMChatBlock
  block_config:
    block_name: question_answerer
    input_cols: messages
    output_cols: answer
    model: openai/gpt-4
    temperature: 0.3
    max_tokens: 1000
  gen_kwargs:
    temperature: 0.7
    max_tokens: 500
```

## Error Handling

The block includes robust error handling with automatic retries:

### Retryable Errors
- Rate limiting (with longer delays)
- Network timeouts
- Server errors (5xx)
- Temporary connection issues

### Non-Retryable Errors
- Authentication failures
- Invalid model names
- Context length exceeded
- Malformed requests

### Custom Retry Configuration

```yaml
block_config:
  max_retries: 3        # Reduce retries for faster failure
  timeout: 60.0         # Shorter timeout
```

## Performance Optimization

### Async Mode

Enable async processing for better performance with large batches:

```yaml
block_config:
  async_mode: true
```

### Provider Selection

Choose providers based on your needs:

- **OpenAI**: High quality, good for complex tasks
- **Anthropic**: Excellent for safety and reasoning
- **Google**: Good for multimodal tasks
- **Local models**: Cost-effective for large volumes

## Monitoring and Debugging

### Enable Debug Logging

```python
import logging
logging.getLogger("sdg_hub.blocks.llm").setLevel(logging.DEBUG)
```

### Monitor Model Usage

The block provides detailed logging including:
- Request parameters
- Response metadata
- Error details and retry attempts
- Performance metrics

## Python API Usage

```python
from sdg_hub.blocks.llm import LLMChatBlock
from datasets import Dataset

# Create block
block = LLMChatBlock(
    block_name="my_chat_block",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4",
    temperature=0.7
)

# Prepare dataset
dataset = Dataset.from_dict({
    "messages": [
        [{"role": "user", "content": "Hello!"}],
        [{"role": "user", "content": "How are you?"}]
    ]
})

# Generate responses
result = block.generate(dataset)
print(result["response"])

# With runtime overrides
result = block.generate(dataset, temperature=0.9, max_tokens=100)
```

## Best Practices

1. **Environment Variables**: Always use environment variables for API keys
2. **Model Selection**: Choose appropriate models for your use case and budget
3. **Error Handling**: Configure appropriate retry settings for your reliability needs
4. **Async Processing**: Use async mode for better performance with large datasets
5. **Parameter Tuning**: Test different temperature and other parameters for optimal results
6. **Monitoring**: Enable logging to monitor usage and debug issues
7. **Local Development**: Use local models for development and testing

## Migration from Legacy Blocks

If you're using older LLM blocks, here's how to migrate:

### From OpenAIChatBlock

**Old:**
```yaml
- block_type: OpenAIChatBlock
  block_config:
    block_name: openai_chat
    input_cols: messages
    output_cols: response
    model_id: gpt-4
```

**New:**
```yaml
- block_type: LLMChatBlock
  block_config:
    block_name: openai_chat
    input_cols: messages
    output_cols: response
    model: openai/gpt-4  # Note: provider prefix required
```

### From AnthropicChatBlock

**Old:**
```yaml
- block_type: AnthropicChatBlock
  block_config:
    block_name: claude_chat
    input_cols: messages
    output_cols: response
    model_id: claude-3-sonnet-20240229
```

**New:**
```yaml
- block_type: LLMChatBlock
  block_config:
    block_name: claude_chat
    input_cols: messages
    output_cols: response
    model: anthropic/claude-3-sonnet-20240229
```

## Troubleshooting

### Common Issues

**"Model not found" errors:**
- Check the model format: `provider/model-name`
- Verify the model is available in your account

**Authentication errors:**
- Ensure API keys are set in environment variables
- Check the correct environment variable name for your provider

**Connection errors for local models:**
- Verify the `api_base` URL is correct
- Ensure the local model server is running

**Rate limiting:**
- Reduce request frequency or increase retry delays
- Consider using different models or providers

**Context length exceeded:**
- Reduce input message length
- Use models with larger context windows

### Getting Help

Enable debug logging to see detailed information:

```python
import logging
logging.getLogger("sdg_hub.blocks.llm").setLevel(logging.DEBUG)
```

This will show request/response details, error context, and retry attempts.

## Related Documentation

- [Prompt Builder Block](prompt-builder-block.md) - For building structured prompts
- [Text Parser Block](text-parser-block.md) - For parsing LLM responses
- [Integration Guide](llm-integration.md) - Using all LLM blocks together