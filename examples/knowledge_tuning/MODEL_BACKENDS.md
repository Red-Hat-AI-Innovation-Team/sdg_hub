# Model Backend Configuration Guide

This guide explains how to configure SDG Hub to work with different model backends and providers. SDG Hub supports various model providers through OpenAI-compatible APIs.

## Supported Backends

### 1. OpenAI (Cloud)

**Setup:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="your-openai-api-key",  # Set OPENAI_API_KEY environment variable
    base_url="https://api.openai.com/v1",  # Default, can be omitted
)

# Available models: gpt-4, gpt-3.5-turbo, etc.
model_id = "gpt-4"
```

**Configuration:**
```yaml
model_id: gpt-4  # Or gpt-3.5-turbo, gpt-4-turbo, etc.
```

### 2. Local vLLM Server

**Setup vLLM Server:**
```bash
# Start vLLM server (example with Llama model)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --dtype float16
```

**Client Configuration:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",  # vLLM doesn't require real API key
    base_url="http://localhost:8000/v1",
)

# Check available models
models = client.models.list()
model_id = models.data[0].id  # Use the model served by vLLM
```

**Configuration:**
```yaml
# Use the exact model ID returned by vLLM (may include leading slash)
model_id: /model/meta-llama/Llama-3.1-8B-Instruct
```

**Common vLLM Model Examples:**
- Llama: `meta-llama/Llama-3.1-8B-Instruct`
- Mixtral: `mistralai/Mixtral-8x7B-Instruct-v0.1`
- Qwen: `Qwen/Qwen2.5-7B-Instruct`

### 3. Ollama

**Setup Ollama:**
```bash
# Install and start Ollama
ollama serve

# Pull a model
ollama pull llama3.1
```

**Client Configuration:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:11434/v1",  # Ollama default port
)

model_id = "llama3.1"  # Model name from ollama list
```

### 4. Azure OpenAI

**Client Configuration:**
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="your-azure-api-key",
    api_version="2024-02-01",
    azure_endpoint="https://your-resource.openai.azure.com/",
)

model_id = "your-deployment-name"  # Azure deployment name, not model name
```

### 5. Anthropic Claude (via LiteLLM)

**Setup LiteLLM Proxy:**
```bash
pip install litellm
litellm --model claude-3-sonnet-20240229
```

**Client Configuration:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="your-anthropic-api-key",
    base_url="http://localhost:8000",  # LiteLLM proxy
)

model_id = "claude-3-sonnet-20240229"
```

### 6. Google Vertex AI (via LiteLLM)

**Client Configuration:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="your-vertex-ai-key",
    base_url="http://localhost:8000",  # LiteLLM proxy
)

model_id = "vertex_ai/gemini-pro"
```

## Context Length Considerations

Different models have different context length limits. Adjust your configuration accordingly:

| Model | Context Length | Recommended max_tokens |
|-------|----------------|------------------------|
| GPT-3.5-turbo | 16,385 | 4,000 |
| GPT-4 | 128,000 | 4,000 |
| Llama-3.1-8B | 4,096 | 1,024 |
| Mixtral-8x7B | 32,768 | 4,000 |
| Claude-3-Sonnet | 200,000 | 4,000 |

**For models with limited context (< 8K tokens):**
1. Use the ChunkingBlock to split documents
2. Reduce max_tokens in your YAML config
3. Use shorter prompts

## Configuration Examples

### High-Resource Setup (GPT-4, Claude)
```yaml
gen_kwargs:
  max_tokens: 4000
  temperature: 0.0
```

### Low-Resource Setup (Local 7B-8B models)
```yaml
# Add chunking at the beginning
- block_type: ChunkingBlock
  block_config:
    block_name: chunk_documents
    input_col: document
    output_col: document
    chunk_size: 1000
    overlap: 100

# Reduce token limits
gen_kwargs:
  max_tokens: 1024
  temperature: 0.0
```

## Environment Variables

Set these environment variables for easier configuration:

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# Local endpoints
export LOCAL_LLM_BASE_URL="http://localhost:8000/v1"
export LOCAL_LLM_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
```

## Troubleshooting

### Common Issues

**1. Model Not Found (404 Error)**
- Check the exact model ID with `client.models.list()`
- vLLM models may have leading slashes in their IDs
- Ensure the model is properly loaded/deployed

**2. Context Length Exceeded (400 Error)**
- Add ChunkingBlock to your flow configuration
- Reduce max_tokens in gen_kwargs
- Use shorter input documents

**3. Connection Refused**
- Verify the server is running on the correct port
- Check firewall settings
- For local setups, ensure the service is bound to the correct interface

**4. Authentication Errors**
- Verify API keys are correct and have proper permissions
- Check API quota and rate limits
- Ensure environment variables are properly set

### Performance Optimization

**For Local Models:**
- Use appropriate tensor parallelism for multi-GPU setups
- Consider quantization for memory-constrained environments
- Adjust batch sizes based on available GPU memory

**For Cloud APIs:**
- Implement proper rate limiting
- Use async processing for large datasets
- Monitor API costs and usage

## Testing Your Setup

Use this code snippet to test your configuration:

```python
from openai import OpenAI

# Replace with your configuration
client = OpenAI(
    api_key="your-api-key-or-EMPTY",
    base_url="your-base-url",
)

try:
    # List available models
    models = client.models.list()
    print("Available models:")
    for model in models.data:
        print(f"  - {model.id}")
    
    # Test a simple completion
    if models.data:
        model_id = models.data[0].id
        response = client.completions.create(
            model=model_id,
            prompt="Hello, world!",
            max_tokens=10
        )
        print(f"✅ Connection successful!")
        print(f"Response: {response.choices[0].text}")
    else:
        print("❌ No models available")
        
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

## Contributing

When adding support for new backends:

1. Test the OpenAI-compatible API endpoints
2. Document any specific configuration requirements
3. Add example configurations to this guide
4. Update the troubleshooting section with common issues

For questions or issues with specific backends, please check the respective provider's documentation or open an issue in the SDG Hub repository.