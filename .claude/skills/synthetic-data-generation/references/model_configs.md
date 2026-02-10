# Model Configurations

SDG Hub uses LiteLLM, supporting 100+ model providers.

## OpenAI

```python
flow.set_model_config(
    model="openai/gpt-4o-mini",
    api_key="sk-..."
)

# Or specific versions
flow.set_model_config(
    model="openai/gpt-4-turbo",
    api_key="sk-..."
)

flow.set_model_config(
    model="openai/gpt-3.5-turbo",
    api_key="sk-..."
)
```

## Anthropic

```python
flow.set_model_config(
    model="anthropic/claude-3-opus-20240229",
    api_key="sk-ant-..."
)

flow.set_model_config(
    model="anthropic/claude-3-sonnet-20240229",
    api_key="sk-ant-..."
)
```

## Azure OpenAI

```python
flow.set_model_config(
    model="azure/your-deployment-name",
    api_base="https://your-resource.openai.azure.com",
    api_key="your-azure-key",
    api_version="2024-02-15-preview"
)
```

## Local Models (vLLM)

```python
# vLLM server running locally
flow.set_model_config(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY"  # vLLM doesn't need key
)

# With custom parameters
flow.set_model_config(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    temperature=0.7,
    max_tokens=2048
)
```

## Ollama

```python
flow.set_model_config(
    model="ollama/llama3",
    api_base="http://localhost:11434/v1",
    api_key="ollama"
)

flow.set_model_config(
    model="ollama/mistral",
    api_base="http://localhost:11434/v1",
    api_key="ollama"
)
```

## Together AI

```python
flow.set_model_config(
    model="together_ai/meta-llama/Llama-3-70b-chat-hf",
    api_key="your-together-key"
)
```

## Groq

```python
flow.set_model_config(
    model="groq/llama3-70b-8192",
    api_key="your-groq-key"
)
```

## Google (Gemini)

```python
flow.set_model_config(
    model="gemini/gemini-pro",
    api_key="your-google-key"
)
```

## AWS Bedrock

```python
flow.set_model_config(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    aws_access_key_id="...",
    aws_secret_access_key="...",
    aws_region_name="us-east-1"
)
```

## Common Parameters

```python
flow.set_model_config(
    model="...",
    api_key="...",

    # Generation parameters
    temperature=0.7,          # 0.0-2.0, higher = more random
    max_tokens=1024,          # Max output tokens
    top_p=1.0,                # Nucleus sampling
    top_k=40,                 # Top-k sampling

    # Concurrency
    max_concurrency=10,       # Max parallel requests
    timeout=120.0,            # Request timeout (seconds)
    num_retries=3,            # Retry failed requests
)
```

## Configuring Specific Blocks

```python
# Apply to all LLM blocks
flow.set_model_config(
    model="openai/gpt-4o-mini",
    api_key="sk-..."
)

# Apply to specific blocks only
flow.set_model_config(
    model="openai/gpt-3.5-turbo",
    api_key="sk-...",
    blocks=["generate_qa"]  # Only this block
)

flow.set_model_config(
    model="openai/gpt-4o-mini",
    api_key="sk-...",
    blocks=["evaluate_quality"]  # Different model for evaluation
)
```

## Direct Block Usage

```python
from sdg_hub.core.blocks import LLMChatBlock

block = LLMChatBlock(
    block_name="gen",
    input_cols="messages",
    output_cols="response",

    # Model config directly
    model="openai/gpt-4o-mini",
    api_key="sk-...",

    # Generation params
    temperature=0.7,
    max_tokens=1024,

    # Async processing
    async_mode=True,
    max_concurrency=10
)
```

## Environment Variables

LiteLLM reads API keys from environment:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export TOGETHER_API_KEY="..."
```

Then in code:

```python
import os

flow.set_model_config(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY")
)
```

## Testing Model Config

```python
# play.py
from sdg_hub.core.blocks import LLMChatBlock
import pandas as pd

# Quick test
block = LLMChatBlock(
    block_name="test",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o-mini",
    api_key="sk-..."
)

df = pd.DataFrame({
    "messages": [[{"role": "user", "content": "Say hello"}]]
})

result = block(df)
print(result["response"].iloc[0])
# Should print a greeting
```

## Checking Model Requirements

```python
from sdg_hub import Flow

flow = Flow.from_yaml("flow.yaml")

# Check if model config needed
if flow.is_model_config_required():
    print("Need to configure model")
    print(f"Default: {flow.get_default_model()}")
    print(f"Compatible: {flow.metadata.recommended_models.compatible}")
```
