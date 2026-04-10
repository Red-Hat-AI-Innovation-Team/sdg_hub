# Model Configurations

SDG Hub uses LiteLLM, supporting 100+ model providers. API keys can be passed directly or via environment variables.

## Provider Quick Reference

### OpenAI

```python
flow.set_model_config(model="openai/gpt-4o-mini", api_key="sk-...")
flow.set_model_config(model="openai/gpt-4-turbo", api_key="sk-...")
```

### Anthropic

```python
flow.set_model_config(model="anthropic/claude-sonnet-4-6", api_key="sk-ant-...")
```

### Azure OpenAI

```python
flow.set_model_config(
    model="azure/your-deployment-name",
    api_base="https://your-resource.openai.azure.com",
    api_key="your-azure-key",
    api_version="2024-02-15-preview"
)
```

### Local Models (vLLM)

```python
flow.set_model_config(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY"
)
```

### Ollama

```python
flow.set_model_config(
    model="ollama/llama3",
    api_base="http://localhost:11434/v1",
    api_key="ollama"
)
```

### Together AI

```python
flow.set_model_config(model="together_ai/meta-llama/Llama-3-70b-chat-hf", api_key="...")
```

### Groq

```python
flow.set_model_config(model="groq/llama3-70b-8192", api_key="...")
```

### Google Gemini

```python
flow.set_model_config(model="gemini/gemini-pro", api_key="...")
```

### AWS Bedrock

```python
flow.set_model_config(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    aws_access_key_id="...",
    aws_secret_access_key="...",
    aws_region_name="us-east-1"
)
```

---

## Common Parameters

```python
flow.set_model_config(
    model="...",
    api_key="...",

    # Generation
    temperature=0.7,        # 0.0-2.0, higher = more creative
    max_tokens=1024,        # Max output tokens
    top_p=1.0,              # Nucleus sampling
    top_k=40,               # Top-k sampling

    # Operational
    max_concurrency=10,     # Parallel requests
    timeout=120.0,          # Request timeout (seconds)
    num_retries=3,          # Retry on failure
)
```

## Per-Block Configuration

Apply different models to different blocks in the same flow:

```python
# Fast model for generation
flow.set_model_config(
    model="openai/gpt-4o-mini",
    api_key="sk-...",
    blocks=["generate_qa"]
)

# Stronger model for evaluation
flow.set_model_config(
    model="openai/gpt-4-turbo",
    api_key="sk-...",
    blocks=["evaluate_quality"]
)
```

## Environment Variables

LiteLLM reads API keys from environment automatically:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export TOGETHER_API_KEY="..."
```

```python
import os
flow.set_model_config(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY")
)
```

## Verifying Model Config

Quick test to confirm your model config works:

```python
from sdg_hub.core.blocks import LLMChatBlock
import pandas as pd

block = LLMChatBlock(
    block_name="test", input_cols="messages", output_cols="response",
    model="openai/gpt-4o-mini", api_key="sk-..."
)
result = block(pd.DataFrame({"messages": [[{"role": "user", "content": "Say hello"}]]}))
print(result["response"].iloc[0])
```

## Checking Flow Model Requirements

```python
flow = Flow.from_yaml("flow.yaml")

if flow.is_model_config_required():
    print(f"Default: {flow.get_default_model()}")
    recs = flow.get_model_recommendations()
    print(f"Compatible: {recs}")
```
