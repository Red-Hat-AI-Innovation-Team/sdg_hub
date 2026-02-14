# User Guide: Model Configuration

This guide covers configuring LLM models for data generation.

## Supported Providers

SDG Hub UI supports any OpenAI-compatible API:

| Provider | Model Format | API Base Required |
|----------|--------------|-------------------|
| **vLLM** (local) | `hosted_vllm/model_name` | Yes |
| **OpenAI** | `openai/gpt-4o` | No |
| **Anthropic** | `anthropic/claude-3-opus` | No |
| **Azure OpenAI** | `azure/deployment-name` | Yes |
| **Together AI** | `together/model_name` | Yes |
| **Anyscale** | `anyscale/model_name` | Yes |

## Configuration Fields

### Model Name

The full model identifier including provider prefix:

```
provider/model_identifier

Examples:
hosted_vllm/meta-llama/Llama-3.3-70B-Instruct
openai/gpt-4o
anthropic/claude-3-5-sonnet-20241022
```

### API Base URL

The endpoint for your model server:

| Provider | API Base |
|----------|----------|
| Local vLLM | `http://localhost:8000/v1` |
| OpenAI | (leave empty) |
| Anthropic | (leave empty) |
| Azure | `https://your-resource.openai.azure.com/` |
| Together AI | `https://api.together.xyz/v1` |

### API Key

Authentication for the model provider:

| Method | Format | Example |
|--------|--------|---------|
| **Direct** | Raw key | `your-api-key-here` |
| **Environment** | `env:VAR_NAME` | `env:OPENAI_API_KEY` |
| **Empty** | `EMPTY` | For local vLLM without auth |

#### Using Environment Variables

You can reference environment variables instead of entering keys directly:

```bash
# Set in your shell
export OPENAI_API_KEY="your-openai-key-here"
```

Then in the UI, enter: `env:OPENAI_API_KEY`

## Advanced Parameters

Click "Show Advanced Parameters" to access:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.7 | Controls randomness (0 = deterministic, 1 = creative) |
| `max_tokens` | 2048 | Maximum response length in tokens |
| `top_p` | 0.95 | Nucleus sampling threshold |

Any other model-specific parameters can be passed as JSON:

```json
{
  "frequency_penalty": 0.5,
  "presence_penalty": 0.3,
  "stop": ["\n\n", "END"]
}
```

## Provider Setup Examples

### Local vLLM

```yaml
Model:    hosted_vllm/meta-llama/Llama-3.3-70B-Instruct
API Base: http://localhost:8000/v1
API Key:  EMPTY
```

### OpenAI

```yaml
Model:    openai/gpt-4o
API Base: (leave empty)
API Key:  env:OPENAI_API_KEY
```

### Anthropic

```yaml
Model:    anthropic/claude-3-5-sonnet-20241022
API Base: (leave empty)
API Key:  env:ANTHROPIC_API_KEY
```

### Azure OpenAI

```yaml
Model:    azure/your-deployment-name
API Base: https://your-resource.openai.azure.com/
API Key:  env:AZURE_OPENAI_API_KEY
```

## Test Connection

After filling in the fields, click **Test Connection** to verify the configuration. The UI sends a test prompt and reports whether the connection succeeded, along with response latency.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| Connection refused | Model server not running | Start the server and verify the API base URL |
| 401 Unauthorized | Invalid API key | Check the key or environment variable is set correctly |
| Model not found | Wrong model name | Verify model name spelling and provider prefix |
| Context length exceeded | Input too large | Reduce input size or use a model with a larger context window |

## Next Steps

- [Dataset Configuration](dataset-configuration.md) — Set up your data
- [Running Generation](generation.md) — Execute with your model
- [Flow Builder](flow-builder.md) — Create custom flows
