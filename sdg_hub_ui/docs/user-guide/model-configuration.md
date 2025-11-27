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

### Temperature

Controls randomness in generation:

| Value | Behavior |
|-------|----------|
| 0.0 | Deterministic, same output each time |
| 0.3-0.5 | Low creativity, focused responses |
| 0.7 | Balanced (default) |
| 1.0 | High creativity, varied outputs |

**Recommendations:**

- Q&A generation: 0.3-0.5
- Creative content: 0.7-0.9
- Factual extraction: 0.0-0.2

### Max Tokens

Maximum length of generated response:

| Use Case | Recommended |
|----------|-------------|
| Short answers | 256-512 |
| Paragraphs | 1024-2048 |
| Long documents | 4096+ |

**Note:** Higher values increase cost and latency.

### Top P (Nucleus Sampling)

Alternative to temperature for controlling diversity:

| Value | Behavior |
|-------|----------|
| 0.1 | Very focused, top 10% probability mass |
| 0.5 | Moderate diversity |
| 0.95 | High diversity (default) |
| 1.0 | Consider all tokens |

**Tip:** Usually set one of temperature OR top_p, not both.

### Additional Parameters

Any other model-specific parameters as JSON:

```json
{
  "frequency_penalty": 0.5,
  "presence_penalty": 0.3,
  "stop": ["\n\n", "END"]
}
```

## Provider-Specific Setup

### Local vLLM

1. **Start vLLM server:**

   ```bash
   vllm serve meta-llama/Llama-3.3-70B-Instruct \
     --port 8000 \
     --tensor-parallel-size 4
   ```

2. **Configure in UI:**

   ```yaml
   Model: hosted_vllm/meta-llama/Llama-3.3-70B-Instruct
   API Base: http://localhost:8000/v1
   API Key: EMPTY
   ```

### OpenAI

1. **Get API key from** [platform.openai.com](https://platform.openai.com/api-keys)

2. **Set environment variable:**

   ```bash
   export OPENAI_API_KEY="your-openai-key-here"
   ```

3. **Configure in UI:**

   ```yaml
   Model: openai/gpt-4o
   API Base: (leave empty)
   API Key: env:OPENAI_API_KEY
   ```

### Anthropic

1. **Get API key from** [console.anthropic.com](https://console.anthropic.com/)

2. **Set environment variable:**

   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-key-here"
   ```

3. **Configure in UI:**

   ```yaml
   Model: anthropic/claude-3-5-sonnet-20241022
   API Base: (leave empty)
   API Key: env:ANTHROPIC_API_KEY
   ```

### Azure OpenAI

1. **Create deployment in Azure Portal**

2. **Configure in UI:**

   ```yaml
   Model: azure/your-deployment-name
   API Base: https://your-resource.openai.azure.com/
   API Key: env:AZURE_OPENAI_API_KEY
   ```

## Model Recommendations

Flow YAML files can include recommended models:

```yaml
# In flow.yaml
metadata:
  model_recommendations:
    default: meta-llama/Llama-3.3-70B-Instruct
    alternatives:
      - gpt-4o
      - claude-3-opus
    notes: "Best results with 70B+ parameter models"
```

The UI displays these recommendations when selecting a flow.

## Troubleshooting

### Connection Errors

**"Connection refused"**

- Verify API base URL is correct
- Check model server is running
- Confirm firewall allows connection

**"401 Unauthorized"**

- Verify API key is correct
- Check environment variable is set
- Ensure key has required permissions

### Model Errors

**"Model not found"**

- Verify model name spelling
- Check model is deployed/available
- Confirm provider prefix is correct

**"Context length exceeded"**

- Reduce input size
- Increase `max_tokens` for output
- Use a model with larger context window

### Performance Issues

**Slow responses:**

- Reduce `max_tokens`
- Lower `max_concurrency` setting
- Check model server resources

**High costs:**

- Use dry runs first
- Limit `num_samples`
- Consider smaller models for testing

## Cost Estimation

For paid APIs, estimate costs before large runs:

```
Cost = (Input Tokens + Output Tokens) × Price per 1K Tokens

Example (GPT-4o):
- 1000 samples × 500 input tokens = 500K input tokens
- 1000 samples × 200 output tokens = 200K output tokens
- Input: 500K × $0.005/1K = $2.50
- Output: 200K × $0.015/1K = $3.00
- Total: ~$5.50
```

**Tip:** Always run a dry run to estimate actual token usage.

## Next Steps

- [Dataset Configuration](dataset-configuration.md) — Set up your data
- [Running Generation](generation.md) — Execute with your model
- [Flow Builder](flow-builder.md) — Create custom flows

