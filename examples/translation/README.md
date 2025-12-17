# Translation Examples

This directory contains examples demonstrating the use of the **TranslationBlock** for multilingual synthetic data generation.

## Overview

The **TranslationBlock** is an LLM-powered block that translates prompt scaffolding from one language to another while intelligently preserving content that is already in the target language. This is particularly useful for multilingual knowledge tuning workflows where:

1. **Prompt templates** are in English (e.g., instructions, formatting rules)
2. **Source documents and ICL examples** are already translated to the target language
3. You want to translate only the scaffolding without touching the source content

## How It Works

The TranslationBlock uses an LLM with a carefully crafted prompt to:

1. **Detect language boundaries** - Identifies which parts of the text are in the source language vs. target language
2. **Translate selectively** - Translates only the source language text (e.g., English instructions)
3. **Preserve structure** - Maintains all formatting, tags, and structure exactly
4. **Protect target language content** - Leaves already-translated content completely unchanged

This LLM-based approach is simple, flexible, and handles edge cases automatically without requiring hardcoded patterns.

## Example: Spanish Knowledge Tuning

See [spanish_knowledge_tuning_flow_example.yaml](./spanish_knowledge_tuning_flow_example.yaml) for a complete flow that:

1. Builds an English prompt with Spanish documents and ICL examples
2. Translates the English scaffolding to Spanish (preserving Spanish content)
3. Generates Spanish Q&A pairs
4. Parses and extracts the results

### Flow Structure

```yaml
blocks:
  # Step 1: Build prompt with English template + Spanish data
  - block_type: PromptBuilderBlock
    input_cols:
      document_spanish: document
      icl_document_spanish: icl_document
      # ...
    output_cols: english_prompt

  # Step 2: Translate English scaffolding to Spanish
  - block_type: TranslationBlock
    input_cols: english_prompt
    output_cols: spanish_prompt
    source_language: "en"
    target_language: "es"
    temperature: 0.3  # Low temp for consistency

  # Step 3: Generate Spanish Q&A
  - block_type: LLMChatBlock
    input_cols: spanish_prompt
    output_cols: raw_response
```

## Usage in Python

```python
from sdg_hub.core.blocks.llm import TranslationBlock
from sdg_hub.core.flow import Flow
import pandas as pd

# Create a TranslationBlock
translation_block = TranslationBlock(
    block_name="translate_to_spanish",
    input_cols="english_prompt",
    output_cols="spanish_prompt",
    source_language="en",
    target_language="es",
    model="openai/gpt-4",
    temperature=0.3,
    max_tokens=8192
)

# Use in a dataset
df = pd.DataFrame({
    "english_prompt": [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Generate questions about: El documento en español..."}
        ]
    ]
})

result = translation_block.generate(df)
print(result["spanish_prompt"][0])
# Output: [
#   {"role": "system", "content": "Eres un asistente útil."},
#   {"role": "user", "content": "Genera preguntas sobre: El documento en español..."}
# ]
```

## Supported Languages

The TranslationBlock supports ISO 639-1 language codes. Common codes include:

- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `ja` - Japanese
- `zh` - Chinese
- `pt` - Portuguese
- `ru` - Russian
- `ar` - Arabic
- `hi` - Hindi
- `it` - Italian
- `ko` - Korean

You can use any language code - if it's not in the predefined list, the block will still work using the title-cased version of the code.

## Key Features

### 1. Format Agnostic

Handles both **messages format** (list of dicts) and **plain text**:

```python
# Messages format
messages = [{"role": "user", "content": "Hello"}]

# Plain text
text = "Hello world"
```

### 2. Async Support

For large batches, use async mode:

```python
block = TranslationBlock(
    ...,
    async_mode=True
)

# Can be controlled at flow level with max_concurrency
output = flow.generate(dataset, max_concurrency=50)
```

### 3. LiteLLM Integration

Supports 100+ LLM providers via LiteLLM:

```python
# OpenAI
model="openai/gpt-4"

# Anthropic
model="anthropic/claude-3-sonnet-20240229"

# Local vLLM
model="hosted_vllm/meta-llama/Llama-3-8B"
api_base="http://localhost:8000/v1"
```

### 4. Configurable Parameters

All LiteLLM parameters are supported:

```python
TranslationBlock(
    ...,
    temperature=0.3,      # Lower for more deterministic translation
    max_tokens=8192,      # Large enough for full prompts
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0
)
```

## Best Practices

1. **Use low temperature (0.2-0.4)** for translation to ensure consistency
2. **Set appropriate max_tokens** - translations can be longer than source text
3. **Test with sample data** first to verify the LLM preserves your target language content
4. **Use async_mode** for large datasets to maximize throughput
5. **Monitor costs** - translation requires one LLM call per prompt (or per message if using messages format)

## Troubleshooting

### Issue: Translation changes my source documents

**Solution**: The translation LLM should detect and preserve target language content automatically. If this happens:
- Check that your source documents are clearly in the target language
- Try a more capable model (e.g., GPT-4 instead of GPT-3.5)
- Verify the source/target language codes are correct

### Issue: Translations are inconsistent

**Solution**: Lower the temperature parameter:
```yaml
temperature: 0.2  # More deterministic
```

### Issue: Out of memory with large prompts

**Solution**: Increase max_tokens or split into smaller chunks:
```yaml
max_tokens: 16384  # Larger limit
```

## License

Apache-2.0
