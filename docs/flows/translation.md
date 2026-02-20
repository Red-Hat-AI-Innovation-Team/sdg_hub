# Flow Translation

Translate any SDG Hub flow and its prompt YAMLs to a target language using LLM-powered translation with automated verification.

## Overview

The `translate_flow()` utility takes an existing flow and produces a fully functional translated copy. It handles:

- **Prompt translation** - All prompt YAML files are translated via an LLM
- **Translation verification** - A second LLM pass validates quality, structural tags, and template variables
- **Flow adaptation** - The `flow.yaml` metadata and prompt paths are updated automatically
- **Registry integration** - Translated flows are registered with `FlowRegistry` for immediate use

```
Source Flow (English)           Translated Flow (Spanish)
├── flow.yaml            →     ├── flow.yaml
└── prompts/                   └── prompts/
    ├── summary.yaml     →         ├── summary_es.yaml
    └── qa.yaml          →         └── qa_es.yaml
```

## Quick Start

### Python API

```python
from sdg_hub.core.utils.translation import translate_flow

translated_flow = translate_flow(
    flow="extractive-summary-knowledge-tuning",  # flow id or name
    lang="Spanish",
    lang_code="es",
    translator_model="openai/gpt-4o",
    verifier_model="openai/gpt-4o",
    translator_api_key="your-api-key",
)

# The translated flow is ready to use
result = translated_flow.generate(dataset)
```

### Command Line

```bash
python -m sdg_hub.core.utils.translation \
    --flow extractive-summary-knowledge-tuning \
    --lang French --lang-code fr
```

Set credentials via environment variables:

```bash
export SDG_TRANSLATION_API_KEY="your-api-key"
export SDG_TRANSLATION_API_BASE="https://api.openai.com/v1"  # optional
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `flow` | `str` | Yes | - | Flow **id** or **name** from `FlowRegistry` |
| `lang` | `str` | Yes | - | Target language name (e.g. `"Spanish"`, `"French"`, `"Japanese"`) |
| `lang_code` | `str` | Yes | - | ISO 639-1 language code (e.g. `"es"`, `"fr"`, `"ja"`) |
| `translator_model` | `str` | No | `"gpt-5.2"` | LLM for translation (litellm format) |
| `verifier_model` | `str` | No | `"gpt-5.2"` | LLM for verification |
| `output_dir` | `str` | No | `"./<flow_dir>_<lang_code>/"` | Output directory |
| `translator_api_key` | `str` | No | `None` | API key for translator model |
| `translator_api_base` | `str` | No | `None` | API base URL for translator model |
| `verifier_api_key` | `str` | No | `None` | API key for verifier (if different) |
| `verifier_api_base` | `str` | No | `None` | API base URL for verifier |
| `max_retries` | `int` | No | `3` | Max translation attempts per prompt on verification failure |
| `verbose` | `bool` | No | `False` | Enable DEBUG-level logging |
| `register` | `bool` | No | `True` | Register translated flow with `FlowRegistry` |

## How It Works

### 1. Flow Resolution

The `flow` parameter is resolved via `FlowRegistry` to a filesystem path. You can pass either a flow **id** (e.g. `"extractive-summary-knowledge-tuning"`) or a flow **name** (e.g. `"Extractive Summary Knowledge Tuning Dataset Generation Flow"`).

### 2. Prompt Discovery

The flow YAML is parsed to discover all `prompt_config_path` references and `TagParserBlock` structural tags. Only prompts referenced by the flow are translated — no hardcoded file lists.

### 3. Translation and Verification Loop

For each prompt YAML, every message's `content` field is translated through a retry loop:

1. **Translate** the content using the translator model
2. **Validate programmatically** — check that Jinja2 variables (`{{document}}`, `{{query}}`) and structural tags (`[QUESTION]`, `[END]`) are preserved
3. **Verify with LLM** — a second model confirms translation quality
4. If validation fails, retry up to `max_retries` times

### 4. Header Comment Preservation

Prompt files that contain header comments (used to differentiate duplicate prompts for PyPI packaging) are preserved and adapted:

```yaml
# Source prompt
# Prompt used in: detailed_summary flow
# Origin: knowledge_infusion/enhanced_multi_summary_qa

# Translated prompt (Origin updated automatically)
# Prompt used in: detailed_summary flow
# Origin: knowledge_infusion/enhanced_multi_summary_qa_es
```

### 5. Flow YAML Adaptation

A new `flow.yaml` is created with:
- Updated `metadata.name` — appends `(<Language>)` (e.g. `"Extractive Summary (Spanish)"`)
- Updated `metadata.id` — appends `-<lang_code>` (e.g. `"extractive-summary-es"`)
- Rewritten `prompt_config_path` values pointing to translated prompt files

### 6. Registry Integration

If `register=True` (default), the output directory is added to `FlowRegistry` search paths and flows are re-discovered. The translated flow is immediately available via `FlowRegistry.get_flow_path()`.

## Examples

### Translate a Single Flow

```python
from sdg_hub.core.utils.translation import translate_flow
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()

# Translate the extractive summary flow to French
flow = translate_flow(
    flow="Extractive Summary Knowledge Tuning Dataset Generation Flow",
    lang="French",
    lang_code="fr",
    translator_model="openai/gpt-4o",
    verifier_model="openai/gpt-4o",
    translator_api_key="sk-...",
)

# Use the translated flow
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
)
result = flow.generate(dataset)
```

### Translate All Four Knowledge Tuning Flows

```python
from sdg_hub.core.utils.translation import translate_flow
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()

FLOWS = [
    "Extractive Summary Knowledge Tuning Dataset Generation Flow",
    "Detailed Summary Knowledge Tuning Dataset Generation Flow",
    "Key Facts Knowledge Tuning Dataset Generation Flow",
    "Document Based Knowledge Tuning Dataset Generation Flow",
]

for flow_name in FLOWS:
    print(f"Translating: {flow_name}")
    translate_flow(
        flow=flow_name,
        lang="German",
        lang_code="de",
        translator_model="openai/gpt-4o",
        verifier_model="openai/gpt-4o",
        translator_api_key="sk-...",
        output_dir=f"./translated_flows/de/{FlowRegistry.get_flow_path(flow_name).split('/')[-2]}",
    )

# All four German flows are now registered and ready
FlowRegistry.discover_flows()
german_flows = FlowRegistry.search_flows(tag="knowledge-tuning")
```

### Idempotent Translation (Skip If Exists)

`translate_flow()` is idempotent — if a translated flow already exists in the registry or output directory, it skips translation and returns the existing flow:

```python
# First call: translates
flow = translate_flow(flow="my-flow", lang="Spanish", lang_code="es", ...)

# Second call: skips translation, returns existing flow
flow = translate_flow(flow="my-flow", lang="Spanish", lang_code="es", ...)
# Logs: "Flow 'my-flow-es' already registered, skipping translation"
```

### Using with the Knowledge Generation Notebook

The [knowledge generation notebook](../../examples/knowledge_tuning/enhanced_summary_knowledge_tuning/knowledge_generation.ipynb) has built-in multilingual support. Set environment variables in your `.env` file:

```dotenv
SDG_LANG=Spanish
SDG_LANG_CODE=es

# Translation model (only needed if translated flows don't exist yet)
TRANSLATOR_MODEL=openai/gpt-4o
TRANSLATOR_API_KEY=sk-...
```

The notebook calls `translate_flow()` on-demand for any flows that aren't already translated.

## Pre-translated Flows

The repository ships with these pre-translated flow variants:

| Language | Code | Location | Status |
|----------|------|----------|--------|
| Spanish | `es` | `src/sdg_hub/flows/knowledge_infusion/enhanced_multi_summary_qa_es/` | All 4 sub-flows |
| Japanese | `ja` | `src/sdg_hub/flows/knowledge_infusion/japanese_multi_summary_qa/` | Single combined flow |

Pre-translated flows are auto-discovered by `FlowRegistry` — no extra configuration needed.

## Troubleshooting

### Translation Verification Failures

If the verifier repeatedly rejects translations, check:

- **Structural tags** — Tags like `[QUESTION]`, `[END]` must appear verbatim in translated output. The translator model is instructed to preserve them, but weaker models may still translate them.
- **Jinja2 variables** — Template variables like `{{document}}` must be preserved exactly. Check the validation issues in the log output.
- **Max retries** — Increase `max_retries` (default 3) for difficult prompts.

### Model Selection

Use capable models for translation. Recommended:
- `openai/gpt-4o` — Good balance of quality and cost
- `anthropic/claude-sonnet-4-20250514` — High quality translations

Smaller models may struggle to preserve structural tags and template variables.

### Output Directory Collisions

When translating multiple sub-flows of the same parent flow, each sub-flow needs its own output directory. The default `output_dir` is derived from the source flow's directory name, so sub-flows within the same parent automatically get unique paths.
