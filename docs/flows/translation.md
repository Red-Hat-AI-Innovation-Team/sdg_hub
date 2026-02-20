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

## 🚨 **Important Note:**  
All prompts and instructions in the flow are automatically translated using an LLM; this is intended to be a **starting point** for your localized data pipeline. While the system also performs automated verification, we **highly recommend manually reviewing and refining translated outputs** to ensure correctness, clarity, and appropriateness for your application. Automated translation can miss subtle context, formatting, or domain nuances.  

