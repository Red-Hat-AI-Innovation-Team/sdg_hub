# Spanish Knowledge Tuning Flows

Spanish-language translations of the [Enhanced Multi-Summary QA](../../) flows for knowledge tuning.
These flows follow the exact same pipeline architecture as the English originals — all prompts are translated to Spanish, the generated QA data is in Spanish.

## Available Flows

| Flow | Registry ID | Description |
|------|-------------|-------------|
| Extractive Summary | `epic-jade-656-es` | Extractive summary → Q&A generation |
| Detailed Summary | `mild-thunder-748-es` | Detailed summary → Q&A generation |
| Key Facts | `heavy-heart-77-es` | Atomic facts extraction → Q&A generation (5 QA pairs per fact) |
| Document Based QA | `stellar-peak-605-es` | Direct document → Q&A generation |

## Usage

The Spanish flows are auto-discovered by the `FlowRegistry`. Set `SDG_LANG=Spanish` in your `.env` and run [knowledge_generation.ipynb](../../../../../examples/knowledge_tuning/enhanced_summary_knowledge_tuning/knowledge_generation.ipynb).

For seed data format, experiment results, and full documentation, see the [main README](../../../../../examples/knowledge_tuning/enhanced_summary_knowledge_tuning/README.md#multilingual-support).
