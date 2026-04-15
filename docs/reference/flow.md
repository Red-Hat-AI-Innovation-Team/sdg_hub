# Flow API Reference

This page provides auto-generated API documentation from source docstrings.
When reading raw markdown, refer to the quick reference below.

## Quick Reference

| Class | Import | Description |
|-------|--------|-------------|
| Flow | `from sdg_hub import Flow` | Pydantic-based flow for chaining data generation blocks into pipelines |
| FlowRegistry | `from sdg_hub import FlowRegistry` | Registry for managing and discovering contributed flows |
| FlowMetadata | `from sdg_hub import FlowMetadata` | Flow metadata including name, version, author, and descriptions |
| RecommendedModels | `from sdg_hub.core.flow.metadata import RecommendedModels` | Recommended model configurations for a flow |
| DatasetRequirements | `from sdg_hub.core.flow.metadata import DatasetRequirements` | Dataset column and format requirements for a flow |
| FlowValidator | `from sdg_hub import FlowValidator` | Validator for flow YAML configurations and execution readiness |

**Key entry points:** `Flow.from_yaml()`, `flow.generate()`, `flow.dry_run()`, `flow.set_model_config()`

---

## Detailed API (auto-generated)

### Core

::: sdg_hub.core.flow.base.Flow
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.flow.registry.FlowRegistry
    options:
      members_order: source
      show_source: false

---

### Metadata

::: sdg_hub.core.flow.metadata.FlowMetadata
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.flow.metadata.RecommendedModels
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.flow.metadata.DatasetRequirements
    options:
      members_order: source
      show_source: false

---

### Validation

::: sdg_hub.core.flow.validation.FlowValidator
    options:
      members_order: source
      show_source: false
