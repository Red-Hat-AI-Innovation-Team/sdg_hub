# Connectors API Reference

This page provides auto-generated API documentation from source docstrings.
When reading raw markdown, refer to the quick reference below.

## Quick Reference

| Class | Import | Description |
|-------|--------|-------------|
| BaseConnector | `from sdg_hub.core.connectors import BaseConnector` | Abstract base class for all connectors |
| ConnectorConfig | `from sdg_hub.core.connectors import ConnectorConfig` | Base configuration for connectors (url, api_key, timeout, retries) |
| ConnectorRegistry | `from sdg_hub.core.connectors import ConnectorRegistry` | Global registry for connector discovery and retrieval by name |
| BaseAgentConnector | `from sdg_hub.core.connectors import BaseAgentConnector` | Base class for agent framework connectors (async-first pattern) |
| LangflowConnector | `from sdg_hub.core.connectors import LangflowConnector` | Connector for Langflow visual LLM app builder |
| LangGraphConnector | `from sdg_hub.core.connectors import LangGraphConnector` | Connector for LangGraph stateful multi-actor agent framework |

---

## Detailed API (auto-generated)

### Base Classes

::: sdg_hub.core.connectors.base.BaseConnector
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.connectors.base.ConnectorConfig
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.connectors.registry.ConnectorRegistry
    options:
      members_order: source
      show_source: false

---

### Agent Connectors

::: sdg_hub.core.connectors.agent.base.BaseAgentConnector
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.connectors.agent.langflow.LangflowConnector
    options:
      members_order: source
      show_source: false

::: sdg_hub.core.connectors.agent.langgraph.LangGraphConnector
    options:
      members_order: source
      show_source: false
