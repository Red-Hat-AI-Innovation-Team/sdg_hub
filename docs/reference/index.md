# API Reference

This section contains the auto-generated API reference for SDG Hub, built
directly from source code docstrings using
[mkdocstrings](https://mkdocstrings.github.io/).

## Modules

| Module | Description |
|--------|-------------|
| [Blocks](blocks.md) | Base classes, LLM blocks, parsing, transform, filtering, agent, and MCP blocks |
| [Flow](flow.md) | Flow orchestration, metadata, registry, and validation |
| [Connectors](connectors.md) | Connector base classes, registry, and agent framework integrations |

## Import Paths

SDG Hub provides convenient top-level imports for the most commonly used
classes:

```python
# Top-level imports (recommended for most use cases)
from sdg_hub import BaseBlock, BlockRegistry, Flow, FlowRegistry

# Submodule imports (for less common classes)
from sdg_hub.core.blocks.llm import LLMChatBlock, PromptBuilderBlock
from sdg_hub.core.blocks.parsing import TagParserBlock, JSONParserBlock
from sdg_hub.core.connectors import LangflowConnector, ConnectorRegistry
```

Both styles work identically. The top-level package re-exports key classes from
their definition modules for convenience.
