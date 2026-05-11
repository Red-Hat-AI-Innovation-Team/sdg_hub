# Architecture

This document describes the high-level architecture of SDG Hub.
If you want to familiarize yourself with the codebase, this is a good place to start.

See also [CLAUDE.md](CLAUDE.md) for development commands and contribution guidelines.

## Overview

SDG Hub is a Python framework for synthetic data generation using composable
blocks and flows. The core data flow is:

```
dataset -> Block_1 -> Block_2 -> ... -> Block_N -> enriched_dataset
```

**Blocks** are self-contained processing units that transform a pandas DataFrame.
**Flows** chain blocks into pipelines defined in YAML. **Connectors** integrate
with external services (agent frameworks, HTTP APIs, code interpreters).

## Source Layout

```
src/sdg_hub/
    __init__.py
    _version.py

    core/                         # Framework internals
        blocks/                   # Block system
            base.py               #   BaseBlock (Pydantic + ABC)
            registry.py           #   BlockRegistry + BlockMetadata
            llm/                  #   LLMChatBlock, PromptBuilderBlock, LLMResponseExtractorBlock, ErrorHandlerBlock
            parsing/              #   TagParserBlock, RegexParserBlock, JSONParserBlock (+ BaseTextParserBlock)
            transform/            #   TextConcatBlock, RenameColumnsBlock, MeltColumnsBlock, RowMultiplierBlock, SamplerBlock, DuplicateColumnsBlock, IndexBasedMapperBlock, JsonStructureBlock, UniformColValSetterBlock
            filtering/            #   ColumnValueFilterBlock
            agent/                #   AgentBlock, AgentResponseExtractorBlock
            mcp/                  #   MCPAgentBlock (agentic tool-use with remote MCP servers)
            code/                 #   PythonInterpreterBlock

        connectors/               # External service integrations
            base.py               #   BaseConnector + ConnectorConfig (Pydantic + ABC)
            registry.py           #   ConnectorRegistry
            exceptions.py         #   ConnectorError, ConnectorHTTPError
            agent/                #   Agent framework connectors
                base.py           #     BaseAgentConnector (MLflow ChatAgent types)
                langflow.py       #     LangflowConnector
                langgraph.py      #     LangGraphConnector
                generic_http.py   #     GenericHTTPConnector
            http/                 #   Shared HTTP client
                client.py         #     HttpClient (httpx + tenacity retry)
            code_interpreter/     #   Code execution connectors
                base.py           #     BaseCodeInterpreterConnector
                monty.py          #     MontyConnector

        flow/                     # Pipeline orchestration
            base.py               #   Flow (Pydantic model, block chaining)
            registry.py           #   FlowRegistry + FlowRegistryEntry
            serialization.py      #   YAML <-> Flow round-tripping
            execution.py          #   Pipeline execution engine
            validation.py         #   Pre-run validation
            checkpointer.py       #   FlowCheckpointer (resumable execution)
            column_tracker.py     #   Column lineage tracking
            model_config.py       #   LLM model configuration
            agent_config.py       #   Agent framework configuration
            metadata.py           #   FlowMetadata, DatasetRequirements
            display.py            #   Rich console display helpers

        utils/                    # Cross-cutting utilities
            logger_config.py      #   setup_logger (structured logging)
            error_handling.py     #   Shared exception types
            config_helpers.py     #   Configuration utilities
            datautils.py          #   DataFrame helpers
            message_formatter.py  #   Chat message formatting
            yaml_utils.py         #   YAML save/load helpers
            path_resolution.py    #   Path resolution for flow assets
            flow_identifier.py    #   Flow ID generation
            flow_metrics.py       #   Execution metrics collection
            time_estimator.py     #   ETA estimation for long runs
            translation.py        #   i18n utilities
            prompts/              #   Shared prompt templates (YAML)

    flows/                        # Pre-built flow definitions (YAML + prompts)
        knowledge_infusion/       #   Document-grounded QA generation (localized variants: Spanish, Japanese)
        evaluation/               #   RAG evaluation, RAG evaluation ICL, MCP eval benchmark
        agentic/                  #   MCP tool-use distillation
        red_team/                 #   Adversarial prompt generation
        text_analysis/            #   Structured text insights
        code_evaluation/          #   Domain code evaluation
```

## Package Layering

Dependencies flow strictly downward through four layers:

```
Layer 4 (pre-built)    flows/                YAML pipeline definitions
                           |
Layer 3 (impl)         blocks/{llm,parsing,  Concrete block/connector classes
                       transform,...}
                       connectors/{agent,
                       http,code_interpreter}
                           |
Layer 2 (registry)     blocks/registry.py    BlockRegistry, ConnectorRegistry,
                       connectors/registry.py FlowRegistry
                       flow/registry.py
                           |
Layer 1 (base)         blocks/base.py        BaseBlock, BaseConnector, Flow
                       connectors/base.py
                       flow/base.py
                           |
Layer 0 (utils)        utils/                logger_config, error_handling,
                                             datautils, config_helpers, ...
```

### Layering rules

1. **Implementations MUST NOT import from other implementations.** A block in
   `blocks/llm/` never imports from `blocks/parsing/` or `blocks/transform/`.
   If two blocks need to share behavior, it belongs in a base class or utility.

2. **Cross-cutting concerns enter through explicit config interfaces.** LLM
   settings flow via `flow/model_config.py`, agent framework settings via
   `flow/agent_config.py`, and logging via `utils/logger_config.py`. Blocks
   receive these through the Flow at execution time, not through direct imports.

3. **Registries depend on base classes only.** `BlockRegistry` knows about
   `BaseBlock` but not any concrete block. Registration happens via decorators
   at import time (`@BlockRegistry.register(...)`).

4. **Flows are data, not code.** Pre-built flows in `flows/` are YAML files
   that reference blocks by their registered names. They contain no Python.

## Extension Points

### Adding a new block

1. Create a file under the appropriate `core/blocks/<category>/` directory.
2. Inherit from `BaseBlock` and implement the `generate()` method.
3. Use Pydantic fields for configuration; use `input_cols`/`output_cols` for
   column handling.
4. Register with `@BlockRegistry.register(name, category, description)`.
5. Add tests under `tests/blocks/`.

See [docs/agent-knowledge/block-invariants.md](docs/agent-knowledge/block-invariants.md)
for the full checklist.

### Adding a new flow

1. Create a directory under `src/sdg_hub/flows/` with a YAML pipeline
   definition and any prompt templates.
2. The flow YAML references blocks by their registered names. Use
   `FlowRegistry.discover_flows()` to verify it is found.
3. Call `Flow.from_yaml(path)` to load, then `flow.generate(dataset)` to run.

See [docs/agent-knowledge/flow-invariants.md](docs/agent-knowledge/flow-invariants.md)
for the full checklist.

### Adding a new connector

1. Create a file in `src/sdg_hub/core/connectors/agent/` (for agent
   frameworks) or the appropriate sub-package.
2. Inherit from `BaseAgentConnector` (or `BaseConnector` for non-agent use
   cases). Implement `build_request()` and `parse_response()`.
3. Register with `@ConnectorRegistry.register("name")`.
4. Configure at runtime via `flow.set_agent_config(agent_framework="name", ...)`.

See [docs/agent-knowledge/connector-invariants.md](docs/agent-knowledge/connector-invariants.md)
for the full checklist.

## Key Entry Points

| What you want to do | Start here |
|---|---|
| Run an existing flow | `FlowRegistry.discover_flows()`, `Flow.from_yaml()` |
| Build a custom pipeline | Write YAML, `Flow.from_yaml()`, `flow.generate()` |
| Dry-run without LLM calls | `flow.dry_run(dataset)` |
| List available blocks | `BlockRegistry.discover_blocks()` |
| List available connectors | `ConnectorRegistry.list()` |
| Resume a failed run | Pass `checkpoint_dir` to `flow.generate()` |
