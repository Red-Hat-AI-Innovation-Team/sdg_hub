# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SDG Hub is a modular synthetic data generation toolkit for LLMs. The framework is built around YAML-configured flows that chain computational blocks together to process and generate data.

## Development Commands

### Code Style
- Use numpy style docstrings 
- All functions and methods must include python type hints 
- Write ruff-compliant code

### Testing
- Run all tests: `pytest tests/`
- Run specific test: `pytest tests/test_filename.py`
- Run tests with coverage: `tox -e py3-unitcov`

### Linting and Code Quality
- Format code: `tox -e ruff fix` or `./scripts/ruff.sh fix`
- Check code formatting: `tox -e ruff check`
- Run linting: `tox -e lint` (full pylint) or `tox -e fastlint` (faster)
- Type checking: `tox -e mypy`
- Run all checks: `make verify` (runs fastlint, mypy, ruff via tox)

### Build and Install
- Install for development: `pip install -e .[dev]`
- Install with web interface: `pip install -e .[web_interface]`
- Install with examples dependencies: `pip install -e .[examples]`

### Git Workflow
- **IMPORTANT**: Always create a feature branch and never push directly to main
- **Use git worktrees for local development**: `git worktree add ../feature-branch-name feature-branch-name`
- Create branch: `git checkout -b feature-branch-name`
- Push to branch: `git push origin feature-branch-name`

## Architecture

### Core Components

1. **Blocks** (`src/sdg_hub/blocks/`): Fundamental computational units
   - `Block`: Abstract base class for all blocks
   - `LLMBlock`: Language model generation blocks
   - Utility blocks: filtering, data transformation, column operations

2. **Flows** (`src/sdg_hub/flow.py`): Orchestrates blocks in YAML-defined pipelines
   - Loads YAML configurations
   - Manages block execution order
   - Handles data flow between blocks

3. **Registry System** (`src/sdg_hub/registry.py`): 
   - `BlockRegistry`: Manages available block types
   - `PromptRegistry`: Manages prompt configurations

4. **Prompts** (`src/sdg_hub/configs/`): YAML-based LLM instruction templates
   - Support Jinja2 templating with variable injection
   - Include system instructions, principles, examples, and generation templates

### Data Flow

- Uses Hugging Face Datasets (Arrow tables) for data representation
- Supports checkpointing for long-running flows
- Blocks process datasets and pass results to subsequent blocks

### Flow Configuration

Flows are defined in YAML files with this structure:
```yaml
- block_type: LLMBlock
  block_config:
    block_name: unique_name
    config_path: path/to/prompt.yaml
    model_id: model_name
    output_cols: [column_names]
  gen_kwargs:
    max_tokens: 512
```

### Block Development

When creating new blocks:
1. Inherit from `Block` base class
2. Register with `@BlockRegistry.register("BlockName")`
3. Implement `generate()` method
4. Use `_validate()` for input validation
5. Use `_load_config()` for YAML configuration loading

### Testing Conventions

- Unit tests in `tests/` directory
- Test data in `testdata/` subdirectories
- Use pytest fixtures for common test setup
- Test both positive and negative cases
- Include edge cases and error conditions

## Additional Tips
- Use `rg` in favor of `grep` whenever it's available
- Use `uv` for Python environment management: always start with `uv sync --extra dev` to init the env and run stuff with `uv run`
