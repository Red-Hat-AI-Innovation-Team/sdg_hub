# Phase 1: Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure SDG Hub's agent-facing documentation into a progressive-disclosure knowledge base — CLAUDE.md as a short table of contents pointing to deep sources of truth in `docs/agent-knowledge/`.

**Architecture:** CLAUDE.md shrinks from 136 to ~80-100 lines. Detailed invariants, principles, and standards move to individual markdown files in `docs/agent-knowledge/`. A new `ARCHITECTURE.md` provides the top-level domain map. PR and issue templates standardize agent output. Legacy config files are removed.

**Tech Stack:** Markdown, GitHub templates (YAML frontmatter), conventional commits.

**Spec:** `docs/superpowers/specs/2026-05-07-agent-maintained-sdg-hub-design.md` — Sections 3.1 through 3.6.

---

## File Structure

```
# New files
ARCHITECTURE.md                                    — Top-level domain map
docs/agent-knowledge/index.md                      — Master index with verification dates
docs/agent-knowledge/core-principles.md            — Golden principles for SDG Hub
docs/agent-knowledge/block-invariants.md           — Rules every block must follow
docs/agent-knowledge/flow-invariants.md            — Rules every flow YAML must follow
docs/agent-knowledge/connector-invariants.md       — Rules every connector must follow
docs/agent-knowledge/testing-standards.md          — What "tested" means, coverage rules
docs/agent-knowledge/grading-criteria.md           — Quality criteria with thresholds
docs/agent-knowledge/decision-rubric.md            — When to auto-fix, flag, or escalate
docs/agent-knowledge/QUALITY.md                    — Quality grades per domain/layer
docs/agent-knowledge/tech-debt-tracker.md          — Known debt, prioritized
docs/exec-plans/active/.gitkeep                    — Execution plan staging
docs/exec-plans/completed/.gitkeep                 — Completed plans (institutional memory)
.github/PULL_REQUEST_TEMPLATE.md                   — Structured PR template for agents
.github/ISSUE_TEMPLATE/bug_report.md               — Bug report issue template
.github/ISSUE_TEMPLATE/feature_request.md          — Feature request issue template
.github/ISSUE_TEMPLATE/new_block.md                — New block request template
.github/ISSUE_TEMPLATE/config.yml                  — Issue template chooser config
.env.example                                       — Environment variable reference

# Modified files
CLAUDE.md                                          — Restructure to table of contents

# Deleted files
.isort.cfg                                         — Redundant with ruff
.pylintrc                                          — Redundant with ruff
```

---

### Task 1: Create ARCHITECTURE.md

**Files:**
- Create: `ARCHITECTURE.md`

- [ ] **Step 1: Write ARCHITECTURE.md**

```markdown
# Architecture

This document describes the high-level architecture of SDG Hub.
If you want to familiarize yourself with the codebase, this is
the place to start.

## Overview

SDG Hub is a Python framework for synthetic data generation using
composable blocks and flows.

Core data flow: `dataset → Block₁ → Block₂ → ... → enriched_dataset`

## Source Layout

```
src/sdg_hub/
├── core/
│   ├── blocks/           # Data processing units (the "atoms")
│   │   ├── base.py       # BaseBlock — all blocks inherit from this
│   │   ├── registry.py   # BlockRegistry — decorator-based registration
│   │   ├── llm/          # LLM-powered blocks (chat, prompt building, extraction)
│   │   ├── parsing/      # Output parsers (tag, regex, JSON)
│   │   ├── transform/    # Data transformers (concat, rename, melt, multiply)
│   │   ├── filtering/    # Row/column filters
│   │   ├── agent/        # Agent framework blocks
│   │   ├── mcp/          # MCP tool-use blocks
│   │   └── code/         # Code execution blocks
│   ├── flow/             # Pipeline engine (the "molecules")
│   │   ├── base.py       # Flow — YAML-defined pipeline runner
│   │   ├── registry.py   # FlowRegistry — auto-discovers built-in flows
│   │   ├── execution.py  # Pipeline execution logic
│   │   ├── validation.py # Pre-run validation
│   │   ├── checkpointer.py # Resumable execution
│   │   └── ...
│   ├── connectors/       # External integrations
│   │   ├── base.py       # BaseConnector
│   │   ├── registry.py   # ConnectorRegistry
│   │   ├── agent/        # Agent framework connectors (langflow, langgraph, generic_http)
│   │   ├── http/         # HTTP client
│   │   └── code_interpreter/ # Code execution connectors
│   └── utils/            # Shared helpers
├── flows/                # Pre-built flow YAML files + prompt templates
└── __init__.py
```

## Package Layering

Dependencies flow in one direction:

```
BaseBlock / BaseConnector / Flow (base classes)
  → Registries (BlockRegistry, FlowRegistry, ConnectorRegistry)
    → Implementations (specific blocks, connectors, flow YAMLs)
      → Utils (helpers consumed by implementations)
```

**Rules:**
- Implementations MUST NOT import from other implementations directly
- Cross-cutting concerns (LLM access, agent config, logging) enter
  through explicit config interfaces: `model_config`, `agent_config`,
  `logger_config`

## Extension Points

- **Adding a block:** Inherit `BaseBlock`, implement `generate()`,
  register with `@BlockRegistry.register()`. See `docs/agent-knowledge/block-invariants.md`.
- **Adding a flow:** Create a YAML in `src/sdg_hub/flows/`. See
  `docs/agent-knowledge/flow-invariants.md`.
- **Adding a connector:** Inherit `BaseConnector`, register with
  `@ConnectorRegistry.register()`. See `docs/agent-knowledge/connector-invariants.md`.
```

- [ ] **Step 2: Verify the file renders correctly**

Run: `head -20 ARCHITECTURE.md`
Expected: The markdown header and overview section display correctly.

- [ ] **Step 3: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "docs: add ARCHITECTURE.md with domain map and package layering"
```

---

### Task 2: Create the Knowledge Base Directory

**Files:**
- Create: `docs/agent-knowledge/index.md`
- Create: `docs/agent-knowledge/core-principles.md`
- Create: `docs/agent-knowledge/block-invariants.md`
- Create: `docs/agent-knowledge/flow-invariants.md`
- Create: `docs/agent-knowledge/connector-invariants.md`
- Create: `docs/agent-knowledge/testing-standards.md`

- [ ] **Step 1: Create `docs/agent-knowledge/index.md`**

```markdown
# Agent Knowledge Base — Index

Last verified: 2026-05-08

This directory is the system of record for agent-facing documentation.
CLAUDE.md points here. Agents should read the relevant file for their
current task, not all files at once (progressive disclosure).

| File | What it covers | Last verified |
|------|---------------|---------------|
| [core-principles.md](core-principles.md) | Golden principles for all SDG Hub development | 2026-05-08 |
| [block-invariants.md](block-invariants.md) | Rules every block must follow | 2026-05-08 |
| [flow-invariants.md](flow-invariants.md) | Rules every flow YAML must follow | 2026-05-08 |
| [connector-invariants.md](connector-invariants.md) | Rules every connector must follow | 2026-05-08 |
| [testing-standards.md](testing-standards.md) | What "tested" means, coverage rules | 2026-05-08 |
| [grading-criteria.md](grading-criteria.md) | Quality criteria with hard thresholds | 2026-05-08 |
| [decision-rubric.md](decision-rubric.md) | When to auto-fix, flag, or escalate | 2026-05-08 |
| [QUALITY.md](QUALITY.md) | Quality grades per domain/layer | 2026-05-08 |
| [tech-debt-tracker.md](tech-debt-tracker.md) | Known debt, prioritized | 2026-05-08 |
```

- [ ] **Step 2: Create `docs/agent-knowledge/core-principles.md`**

```markdown
# Core Principles

These are the golden principles for all SDG Hub development.
Agents and humans should follow these in every change.

1. **Prefer shared utility packages over hand-rolled helpers.**
   Check `src/sdg_hub/core/utils/` before writing a new helper.
   If something similar exists, extend it rather than duplicating.

2. **Validate data shapes at boundaries.**
   Use Pydantic models for configuration, not raw dicts. Blocks
   use Pydantic fields. Flow configs are validated on load.

3. **Every block must be registered, tested, and documented.**
   See `block-invariants.md` for the full checklist.

4. **Flow YAMLs must have metadata.**
   Every flow YAML must include `metadata` with `author`,
   `description`, and `tags`. See `flow-invariants.md`.

5. **Structured logging only.**
   Use `from ..utils.logger_config import setup_logger` — never
   raw `print()`. Ruff rule T201 enforces this.

6. **Tests must assert meaningful output.**
   Don't just assert "something was returned." Assert specific
   values, shapes, or properties. See `testing-standards.md`.
```

- [ ] **Step 3: Create `docs/agent-knowledge/block-invariants.md`**

```markdown
# Block Invariants

Every block in SDG Hub must satisfy these invariants.
Structural tests in `tests/structural/` enforce these mechanically.

## Required

1. **Inherit from `BaseBlock`** (`src/sdg_hub/core/blocks/base.py`)
2. **Implement the `generate()` method** — this is the core processing logic
3. **Use Pydantic fields for configuration** — not `__init__` args or raw dicts
4. **Declare `input_cols` and `output_cols`** — these define the block's data contract
5. **Register with the BlockRegistry:**
   ```python
   @BlockRegistry.register(
       name="my_block",
       category="transform",
       description="Short description of what this block does"
   )
   class MyBlock(BaseBlock):
       ...
   ```
6. **Have a corresponding test file** at `tests/blocks/{category}/test_{name}.py`

## Naming Conventions

- Block class names end in `Block` (e.g., `TextConcatBlock`, `TagParserBlock`)
- File names use snake_case matching the class (e.g., `text_concat.py`)
- Registration name uses snake_case (e.g., `text_concat`)

## Where to Put New Blocks

| Category | Directory | When to use |
|----------|-----------|-------------|
| `llm` | `src/sdg_hub/core/blocks/llm/` | Blocks that call LLMs |
| `parsing` | `src/sdg_hub/core/blocks/parsing/` | Blocks that parse LLM output |
| `transform` | `src/sdg_hub/core/blocks/transform/` | Data transformation blocks |
| `filtering` | `src/sdg_hub/core/blocks/filtering/` | Row/column filtering blocks |
| `agent` | `src/sdg_hub/core/blocks/agent/` | Agent framework blocks |
| `mcp` | `src/sdg_hub/core/blocks/mcp/` | MCP tool-use blocks |
| `code` | `src/sdg_hub/core/blocks/code/` | Code execution blocks |
```

- [ ] **Step 4: Create `docs/agent-knowledge/flow-invariants.md`**

```markdown
# Flow Invariants

Every flow YAML in SDG Hub must satisfy these invariants.

## Required Structure

Every flow YAML must have two top-level keys: `metadata` and `blocks`.

```yaml
metadata:
  name: my_flow
  author: "Your Name"
  description: "What this flow does"
  tags:
    - category_tag
    - use_case_tag

blocks:
  - name: step_1
    block_type: registered_block_name
    config:
      # block-specific configuration
```

## Metadata Requirements

- `name` — unique identifier (snake_case)
- `author` — who created this flow
- `description` — one-line summary of what the flow does
- `tags` — at least one tag for categorization

## Location

- Pre-built flows go in `src/sdg_hub/flows/{category}/`
- Categories: `agentic/`, `evaluation/`, `knowledge_infusion/`,
  `red_team/`, `text_analysis/`, `code_evaluation/`

## Validation

- `FlowRegistry.discover_flows()` must find the flow
- `flow.dry_run(dataset)` must succeed without errors
- All referenced `block_type` values must be registered in `BlockRegistry`
```

- [ ] **Step 5: Create `docs/agent-knowledge/connector-invariants.md`**

```markdown
# Connector Invariants

Every connector in SDG Hub must satisfy these invariants.

## Required

1. **Inherit from the appropriate base class:**
   - Agent connectors: `BaseAgentConnector` (`src/sdg_hub/core/connectors/agent/base.py`)
   - Code interpreters: `BaseCodeInterpreter` (`src/sdg_hub/core/connectors/code_interpreter/base.py`)

2. **Implement required methods:**
   - Agent connectors: `build_request()` and `parse_response()`
   - Code interpreters: `execute()`

3. **Register with ConnectorRegistry:**
   ```python
   @ConnectorRegistry.register("my_connector")
   class MyConnector(BaseAgentConnector):
       ...
   ```

4. **Have a corresponding test file** at `tests/connectors/{type}/test_{name}.py`

## Naming Conventions

- Connector class names end in `Connector` (e.g., `LangflowConnector`)
- File names use snake_case (e.g., `langflow.py`)

## Existing Connectors

| Name | Type | Location |
|------|------|----------|
| `langflow` | agent | `src/sdg_hub/core/connectors/agent/langflow.py` |
| `langgraph` | agent | `src/sdg_hub/core/connectors/agent/langgraph.py` |
| `generic_http` | agent | `src/sdg_hub/core/connectors/agent/generic_http.py` |
| `monty` | code_interpreter | `src/sdg_hub/core/connectors/code_interpreter/monty.py` |
```

- [ ] **Step 6: Create `docs/agent-knowledge/testing-standards.md`**

```markdown
# Testing Standards

## What "Tested" Means

A block/flow/connector is considered tested when:

1. A test file exists at the expected path (see invariants docs)
2. Tests cover both **success** and **error** cases
3. Tests assert **specific values or properties**, not just
   "something was returned"
4. Test coverage for the module is ≥80%

## Test Organization

```
tests/
├── blocks/           # One subdirectory per block category
│   ├── llm/
│   ├── parsing/
│   ├── transform/
│   ├── filtering/
│   ├── agent/
│   ├── mcp_blocks/
│   └── testdata/     # YAML config fixtures
├── connectors/       # Mirrors src/sdg_hub/core/connectors/
├── flow/             # Flow engine tests
│   └── regression/   # Auto-generated flow regression tests
├── utils/            # Utility tests
└── integration/      # End-to-end tests (require API keys)
```

## Test Naming

- Test files: `test_{module_name}.py`
- Test functions: `test_{behavior_being_tested}`
- Example: `test_text_concat.py::test_concatenates_two_columns`

## Mocking LLM Clients

When testing LLM-powered blocks, mock the LLM client:

```python
from unittest.mock import MagicMock, patch

@patch("sdg_hub.core.blocks.llm.chat_block.completion")
def test_llm_chat_block(mock_completion):
    mock_completion.return_value = MockResponse(content="mocked output")
    # ... test the block with mocked LLM
```

## Running Tests

```bash
# Unit tests (excludes slow/integration)
uv run pytest tests/blocks tests/connectors tests/flow tests/utils \
  -m "not (examples or slow)"

# With coverage
uv run pytest --cov=sdg_hub --cov-report=term \
  tests/blocks tests/connectors tests/flow tests/utils

# Structural tests (architecture enforcement)
uv run pytest tests/structural/
```

## Marks

- `@pytest.mark.integration` — requires API keys
- `@pytest.mark.slow` — takes >30 seconds
- `@pytest.mark.examples` — requires examples dependencies
```

- [ ] **Step 7: Commit**

```bash
git add docs/agent-knowledge/
git commit -m "docs: add agent knowledge base with invariants and standards"
```

---

### Task 3: Create Grading Criteria, Decision Rubric, Quality Grades, and Tech Debt Tracker

**Files:**
- Create: `docs/agent-knowledge/grading-criteria.md`
- Create: `docs/agent-knowledge/decision-rubric.md`
- Create: `docs/agent-knowledge/QUALITY.md`
- Create: `docs/agent-knowledge/tech-debt-tracker.md`

- [ ] **Step 1: Create `docs/agent-knowledge/grading-criteria.md`**

```markdown
# Grading Criteria

Agents grade their own and each other's work against these criteria.
Each criterion has a hard threshold — if any fails, the change is
rejected with actionable feedback.

| Criterion | What it measures | Threshold |
|-----------|-----------------|-----------|
| **Correctness** | Does the block/flow produce expected output for known inputs? | Hard fail if wrong |
| **Composability** | Does it integrate into the block/flow/connector system cleanly? | Must follow registry pattern |
| **Test quality** | Are tests meaningful? Do they cover success and error cases? | ≥80% coverage, both paths tested |
| **Documentation** | Is usage clear from docstrings and YAML metadata? | Public methods have docstrings |

## How to Grade

When evaluating a change, check each criterion in order.
Report the first failure with specific, actionable feedback:

- **PASS** — all criteria met, with evidence citations
- **NEEDS_WORK** — specific findings with file paths and line numbers

## Example Evaluator Output

```
NEEDS_WORK

- Correctness: PASS — test_text_concat.py passes, output matches expected
- Composability: PASS — registered with @BlockRegistry.register()
- Test quality: FAIL — tests/blocks/transform/test_new_block.py only tests
  the happy path. Missing: test for empty DataFrame input, test for missing
  column. See docs/agent-knowledge/testing-standards.md
- Documentation: PASS — generate() has docstring
```
```

- [ ] **Step 2: Create `docs/agent-knowledge/decision-rubric.md`**

```markdown
# Decision Rubric

When agents encounter issues, use this rubric to decide whether
to auto-fix, flag for review, or escalate to a human.

## Confidence Thresholds

| Confidence | Action |
|-----------|--------|
| High (>80%) | Auto-fix silently |
| Medium (50-80%) | Fix but add `needs-review` label to PR |
| Low (<50%) | Escalate: add `needs-human-review` label, comment explaining what's needed |

## Trust Tiers

1. **Admin** (human maintainer) — can override anything
2. **CI/precheck gate** — non-overridable, even by admins in automated flows
3. **Agent reviewer** — findings must be validated (agents can hallucinate)
4. **External contributor** — requires human approval for merge

## When to Escalate

Always escalate when:
- The change affects the public API (new parameters, removed methods)
- The change modifies core base classes (`BaseBlock`, `BaseConnector`, `Flow`)
- The change touches security-sensitive code (API key handling, auth)
- The agent is unsure whether a test failure is a real bug or a flaky test
- The change would break backward compatibility
- The agent has attempted 5+ iterations without resolving reviewer feedback
```

- [ ] **Step 3: Create `docs/agent-knowledge/QUALITY.md`**

```markdown
# Quality Grades — SDG Hub

Last updated: 2026-05-08

These grades are updated daily by the quality grading agent.
Agents reference this to prioritize improvement work.

| Domain | Test Coverage | Lint | Types | Docs | Overall |
|--------|-------------|------|-------|------|---------|
| blocks/llm | — | — | — | — | Pending |
| blocks/parsing | — | — | — | — | Pending |
| blocks/transform | — | — | — | — | Pending |
| blocks/filtering | — | — | — | — | Pending |
| blocks/agent | — | — | — | — | Pending |
| blocks/mcp | — | — | — | — | Pending |
| blocks/code | — | — | — | — | Pending |
| flow/ | — | — | — | — | Pending |
| connectors/ | — | — | — | — | Pending |
| utils/ | — | — | — | — | Pending |

Grades will be populated when the quality grading Autopilot runs
for the first time (Phase 4).
```

- [ ] **Step 4: Create `docs/agent-knowledge/tech-debt-tracker.md`**

```markdown
# Tech Debt Tracker

Known technical debt, prioritized by impact.

## High Priority

- [ ] Legacy `.isort.cfg` and `.pylintrc` files are redundant with ruff
  (removed in Phase 1)
- [ ] Stale `__pycache__` directories exist for deleted test modules
  under `tests/blocks/evaluation/`, `tests/blocks/utilblocks/`,
  `tests/blocks/column_ops/`
- [ ] Two documentation systems (MkDocs + Next.js website) create drift risk

## Medium Priority

- [ ] No top-level `conftest.py` with shared test fixtures
- [ ] No `.env.example` at project root
- [ ] Some block categories missing in the block category table in CLAUDE.md
  (evaluation, generator, code are not listed)

## Low Priority

- [ ] `mypy` config disables `import-not-found` and `import-untyped` errors
- [ ] 5 legacy files excluded from mypy checking
```

- [ ] **Step 5: Commit**

```bash
git add docs/agent-knowledge/grading-criteria.md \
        docs/agent-knowledge/decision-rubric.md \
        docs/agent-knowledge/QUALITY.md \
        docs/agent-knowledge/tech-debt-tracker.md
git commit -m "docs: add grading criteria, decision rubric, quality grades, and tech debt tracker"
```

---

### Task 4: Restructure CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Rewrite CLAUDE.md as a table of contents**

Replace the entire contents of `CLAUDE.md` with a shorter version that keeps high-frequency reference sections (dev commands, CI table) but moves detailed orientation to the knowledge base. Target: ~80-100 lines.

```markdown
# CLAUDE.md

## Project Overview

**Requirements:** Python 3.10+

SDG Hub is a Python framework for synthetic data generation using composable blocks and flows. Blocks are processing units that transform datasets; flows chain blocks into pipelines defined in YAML.

Core concept: `dataset → Block₁ → Block₂ → Block₃ → enriched_dataset`

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full domain map and package layering.

## Development Commands

**Use `uv` for all Python commands and package management.**

```bash
# Install with dev dependencies
uv pip install .[dev]

# IMPORTANT: Always install pre-commit hooks after cloning
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

### Testing

```bash
# Unit tests (excludes slow/integration)
uv run pytest tests/blocks tests/connectors tests/flow tests/utils -m "not (examples or slow)"

# Structural tests (architecture enforcement)
uv run pytest tests/structural/

# With coverage
uv run pytest --cov=sdg_hub --cov-report=term tests/blocks tests/connectors tests/flow tests/utils
```

### Linting and Formatting

```bash
uv run ruff check --fix src/ tests/    # Lint with auto-fix
uv run ruff format src/ tests/         # Format
uv run mypy src/                       # Type check
```

## Agent Knowledge Base

Before making changes, read the relevant doc for your task:

| Task | Read this first |
|------|----------------|
| Adding a block | [docs/agent-knowledge/block-invariants.md](docs/agent-knowledge/block-invariants.md) |
| Adding a flow | [docs/agent-knowledge/flow-invariants.md](docs/agent-knowledge/flow-invariants.md) |
| Adding a connector | [docs/agent-knowledge/connector-invariants.md](docs/agent-knowledge/connector-invariants.md) |
| Writing tests | [docs/agent-knowledge/testing-standards.md](docs/agent-knowledge/testing-standards.md) |
| Reviewing code | [docs/agent-knowledge/grading-criteria.md](docs/agent-knowledge/grading-criteria.md) |
| Deciding to fix vs escalate | [docs/agent-knowledge/decision-rubric.md](docs/agent-knowledge/decision-rubric.md) |
| Checking quality status | [docs/agent-knowledge/QUALITY.md](docs/agent-knowledge/QUALITY.md) |
| All principles | [docs/agent-knowledge/core-principles.md](docs/agent-knowledge/core-principles.md) |

Full index: [docs/agent-knowledge/index.md](docs/agent-knowledge/index.md)

## Common Pitfalls

- `flow.set_model_config(model="...", api_key="...")` must be called before `generate()` for any flow containing LLM blocks
- Use `flow.dry_run(dataset)` to validate a pipeline end-to-end without making LLM calls
- `runtime_params` can be passed to `flow.generate(dataset, runtime_params={...})` to override block config at execution time
- LiteLLM reads standard env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) automatically
- The `FlowCheckpointer` enables resumable execution; pass `checkpoint_dir` to `generate()`

## CI Requirements

All PRs must pass:

| Check | Command | Workflow |
|-------|---------|----------|
| Conventional Commits | `commitlint` | commitlint.yml |
| Ruff formatting | `ruff format --check src/ tests/` | lint.yml |
| Ruff linting | `ruff check src/ tests/` | lint.yml |
| Type checking | `mypy src/sdg_hub` | lint.yml |
| Unit tests | `pytest tests/blocks tests/connectors tests/flow tests/utils` | test.yml |
| Structural tests | `pytest tests/structural/` | test.yml |
| Lock file sync | `uv lock --check` | lock.yml |
| Markdown linting | `markdownlint-cli2` | docs.yml |
| GitHub Actions lint | `actionlint` | actionlint.yml |

Commit prefixes: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`
```

- [ ] **Step 2: Verify line count is ~80-100**

Run: `wc -l CLAUDE.md`
Expected: Between 75 and 105 lines.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: restructure CLAUDE.md as knowledge base table of contents"
```

---

### Task 5: Create PR and Issue Templates

**Files:**
- Create: `.github/PULL_REQUEST_TEMPLATE.md`
- Create: `.github/ISSUE_TEMPLATE/bug_report.md`
- Create: `.github/ISSUE_TEMPLATE/feature_request.md`
- Create: `.github/ISSUE_TEMPLATE/new_block.md`
- Create: `.github/ISSUE_TEMPLATE/config.yml`

- [ ] **Step 1: Create `.github/PULL_REQUEST_TEMPLATE.md`**

```markdown
## Summary
<!-- 1-3 sentences: what changed and why -->

## Changes
<!-- Bulleted list of files changed and what was modified -->

## Test Plan
<!-- How was this tested? What evidence exists? -->

## Checklist
- [ ] Tests pass (`uv run pytest`)
- [ ] Structural tests pass (`uv run pytest tests/structural/`)
- [ ] Lint clean (`uv run ruff check src/ tests/`)
- [ ] Types clean (`uv run mypy src/sdg_hub`)
- [ ] Docs updated if public API changed
- [ ] No new lint warnings introduced

## Agent Metadata
<!-- Filled by agent PRs -->
- **Agent:** <!-- claude-code | cursor | human -->
- **Confidence:** <!-- high | medium | low -->
- **Auto-merge eligible:** <!-- yes | no -->
```

- [ ] **Step 2: Create `.github/ISSUE_TEMPLATE/bug_report.md`**

```markdown
---
name: Bug Report
about: Report a bug in SDG Hub
labels: bug
---

## Bug Description
<!-- What happened? What did you expect? -->

## Reproduction Steps
1.
2.
3.

## Environment
- Python version:
- SDG Hub version:
- OS:

## Error Output
```
<!-- Paste error traceback here -->
```

## Relevant Code
```python
# Minimal code that reproduces the issue
```
```

- [ ] **Step 3: Create `.github/ISSUE_TEMPLATE/feature_request.md`**

```markdown
---
name: Feature Request
about: Suggest a new feature or improvement
labels: enhancement
---

## Feature Description
<!-- What do you want to add or change? -->

## Use Case
<!-- Why is this needed? What problem does it solve? -->

## Proposed Approach
<!-- How should this be implemented? (optional) -->

## Acceptance Criteria
<!-- How do we know this is done? -->
- [ ]
- [ ]
```

- [ ] **Step 4: Create `.github/ISSUE_TEMPLATE/new_block.md`**

```markdown
---
name: New Block
about: Request a new data processing block
labels: enhancement, block
---

## Block Description
<!-- What does this block do? -->

## Category
<!-- Which category? llm, parsing, transform, filtering, agent, mcp, code -->

## Input/Output
- **Input columns:**
- **Output columns:**
- **Configuration fields:**

## Example Usage
```yaml
- name: my_step
  block_type: proposed_block_name
  config:
    # expected configuration
```

## Acceptance Criteria
- [ ] Block registered with `@BlockRegistry.register()`
- [ ] `generate()` method implemented
- [ ] `input_cols` and `output_cols` declared
- [ ] Test file at `tests/blocks/{category}/test_{name}.py`
- [ ] Tests cover success and error cases
```

- [ ] **Step 5: Create `.github/ISSUE_TEMPLATE/config.yml`**

```yaml
blank_issues_enabled: true
contact_links:
  - name: Documentation
    url: https://sdg-hub.github.io/
    about: Check the documentation before opening an issue
```

- [ ] **Step 6: Commit**

```bash
git add .github/PULL_REQUEST_TEMPLATE.md .github/ISSUE_TEMPLATE/
git commit -m "docs: add PR and issue templates for agent-structured output"
```

---

### Task 6: Create Execution Plan Directory and Environment Example

**Files:**
- Create: `docs/exec-plans/active/.gitkeep`
- Create: `docs/exec-plans/completed/.gitkeep`
- Create: `.env.example`

- [ ] **Step 1: Create execution plan directories**

```bash
mkdir -p docs/exec-plans/active docs/exec-plans/completed
touch docs/exec-plans/active/.gitkeep docs/exec-plans/completed/.gitkeep
```

- [ ] **Step 2: Create `.env.example`**

```bash
# SDG Hub Environment Variables

# LLM API keys (LiteLLM reads these automatically)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Optional: specify default model
# SDG_HUB_DEFAULT_MODEL=gpt-4o

# Optional: agent framework connector URLs
# LANGFLOW_API_URL=http://localhost:7860
# LANGGRAPH_API_URL=http://localhost:8000
```

- [ ] **Step 3: Commit**

```bash
git add docs/exec-plans/ .env.example
git commit -m "docs: add execution plan directories and .env.example"
```

---

### Task 7: Legacy Cleanup

**Files:**
- Delete: `.isort.cfg`
- Delete: `.pylintrc`

- [ ] **Step 1: Remove legacy config files**

```bash
git rm .isort.cfg .pylintrc
```

- [ ] **Step 2: Verify ruff config still handles isort**

Run: `uv run ruff check --select I src/sdg_hub/ --quiet`
Expected: No errors (ruff's isort rule works without `.isort.cfg`).

- [ ] **Step 3: Clean stale `__pycache__` directories**

```bash
find tests/ -type d -name __pycache__ | while read dir; do
  parent=$(dirname "$dir")
  # Check if the parent has any .py files — if not, the __pycache__ is stale
  if [ -z "$(find "$parent" -maxdepth 1 -name '*.py' -print -quit)" ]; then
    echo "Removing stale: $dir"
    rm -rf "$dir"
  fi
done
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove legacy .isort.cfg and .pylintrc, clean stale __pycache__"
```

---

### Task 8: Verify Phase 1 Completeness

- [ ] **Step 1: Verify all knowledge base files exist**

Run: `ls docs/agent-knowledge/`
Expected: 10 files listed (index.md, core-principles.md, block-invariants.md, flow-invariants.md, connector-invariants.md, testing-standards.md, grading-criteria.md, decision-rubric.md, QUALITY.md, tech-debt-tracker.md).

- [ ] **Step 2: Verify ARCHITECTURE.md exists**

Run: `test -f ARCHITECTURE.md && echo "OK" || echo "MISSING"`
Expected: `OK`

- [ ] **Step 3: Verify CLAUDE.md is the right size**

Run: `wc -l CLAUDE.md`
Expected: Between 75 and 105 lines.

- [ ] **Step 4: Verify templates exist**

Run: `ls .github/PULL_REQUEST_TEMPLATE.md .github/ISSUE_TEMPLATE/`
Expected: PR template and 4 issue template files listed.

- [ ] **Step 5: Verify legacy files are gone**

Run: `test ! -f .isort.cfg && test ! -f .pylintrc && echo "OK" || echo "STILL EXISTS"`
Expected: `OK`

- [ ] **Step 6: Verify all cross-links in the knowledge base resolve**

Run: `grep -roh '\[.*\](.*\.md)' docs/agent-knowledge/ | grep -oP '\(.*?\)' | tr -d '()' | while read link; do test -f "docs/agent-knowledge/$link" || echo "BROKEN: $link"; done`
Expected: No output (all links resolve).

- [ ] **Step 7: Run existing CI checks to ensure nothing is broken**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/sdg_hub && uv run pytest tests/blocks tests/connectors tests/flow tests/utils -m "not (examples or slow)" -x -q`
Expected: All pass. Knowledge base changes are docs-only — no code was modified.
