# Core Principles

Six golden rules that apply to every contribution to the SDG Hub codebase.

## 1. Prefer shared utility packages over hand-rolled helpers

Before writing a new helper function, check `src/sdg_hub/core/utils/` first.
Utilities for logging (`logger_config`), error handling (`error_handling`),
and other common operations already exist there. Duplicating them creates
maintenance burden and inconsistency.

## 2. Validate data shapes at boundaries

Use Pydantic models to validate inputs and outputs at block and connector
boundaries. Do not pass raw dicts between components -- define a model,
validate against it, and let Pydantic surface errors early.

## 3. Every block must be registered, tested, and documented

A block is not done until it is:

- Registered with `@BlockRegistry.register(name, category, description)`
- Tested with a file at `tests/blocks/{category}/test_{name}.py`
- Documented with a docstring describing its purpose, parameters, and behavior

See [block-invariants.md](block-invariants.md) for the full checklist.

## 4. Flow YAMLs must have metadata

Every flow YAML must include a `metadata` section with at least `name`,
`author`, `description`, and `tags` (one or more). Flows without metadata
are undiscoverable and unmaintainable.

See [flow-invariants.md](flow-invariants.md) for the full schema.

## 5. Structured logging only -- no raw `print()`

Use the project logger (`from sdg_hub.core.utils.logger_config import
setup_logger`) for all output. Raw `print()` statements are caught and
rejected by ruff rule `T201`. This keeps output machine-parseable and
filterable by log level.

## 6. Tests must assert meaningful output

Tests that only check "something was returned" (e.g., `assert result is not
None`) provide no safety net. Every test must assert specific values,
properties, shapes, or behaviors that would catch real regressions.

See [testing-standards.md](testing-standards.md) for detailed guidance.
