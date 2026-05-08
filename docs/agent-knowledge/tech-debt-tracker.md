# Tech Debt Tracker

Known technical debt organized by priority. Items are removed as they are
resolved; new items are added as they are discovered.

## High Priority

- **Legacy .isort.cfg and .pylintrc redundant with ruff.** These config files
  are superseded by ruff which handles both linting and import sorting. Being
  removed in Phase 1.

- **Stale \_\_pycache\_\_ directories for deleted test modules.** Leftover
  bytecode caches exist for test modules that have been removed:
  `tests/blocks/evaluation/`, `tests/blocks/utilblocks/`,
  `tests/blocks/column_ops/`. These should be cleaned up and added to
  `.gitignore` if not already excluded.

- **Two documentation systems (MkDocs + Next.js) creating drift risk.** Having
  two separate documentation builds means content can diverge. Consolidate to
  a single system or establish a clear ownership boundary between them.

## Medium Priority

- **No top-level conftest.py with shared test fixtures.** Common test helpers
  (mock LLM clients, sample datasets, temporary flow YAML) are duplicated
  across test modules instead of shared via a root `conftest.py`.

- **No .env.example at project root.** New contributors have no reference for
  which environment variables are expected. Being added in Phase 1.

- **Some block categories missing in CLAUDE.md block category table.** The
  `blocks/code` category and any newly added categories are not listed in the
  orientation table, making them harder to discover.

## Low Priority

- **mypy config disables import-not-found and import-untyped errors.** This
  suppresses real type-checking issues from untyped or missing dependencies.
  Ideally these should be re-enabled with targeted `type: ignore` comments
  where needed.

- **5 legacy files excluded from mypy checking.** These exclusions accumulate
  untyped code that is invisible to CI. Each should be brought into compliance
  and removed from the exclusion list.
