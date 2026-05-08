# Phase 2: Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce SDG Hub's architectural invariants mechanically — via ruff rules and pytest structural tests — so agents get actionable error messages when they violate conventions.

**Architecture:** Enable ruff rules T201 (no print), add `tests/structural/` with 4 test files that verify block registration, test coverage, import architecture, and flow schema compliance. Add a CI workflow for knowledge base link validation. Create `eval/score.py` for three-tier composite scoring.

**Tech Stack:** Python, pytest, ruff, mypy, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-05-07-agent-maintained-sdg-hub-design.md` — Sections 4.1 through 4.5.

---

## File Structure

```
# New files
tests/structural/__init__.py
tests/structural/test_block_registration.py     — Every BaseBlock subclass is registered
tests/structural/test_block_coverage.py          — Every registered block has a test file
tests/structural/test_architecture.py            — Import direction rules, file size limits
tests/structural/test_flow_schemas.py            — All flow YAMLs have valid metadata
.github/workflows/knowledge-validation.yml       — CI for knowledge base freshness
eval/__init__.py
eval/score.py                                    — Three-tier composite scoring script

# Modified files
pyproject.toml                                   — Add ruff rule T201
```

---

### Task 1: Enable ruff rule T201 (ban print)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add T201 to ruff select and add per-file-ignores for tests**

In `pyproject.toml` under `[tool.ruff.lint]`, add `"T20"` (flake8-print) to the `select` list. Add a `[tool.ruff.lint.per-file-ignores]` section to allow print in tests and examples:

```toml
select = [
    "E",   # pycodestyle
    "F",   # Pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "Q",   # flake8-quotes
    "T20", # flake8-print (bans print())
    "TID", # flake8-tidy-imports
]
```

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**" = ["T20"]
"examples/**" = ["T20"]
```

- [ ] **Step 2: Check for existing print() violations in src/**

Run: `uv run ruff check --select T20 src/sdg_hub/ 2>&1 | head -30`

If there are violations, they need to be fixed (replace print with logger calls) or the files need per-file-ignores. The spec says print was already largely removed in PR #700.

- [ ] **Step 3: Fix any remaining violations or add targeted ignores**

For each violation, replace `print(...)` with `logger.info(...)` using the existing `setup_logger` pattern. If the print is in a CLI display function (like `discover_blocks()`), add a `# noqa: T201` comment.

- [ ] **Step 4: Verify ruff passes**

Run: `uv run ruff check src/ tests/`
Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/
git commit -m "build: enable ruff T201 rule to ban print() in src/"
```

---

### Task 2: Create structural test — block registration

**Files:**
- Create: `tests/structural/__init__.py`
- Create: `tests/structural/test_block_registration.py`

- [ ] **Step 1: Create `tests/structural/__init__.py`**

Empty file.

- [ ] **Step 2: Write `tests/structural/test_block_registration.py`**

```python
"""Verify every BaseBlock subclass is registered in BlockRegistry."""

import importlib
import inspect
import pkgutil

from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry


def _discover_block_subclasses():
    """Import all block modules and return BaseBlock subclasses."""
    import sdg_hub.core.blocks as blocks_pkg

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        blocks_pkg.__path__, prefix=blocks_pkg.__name__ + "."
    ):
        try:
            importlib.import_module(modname)
        except ImportError:
            continue

    return [
        cls
        for cls in BaseBlock.__subclasses__()
        if not inspect.isabstract(cls) and cls.__module__.startswith("sdg_hub.core.blocks")
    ]


def test_all_block_subclasses_are_registered():
    """Every concrete BaseBlock subclass must be registered in BlockRegistry."""
    registered_classes = set()
    for name in BlockRegistry.list_blocks():
        block_cls = BlockRegistry._get(name)
        registered_classes.add(block_cls)

    subclasses = _discover_block_subclasses()
    unregistered = [
        cls for cls in subclasses if cls not in registered_classes
    ]

    assert not unregistered, (
        f"Found {len(unregistered)} unregistered BaseBlock subclass(es): "
        f"{[cls.__name__ for cls in unregistered]}. "
        f"Register each with @BlockRegistry.register() following "
        f"docs/agent-knowledge/block-invariants.md"
    )
```

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/structural/test_block_registration.py -v`
Expected: Either PASS (all registered) or FAIL with a clear message naming unregistered blocks.

- [ ] **Step 4: If the test fails, investigate the unregistered block**

The codebase has 21 BaseBlock subclasses but only 20 registrations. Find the unregistered one and either register it or exclude it if it's intentionally abstract/internal.

- [ ] **Step 5: Commit**

```bash
git add tests/structural/
git commit -m "test: add structural test verifying all blocks are registered"
```

---

### Task 3: Create structural test — block test coverage

**Files:**
- Create: `tests/structural/test_block_coverage.py`

- [ ] **Step 1: Write `tests/structural/test_block_coverage.py`**

```python
"""Verify every registered block has a corresponding test file."""

from pathlib import Path

from sdg_hub.core.blocks.registry import BlockRegistry


def _get_block_category(block_name: str) -> str:
    """Get the category of a registered block."""
    block_cls = BlockRegistry._get(block_name)
    module = block_cls.__module__
    parts = module.split(".")
    # sdg_hub.core.blocks.<category>.<module_name>
    if len(parts) >= 5:
        return parts[3]
    return "unknown"


def _normalize_test_path(block_name: str, category: str) -> list[Path]:
    """Return possible test file paths for a block."""
    tests_root = Path("tests/blocks")
    # Handle special directory naming (mcp -> mcp_blocks in tests)
    category_dirs = [category]
    if category == "mcp":
        category_dirs.append("mcp_blocks")
    
    paths = []
    for cat_dir in category_dirs:
        paths.append(tests_root / cat_dir / f"test_{block_name}.py")
        # Some tests use the class module name instead of registration name
        paths.append(tests_root / cat_dir / f"test_{block_name}_block.py")
    return paths


def test_every_registered_block_has_tests():
    """Every registered block must have a corresponding test file."""
    missing = []
    for block_name in BlockRegistry.list_blocks():
        category = _get_block_category(block_name)
        possible_paths = _normalize_test_path(block_name, category)
        if not any(p.exists() for p in possible_paths):
            missing.append(
                f"  - Block '{block_name}' (category: {category}): "
                f"expected test at {possible_paths[0]}"
            )

    assert not missing, (
        f"Found {len(missing)} registered block(s) without test files:\n"
        + "\n".join(missing)
        + "\n\nCreate test files following docs/agent-knowledge/testing-standards.md"
    )
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/structural/test_block_coverage.py -v`

- [ ] **Step 3: If tests fail, note which blocks are missing tests but don't create them now**

This is expected — some blocks may not have test files yet. The structural test documents the gap. If there are failures, add a `pytest.mark.xfail` with a reason so the test suite passes but the gap is visible.

- [ ] **Step 4: Commit**

```bash
git add tests/structural/test_block_coverage.py
git commit -m "test: add structural test verifying all blocks have test files"
```

---

### Task 4: Create structural test — flow schema validation

**Files:**
- Create: `tests/structural/test_flow_schemas.py`

- [ ] **Step 1: Write `tests/structural/test_flow_schemas.py`**

```python
"""Verify all built-in flow YAMLs have valid metadata."""

import pytest
import yaml
from pathlib import Path


FLOWS_DIR = Path("src/sdg_hub/flows")
REQUIRED_METADATA_FIELDS = {"name", "author", "description", "tags"}


def _discover_flow_yamls():
    """Find all YAML files in the flows directory."""
    if not FLOWS_DIR.exists():
        return []
    return list(FLOWS_DIR.rglob("*.yaml")) + list(FLOWS_DIR.rglob("*.yml"))


def _load_flow_yaml(path: Path) -> dict | None:
    """Load a YAML file, return None if it's not a flow."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return None
    if "metadata" in data and "blocks" in data:
        return data
    return None


@pytest.mark.parametrize(
    "flow_path",
    _discover_flow_yamls(),
    ids=lambda p: str(p.relative_to(FLOWS_DIR)),
)
def test_flow_yaml_has_required_metadata(flow_path):
    """Every flow YAML must have metadata with name, author, description, tags."""
    flow = _load_flow_yaml(flow_path)
    if flow is None:
        pytest.skip(f"{flow_path.name} is not a flow YAML (no metadata+blocks)")

    metadata = flow.get("metadata", {})
    missing = REQUIRED_METADATA_FIELDS - set(metadata.keys())
    assert not missing, (
        f"Flow {flow_path.relative_to(FLOWS_DIR)} is missing metadata fields: "
        f"{missing}. See docs/agent-knowledge/flow-invariants.md"
    )

    tags = metadata.get("tags", [])
    assert isinstance(tags, list) and len(tags) > 0, (
        f"Flow {flow_path.relative_to(FLOWS_DIR)} must have at least one tag. "
        f"See docs/agent-knowledge/flow-invariants.md"
    )
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/structural/test_flow_schemas.py -v 2>&1 | tail -20`

- [ ] **Step 3: Handle failures**

Some existing flows may be missing metadata fields. For known gaps, mark tests as `xfail` or fix the flow YAMLs. The structural test documents the gap either way.

- [ ] **Step 4: Commit**

```bash
git add tests/structural/test_flow_schemas.py
git commit -m "test: add structural test verifying flow YAML metadata"
```

---

### Task 5: Create structural test — architecture (import rules + file size)

**Files:**
- Create: `tests/structural/test_architecture.py`

- [ ] **Step 1: Write `tests/structural/test_architecture.py`**

```python
"""Verify architectural invariants: file size limits and no cross-implementation imports."""

from pathlib import Path

BLOCKS_DIR = Path("src/sdg_hub/core/blocks")
CONNECTORS_DIR = Path("src/sdg_hub/core/connectors")
SRC_DIR = Path("src/sdg_hub")
MAX_FILE_LINES = 500

# Implementation directories — files in these dirs must not import from
# sibling implementation dirs
BLOCK_IMPL_DIRS = [
    d for d in BLOCKS_DIR.iterdir()
    if d.is_dir() and d.name not in ("__pycache__",)
]


def test_no_python_file_exceeds_line_limit():
    """No Python file in src/ should exceed the line limit."""
    oversized = []
    for py_file in SRC_DIR.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        line_count = len(py_file.read_text().splitlines())
        if line_count > MAX_FILE_LINES:
            oversized.append(f"  - {py_file}: {line_count} lines (max {MAX_FILE_LINES})")

    assert not oversized, (
        f"Found {len(oversized)} file(s) exceeding {MAX_FILE_LINES} lines:\n"
        + "\n".join(oversized)
        + "\nConsider splitting large files into smaller, focused modules."
    )


def test_block_implementations_do_not_cross_import():
    """Block implementation modules must not import from sibling implementation dirs.

    e.g., blocks/llm/ must not import from blocks/parsing/
    Only imports from blocks/base.py, blocks/registry.py, and core/utils/ are allowed.
    """
    violations = []
    for impl_dir in BLOCK_IMPL_DIRS:
        if not impl_dir.is_dir():
            continue
        sibling_names = [
            d.name for d in BLOCK_IMPL_DIRS
            if d != impl_dir and d.is_dir()
        ]
        for py_file in impl_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            content = py_file.read_text()
            for sibling in sibling_names:
                import_pattern = f"from sdg_hub.core.blocks.{sibling}"
                alt_pattern = f"import sdg_hub.core.blocks.{sibling}"
                if import_pattern in content or alt_pattern in content:
                    violations.append(
                        f"  - {py_file} imports from blocks/{sibling}/"
                    )

    assert not violations, (
        f"Found {len(violations)} cross-implementation import(s):\n"
        + "\n".join(violations)
        + "\nBlock implementations must not import from sibling implementation dirs. "
        + "See ARCHITECTURE.md for dependency direction rules."
    )
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/structural/test_architecture.py -v`

- [ ] **Step 3: Handle failures**

If files exceed 500 lines, note them but use `xfail` rather than fixing them now. If cross-imports exist, investigate — they may be legitimate (shared types) or violations.

- [ ] **Step 4: Commit**

```bash
git add tests/structural/test_architecture.py
git commit -m "test: add structural test for file size limits and import rules"
```

---

### Task 6: Create knowledge base validation CI workflow

**Files:**
- Create: `.github/workflows/knowledge-validation.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: Knowledge Base Validation

on:
  push:
    branches: [main]
    paths: ['docs/agent-knowledge/**']
  pull_request:
    branches: [main]
    paths: ['docs/agent-knowledge/**']

permissions:
  contents: read

jobs:
  validate-knowledge-base:
    runs-on: ubuntu-latest
    steps:
      - uses: step-security/harden-runner@v2
        with:
          egress-policy: audit

      - uses: actions/checkout@v4

      - name: Check all cross-links resolve
        run: |
          errors=0
          for file in docs/agent-knowledge/*.md; do
            # Extract markdown links to .md files
            grep -oP '\[.*?\]\(\K[^)]+\.md' "$file" 2>/dev/null | while read link; do
              dir=$(dirname "$file")
              if [ ! -f "$dir/$link" ] && [ ! -f "$link" ]; then
                echo "::error file=$file::Broken link: $link"
                errors=$((errors + 1))
              fi
            done
          done
          if [ $errors -gt 0 ]; then
            echo "Found $errors broken link(s) in knowledge base"
            exit 1
          fi

      - name: Check index.md lists all knowledge base files
        run: |
          # Get all .md files except index.md
          all_files=$(ls docs/agent-knowledge/*.md | grep -v index.md | xargs -I{} basename {})
          missing=0
          for file in $all_files; do
            if ! grep -q "$file" docs/agent-knowledge/index.md; then
              echo "::warning file=docs/agent-knowledge/index.md::$file not listed in index"
              missing=$((missing + 1))
            fi
          done
          if [ $missing -gt 0 ]; then
            echo "::warning::$missing file(s) not listed in index.md"
          fi
```

- [ ] **Step 2: Validate the workflow syntax**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/knowledge-validation.yml'))"`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/knowledge-validation.yml
git commit -m "ci: add knowledge base validation workflow"
```

---

### Task 7: Create three-tier composite scoring script

**Files:**
- Create: `eval/__init__.py`
- Create: `eval/score.py`

- [ ] **Step 1: Create `eval/__init__.py`**

Empty file.

- [ ] **Step 2: Write `eval/score.py`**

A Python script that runs the three-tier evaluation and outputs `{"score": 0.0-1.0}`:

```python
"""Three-tier composite scoring for SDG Hub.

Tiers:
  1. Hygiene (0.30) — tests, lint, type check, coverage, structural, commit format
  2. Growth (0.20) — capability surface, test diversity, doc completeness
  3. Project (0.50) — flow regression, block correctness, connector health

Usage:
  uv run python eval/score.py
  uv run python eval/score.py --json     # Machine-readable output
"""

import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: str, timeout: int = 120) -> tuple[int, str]:
    """Run a command, return (exit_code, output)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout + result.stderr


def score_hygiene() -> dict[str, float]:
    """Tier 1: Hygiene checks."""
    scores = {}

    # Tests pass?
    rc, _ = _run("uv run pytest tests/blocks tests/connectors tests/flow tests/utils "
                  "-m 'not (examples or slow)' -x -q")
    scores["tests"] = 1.0 if rc == 0 else 0.0

    # Lint clean?
    rc, _ = _run("uv run ruff check src/ tests/ --quiet")
    scores["lint"] = 1.0 if rc == 0 else 0.0

    # Type check?
    rc, _ = _run("uv run mypy src/sdg_hub --no-error-summary")
    scores["type_check"] = 1.0 if rc == 0 else 0.0

    # Structural tests?
    rc, _ = _run("uv run pytest tests/structural/ -x -q")
    scores["structural"] = 1.0 if rc == 0 else 0.0

    weights = {"tests": 0.35, "lint": 0.20, "type_check": 0.15, "structural": 0.30}
    weighted = sum(scores[k] * weights[k] for k in scores)
    return {"dimensions": scores, "weighted": weighted}


def score_growth() -> dict[str, float]:
    """Tier 2: Growth indicators."""
    scores = {}

    # Count registered blocks + flows + connectors
    try:
        rc, output = _run(
            "uv run python -c \""
            "from sdg_hub.core.blocks.registry import BlockRegistry; "
            "from sdg_hub.core.flow.registry import FlowRegistry; "
            "from sdg_hub.core.connectors.registry import ConnectorRegistry; "
            "import sdg_hub.core.blocks.llm, sdg_hub.core.blocks.parsing, "
            "sdg_hub.core.blocks.transform, sdg_hub.core.blocks.filtering, "
            "sdg_hub.core.blocks.agent, sdg_hub.core.blocks.mcp; "
            "print(len(BlockRegistry.list_blocks()), len(FlowRegistry._registry), "
            "len(ConnectorRegistry.list_all()))\""
        )
        if rc == 0:
            parts = output.strip().split()
            total = sum(int(p) for p in parts)
            scores["capability_surface"] = min(1.0, total / 30.0)
        else:
            scores["capability_surface"] = 0.0
    except Exception:
        scores["capability_surface"] = 0.0

    # Test directory coverage
    test_dirs = ["blocks", "connectors", "flow", "utils"]
    existing = sum(1 for d in test_dirs if Path(f"tests/{d}").exists())
    scores["test_diversity"] = existing / len(test_dirs)

    weights = {"capability_surface": 0.60, "test_diversity": 0.40}
    weighted = sum(scores[k] * weights[k] for k in scores)
    return {"dimensions": scores, "weighted": weighted}


def score_project() -> dict[str, float]:
    """Tier 3: Project-specific quality."""
    scores = {}

    # Block tests pass?
    rc, _ = _run("uv run pytest tests/blocks/ -x -q -m 'not (examples or slow)'")
    scores["block_correctness"] = 1.0 if rc == 0 else 0.0

    # Flow tests pass?
    rc, _ = _run("uv run pytest tests/flow/ -x -q")
    scores["flow_regression"] = 1.0 if rc == 0 else 0.0

    # Connector tests pass?
    rc, _ = _run("uv run pytest tests/connectors/ -x -q")
    scores["connector_health"] = 1.0 if rc == 0 else 0.0

    weights = {
        "block_correctness": 0.35,
        "flow_regression": 0.40,
        "connector_health": 0.25,
    }
    weighted = sum(scores[k] * weights[k] for k in scores)
    return {"dimensions": scores, "weighted": weighted}


def main():
    tier_weights = {"hygiene": 0.30, "growth": 0.20, "project": 0.50}

    hygiene = score_hygiene()
    growth = score_growth()
    project = score_project()

    composite = (
        hygiene["weighted"] * tier_weights["hygiene"]
        + growth["weighted"] * tier_weights["growth"]
        + project["weighted"] * tier_weights["project"]
    )

    result = {
        "score": round(composite, 4),
        "tiers": {
            "hygiene": hygiene,
            "growth": growth,
            "project": project,
        },
        "weights": tier_weights,
    }

    if "--json" in sys.argv:
        print(json.dumps(result, indent=2))
    else:
        print(f"Composite Score: {result['score']:.2%}")
        print(f"  Hygiene ({tier_weights['hygiene']:.0%}): {hygiene['weighted']:.2%}")
        for k, v in hygiene["dimensions"].items():
            print(f"    {k}: {'PASS' if v == 1.0 else 'FAIL'}")
        print(f"  Growth ({tier_weights['growth']:.0%}): {growth['weighted']:.2%}")
        for k, v in growth["dimensions"].items():
            print(f"    {k}: {v:.2%}")
        print(f"  Project ({tier_weights['project']:.0%}): {project['weighted']:.2%}")
        for k, v in project["dimensions"].items():
            print(f"    {k}: {'PASS' if v == 1.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the scoring script**

Run: `uv run python eval/score.py`
Expected: A composite score with per-tier breakdowns.

- [ ] **Step 4: Commit**

```bash
git add eval/
git commit -m "feat: add three-tier composite scoring script"
```

---

### Task 8: Verify Phase 2 completeness

- [ ] **Step 1: Verify structural tests exist and run**

Run: `uv run pytest tests/structural/ -v 2>&1 | tail -20`
Expected: All structural tests run (some may xfail for known gaps).

- [ ] **Step 2: Verify ruff T201 is enabled**

Run: `uv run ruff check --select T20 src/sdg_hub/ --quiet`
Expected: No violations (or only noqa-marked ones).

- [ ] **Step 3: Verify scoring script works**

Run: `uv run python eval/score.py --json`
Expected: JSON output with score between 0 and 1.

- [ ] **Step 4: Verify knowledge validation workflow exists**

Run: `test -f .github/workflows/knowledge-validation.yml && echo "OK"`
Expected: OK

- [ ] **Step 5: Run full test suite to ensure nothing is broken**

Run: `uv run pytest tests/blocks tests/connectors tests/flow tests/utils tests/structural -m "not (examples or slow)" -x -q`
Expected: All tests pass.
