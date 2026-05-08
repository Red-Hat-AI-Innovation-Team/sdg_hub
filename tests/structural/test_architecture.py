"""Verify architectural invariants for the sdg_hub codebase.

See ARCHITECTURE.md for the full rationale behind these constraints.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "sdg_hub"
BLOCKS_DIR = SRC_DIR / "core" / "blocks"

LINE_LIMIT = 500

# Files that currently exceed the line limit.  Tracked as xfail so the build
# stays green while the team works on splitting them.
KNOWN_OVERSIZED: set[str] = {
    "src/sdg_hub/core/flow/execution.py",
    "src/sdg_hub/core/utils/translation.py",
    "src/sdg_hub/core/blocks/llm/llm_chat_block.py",
    "src/sdg_hub/core/blocks/mcp/mcp_agent_block.py",
}


# ---------------------------------------------------------------------------
# test_no_python_file_exceeds_line_limit
# ---------------------------------------------------------------------------


def _collect_py_files() -> list[Path]:
    """Return all .py files under src/sdg_hub/, excluding __pycache__."""
    return sorted(p for p in SRC_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _py_file_id(path: Path) -> str:
    return str(path.relative_to(SRC_DIR.parent.parent))


PY_FILES = _collect_py_files()


@pytest.mark.parametrize("py_path", PY_FILES, ids=[_py_file_id(p) for p in PY_FILES])
def test_no_python_file_exceeds_line_limit(py_path: Path) -> None:
    """No source file should exceed 500 lines.  See ARCHITECTURE.md."""
    rel = str(py_path.relative_to(SRC_DIR.parent.parent))

    if rel in KNOWN_OVERSIZED:
        pytest.xfail(
            f"{rel} is a known oversized file ({LINE_LIMIT}+ lines). "
            f"See ARCHITECTURE.md for the refactoring plan."
        )

    line_count = len(py_path.read_text().splitlines())
    assert line_count <= LINE_LIMIT, (
        f"{rel} has {line_count} lines (limit {LINE_LIMIT}). "
        f"See ARCHITECTURE.md for guidelines on splitting large files."
    )


# ---------------------------------------------------------------------------
# test_block_implementations_do_not_cross_import
# ---------------------------------------------------------------------------

# Pattern matching imports from sdg_hub.core.blocks.<subdir>
_BLOCK_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+sdg_hub\.core\.blocks\.(\w+)")

# Imports from these modules are always allowed.
_ALLOWED_MODULES = {"base", "registry"}


def _collect_block_dirs() -> list[Path]:
    """Return implementation subdirectories under core/blocks/."""
    return sorted(
        d for d in BLOCKS_DIR.iterdir() if d.is_dir() and d.name != "__pycache__"
    )


def _block_cross_import_cases() -> list[tuple[Path, Path]]:
    """Return (block_dir, py_file) pairs for parametrization."""
    cases: list[tuple[Path, Path]] = []
    for block_dir in _collect_block_dirs():
        for py_file in sorted(block_dir.rglob("*.py")):
            if "__pycache__" not in py_file.parts:
                cases.append((block_dir, py_file))
    return cases


def _cross_import_id(item: tuple[Path, Path]) -> str:
    block_dir, py_file = item
    return f"{block_dir.name}/{py_file.name}"


CROSS_IMPORT_CASES = _block_cross_import_cases()


@pytest.mark.parametrize(
    "case", CROSS_IMPORT_CASES, ids=[_cross_import_id(c) for c in CROSS_IMPORT_CASES]
)
def test_block_implementations_do_not_cross_import(case: tuple[Path, Path]) -> None:
    """Block implementation dirs must not import from sibling dirs.

    Imports from blocks/base.py, blocks/registry.py, and core/utils/ are
    allowed.  See ARCHITECTURE.md for rationale.
    """
    block_dir, py_file = case
    own_dir_name = block_dir.name
    content = py_file.read_text()

    violations: list[str] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        match = _BLOCK_IMPORT_RE.match(line)
        if match:
            imported_module = match.group(1)
            if imported_module in _ALLOWED_MODULES:
                continue
            if imported_module == own_dir_name:
                continue
            violations.append(
                f"  line {lineno}: {line.strip()}  (imports from {imported_module}/)"
            )

    assert not violations, (
        f"{block_dir.name}/{py_file.name} has cross-imports from sibling block dirs:\n"
        + "\n".join(violations)
        + "\nSee ARCHITECTURE.md for the module dependency rules."
    )
