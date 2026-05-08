# SPDX-License-Identifier: Apache-2.0
"""Verify every registered block has a corresponding test file.

Blocks without a test file are reported via xfail so the suite passes
but the gap remains visible in pytest output.
"""

from pathlib import Path

import pytest

from sdg_hub.core.blocks.registry import BlockRegistry

# Root of the block test tree
_TESTS_BLOCKS_DIR = Path(__file__).resolve().parent.parent / "blocks"

# Map source categories to test directory names when they differ
_CATEGORY_DIR_ALIASES: dict[str, list[str]] = {
    "mcp": ["mcp_blocks", "mcp"],
}


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case, handling acronyms like LLM, MCP, JSON."""
    import re

    # Insert underscore between an uppercase acronym and a capitalized word
    # e.g. LLMChat -> LLM_Chat, JSONParser -> JSON_Parser, MCPAgent -> MCP_Agent
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    # Insert underscore between a lowercase/digit and an uppercase letter
    # e.g. column_Value -> column_Value (already handled), chatBlock -> chat_Block
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def _test_file_exists(block_name: str, category: str) -> bool:
    """Check whether a test file exists for *block_name* in its category dir.

    Checks several common naming conventions:
      - test_{name}.py
      - test_{name}_block.py
      - test_{name_lower}.py  (condensed lowercase)
    across both the canonical category dir and any known aliases.
    """
    name_lower = block_name.lower()
    snake = _camel_to_snake(block_name)

    candidate_names = {
        f"test_{name_lower}.py",
        f"test_{snake}.py",
        f"test_{name_lower}_block.py",
        f"test_{snake}_block.py",
    }
    # Also handle names that already end with "block" or "_block"
    for suffix in ("_block", "block"):
        for variant in (snake, name_lower):
            if variant.endswith(suffix):
                stem = variant[: -len(suffix)].rstrip("_")
                candidate_names.add(f"test_{stem}.py")
                candidate_names.add(f"test_{stem}_block.py")

    # Directories to search
    dirs_to_check = [category]
    dirs_to_check.extend(_CATEGORY_DIR_ALIASES.get(category, []))

    for dir_name in dirs_to_check:
        category_dir = _TESTS_BLOCKS_DIR / dir_name
        if not category_dir.is_dir():
            continue
        existing_files = {f.name for f in category_dir.iterdir() if f.is_file()}
        if candidate_names & existing_files:
            return True
        # Fallback: check if any test file in the directory imports the class
        for f in category_dir.glob("test_*.py"):
            if block_name in f.read_text(encoding="utf-8"):
                return True
    return False


def _collect_blocks_without_tests() -> list[tuple[str, str]]:
    """Return list of (block_name, category) pairs that lack a test file."""
    missing: list[tuple[str, str]] = []
    for block_name in BlockRegistry.list_blocks():
        meta = BlockRegistry._metadata[block_name]
        if not _test_file_exists(block_name, meta.category):
            missing.append((block_name, meta.category))
    return missing


# Discover missing tests at module import time so parametrize works
_MISSING = _collect_blocks_without_tests()
_ALL_BLOCKS = sorted(BlockRegistry.list_blocks())


@pytest.mark.parametrize("block_name", _ALL_BLOCKS)
def test_block_has_test_file(block_name: str) -> None:
    """Each registered block should have a test file under tests/blocks/."""
    meta = BlockRegistry._metadata[block_name]
    missing_names = {name for name, _ in _MISSING}

    if block_name in missing_names:
        pytest.xfail(
            f"No test file found for block '{block_name}' "
            f"(category={meta.category}). "
            f"Expected in tests/blocks/{meta.category}/test_*.py"
        )

    assert _test_file_exists(block_name, meta.category)
