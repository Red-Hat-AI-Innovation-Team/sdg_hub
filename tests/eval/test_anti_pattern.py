# SPDX-License-Identifier: Apache-2.0
"""Tests for eval/anti_pattern.py -- similarity checking for reverted diffs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import sys

# eval/ is not a package; add it to sys.path so we can import it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eval"))

from anti_pattern import SIMILARITY_THRESHOLD, _load_reverted_diffs  # noqa: E402


def _write_diff(archive_dir: Path, pr_number: str, content: str) -> None:
    """Write a fake diff file into the archive directory."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / f"{pr_number}.diff").write_text(content)


def _check_similarity(new_diff: str, archive_dir: Path) -> int:
    """Re-implement the core check logic (without subprocess calls).

    Returns 1 if blocked, 0 if OK.  This mirrors cmd_check() but takes
    the diff as a string instead of calling git.
    """
    import difflib

    with patch("anti_pattern.REVERTED_DIFFS_DIR", archive_dir):
        reverted_diffs = _load_reverted_diffs()

    if not reverted_diffs:
        return 0

    for _pr_number, reverted_diff in reverted_diffs.items():
        ratio = difflib.SequenceMatcher(None, new_diff, reverted_diff).ratio()
        if ratio > SIMILARITY_THRESHOLD:
            return 1

    return 0


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


class TestAntiPattern:
    def test_check_no_reverted_diffs(self, tmp_path: Path) -> None:
        """Empty archive dir -> returns 0 (pass)."""
        archive = tmp_path / "reverted_diffs"
        # Don't create the directory -- simulates no history.

        result = _check_similarity("diff --git a/foo.py b/foo.py\n+x = 1", archive)

        assert result == 0

    def test_check_below_threshold(self, tmp_path: Path) -> None:
        """Diff with <60% similarity -> returns 0."""
        archive = tmp_path / "reverted_diffs"
        _write_diff(
            archive,
            "100",
            "diff --git a/old.py b/old.py\n-old_function()\n+completely_different()\n",
        )

        new_diff = (
            "diff --git a/new.py b/new.py\n"
            "+brand_new_unrelated_code_that_has_no_similarity()\n"
            "+with_additional_unique_lines()\n"
            "+even_more_unique_content()\n"
        )
        result = _check_similarity(new_diff, archive)

        assert result == 0

    def test_check_above_threshold(self, tmp_path: Path) -> None:
        """Diff with >60% similarity -> returns 1 (block)."""
        archive = tmp_path / "reverted_diffs"
        reverted = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,5 +1,5 @@\n"
            " import os\n"
            "-def old_handler():\n"
            "+def new_handler():\n"
            "     return True\n"
        )
        _write_diff(archive, "200", reverted)

        # Nearly identical diff -- just different PR metadata
        new_diff = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,5 +1,5 @@\n"
            " import os\n"
            "-def old_handler():\n"
            "+def new_handler():\n"
            "     return True\n"
        )
        result = _check_similarity(new_diff, archive)

        assert result == 1

    def test_check_at_exact_threshold(self, tmp_path: Path) -> None:
        """Verify boundary behavior -- ratio == 0.6 should NOT block.

        The code uses ``ratio > SIMILARITY_THRESHOLD`` (strict greater-than),
        so exactly 0.6 is allowed.
        """
        assert SIMILARITY_THRESHOLD == 0.6  # guard against constant changes

        archive = tmp_path / "reverted_diffs"
        # Use the same helper but patch SequenceMatcher to return exactly 0.6.
        _write_diff(archive, "300", "any content")

        with patch("difflib.SequenceMatcher") as mock_sm:
            mock_sm.return_value.ratio.return_value = 0.6
            result = _check_similarity("any new content", archive)

        # Exactly at threshold -> NOT blocked (strict >)
        assert result == 0
