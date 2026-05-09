# SPDX-License-Identifier: Apache-2.0
"""Tests for eval/ace/playbook.py -- self-evolving behavioral playbooks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import sys

import pytest

# eval/ is not a package; add it to sys.path so we can import it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eval"))

from ace.playbook import (  # noqa: E402
    PlaybookEntry,
    add_entry,
    curate,
    load_playbook,
    record_outcome,
    save_playbook,
)


@pytest.fixture()
def playbook_dir(tmp_path: Path):
    """Patch PLAYBOOKS_DIR to a temp directory for test isolation."""
    pb_dir = tmp_path / "playbooks"
    pb_dir.mkdir()
    with (
        patch("ace.playbook.PLAYBOOKS_DIR", pb_dir),
        patch("ace.playbook._playbook_path", lambda role: pb_dir / f"{role}.md"),
    ):
        yield pb_dir


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


class TestPlaybook:
    def test_save_and_load_roundtrip(self, playbook_dir: Path) -> None:
        entries = [
            PlaybookEntry(
                id="test-0001",
                category="DO",
                rule="Always write tests",
                helpful=5,
                harmful=1,
                created_at="2026-01-01T00:00:00+00:00",
            ),
            PlaybookEntry(
                id="test-0002",
                category="DONT",
                rule="Never skip CI",
                helpful=3,
                harmful=0,
                created_at="2026-01-02T00:00:00+00:00",
            ),
        ]
        save_playbook("tester", entries)
        loaded = load_playbook("tester")

        assert len(loaded) == 2
        assert loaded[0].id == "test-0001"
        assert loaded[0].category == "DO"
        assert loaded[0].rule == "Always write tests"
        assert loaded[0].helpful == 5
        assert loaded[0].harmful == 1
        assert loaded[1].id == "test-0002"
        assert loaded[1].category == "DONT"
        assert loaded[1].rule == "Never skip CI"

    def test_add_entry(self, playbook_dir: Path) -> None:
        entry = add_entry("tester", "DO", "Run linting before commit")

        assert entry.helpful == 0
        assert entry.harmful == 0
        assert entry.category == "DO"
        assert entry.rule == "Run linting before commit"

        loaded = load_playbook("tester")
        assert len(loaded) == 1
        assert loaded[0].id == entry.id

    def test_record_outcome_kept(self, playbook_dir: Path) -> None:
        entry = add_entry("tester", "DO", "Check types")
        assert entry.helpful == 0

        record_outcome("tester", entry.id, "kept")

        loaded = load_playbook("tester")
        assert loaded[0].helpful == 1
        assert loaded[0].harmful == 0

    def test_record_outcome_reverted(self, playbook_dir: Path) -> None:
        entry = add_entry("tester", "DONT", "Skip reviews")
        assert entry.harmful == 0

        record_outcome("tester", entry.id, "reverted")

        loaded = load_playbook("tester")
        assert loaded[0].harmful == 1
        assert loaded[0].helpful == 0

    def test_curate_removes_net_negative(self, playbook_dir: Path) -> None:
        entries = [
            PlaybookEntry(
                id="test-0001",
                category="DO",
                rule="Good rule",
                helpful=5,
                harmful=0,
            ),
            PlaybookEntry(
                id="test-0002",
                category="DO",
                rule="Bad rule",
                helpful=1,
                harmful=4,  # net negative, observations=5 >= min_observations=3
            ),
        ]
        save_playbook("tester", entries)

        curated = curate("tester", min_observations=3)

        assert len(curated) == 1
        assert curated[0].id == "test-0001"

    def test_curate_keeps_insufficient_observations(self, playbook_dir: Path) -> None:
        entries = [
            PlaybookEntry(
                id="test-0001",
                category="DO",
                rule="New bad rule",
                helpful=0,
                harmful=2,  # net negative, but observations=2 < min_observations=3
            ),
        ]
        save_playbook("tester", entries)

        curated = curate("tester", min_observations=3)

        assert len(curated) == 1
        assert curated[0].id == "test-0001"

    def test_curate_deduplicates(self, playbook_dir: Path) -> None:
        entries = [
            PlaybookEntry(
                id="test-0001",
                category="DO",
                rule="Always run linting before committing code",
                helpful=5,
                harmful=0,
            ),
            PlaybookEntry(
                id="test-0002",
                category="DO",
                rule="Always run linting before committing changes",  # >75% similar
                helpful=3,
                harmful=0,
            ),
        ]
        save_playbook("tester", entries)

        curated = curate("tester")

        assert len(curated) == 1
        # Should keep the higher-scored one (test-0001 with net_score=5)
        assert curated[0].id == "test-0001"

    def test_curate_caps_at_max(self, playbook_dir: Path) -> None:
        # Rules must be dissimilar enough to avoid dedup (< 75% similarity).
        distinct_rules = [
            "Always run the full linting suite before opening a pull request",
            "Mock external HTTP calls in every integration test you write",
            "Pin all dependency versions explicitly in pyproject.toml",
            "Use structured logging with JSON format for production services",
            "Write docstrings for every public function and class method",
        ]
        entries = [
            PlaybookEntry(
                id=f"test-{i:04d}",
                category="DO",
                rule=distinct_rules[i - 1],
                helpful=i,
                harmful=0,
            )
            for i in range(1, 6)
        ]
        save_playbook("tester", entries)

        curated = curate("tester", max_entries=3)

        assert len(curated) == 3
        # Top 3 by net score descending: 5, 4, 3
        assert [e.helpful for e in curated] == [5, 4, 3]
