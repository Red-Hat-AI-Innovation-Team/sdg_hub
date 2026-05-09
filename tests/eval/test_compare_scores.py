# SPDX-License-Identifier: Apache-2.0
"""Tests for eval/compare_scores.py -- CI gate decision maker."""

from __future__ import annotations

from pathlib import Path
import sys

# eval/ is not a package; add it to sys.path so we can import it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eval"))

from compare_scores import build_report  # noqa: E402


def _scores(
    tiers: dict[str, float],
    composite: float,
) -> dict:
    """Build a minimal score dict matching eval/score.py output shape."""
    return {
        "tiers": {name: {"score": score} for name, score in tiers.items()},
        "score": composite,
    }


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


class TestBuildReport:
    def test_all_tiers_pass(self) -> None:
        before = _scores({"hygiene": 0.80, "growth": 0.70, "project": 0.75}, 0.85)
        after = _scores({"hygiene": 0.85, "growth": 0.75, "project": 0.80}, 0.90)

        report, passed = build_report(before, after)

        assert passed is True
        assert "PASSED" in report

    def test_tier_regression_fails(self) -> None:
        before = _scores({"hygiene": 0.90, "growth": 0.80, "project": 0.85}, 0.88)
        after = _scores({"hygiene": 0.70, "growth": 0.85, "project": 0.90}, 0.88)

        _, passed = build_report(before, after)

        assert passed is False

    def test_composite_regression_fails(self) -> None:
        before = _scores({"hygiene": 0.90, "growth": 0.80, "project": 0.85}, 0.90)
        after = _scores({"hygiene": 0.90, "growth": 0.80, "project": 0.85}, 0.85)

        _, passed = build_report(before, after)

        assert passed is False

    def test_below_absolute_threshold(self) -> None:
        """After composite is 0.79 (below 0.8) even though it improved."""
        before = _scores({"hygiene": 0.60, "growth": 0.50, "project": 0.55}, 0.70)
        after = _scores({"hygiene": 0.65, "growth": 0.55, "project": 0.60}, 0.79)

        report, passed = build_report(before, after)

        assert passed is False
        assert "Absolute threshold not met" in report

    def test_missing_tier_in_after(self) -> None:
        """Before has a tier that after does not -- should not crash."""
        before = _scores({"hygiene": 0.80, "growth": 0.70, "project": 0.75}, 0.85)
        after = _scores({"hygiene": 0.85, "growth": 0.75}, 0.85)

        # Must not raise KeyError.
        report, passed = build_report(before, after)

        # The missing tier row should appear in the report.
        assert "removed" in report
        assert passed is True
