# SPDX-License-Identifier: Apache-2.0
"""Tests for eval/score.py -- weight invariants and composite calculation."""

from __future__ import annotations

from pathlib import Path
import math
import sys

# eval/ is not a package; add it to sys.path so we can import it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eval"))

from score import TIERS, evaluate  # noqa: E402

# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


class TestScoreWeights:
    def test_tier_weights_sum_to_one(self) -> None:
        total = sum(tier["weight"] for tier in TIERS.values())
        assert math.isclose(total, 1.0, rel_tol=1e-9), (
            f"Tier weights sum to {total}, expected 1.0"
        )

    def test_check_weights_sum_to_one_per_tier(self) -> None:
        for tier_name, tier_def in TIERS.items():
            total = sum(c["weight"] for c in tier_def["checks"].values())
            assert math.isclose(total, 1.0, rel_tol=1e-9), (
                f"{tier_name} check weights sum to {total}, expected 1.0"
            )

    def test_evaluate_with_mocked_checks(self) -> None:
        """Mock all check functions and verify composite calculation.

        TIERS stores direct function references, so we must replace the
        ``fn`` values inside the dict rather than patching module-level names.
        """
        # Map: (tier_name, check_name) -> mock return value
        mock_scores: dict[tuple[str, str], float] = {
            ("hygiene", "pytest"): 1.0,
            ("hygiene", "ruff"): 1.0,
            ("hygiene", "mypy"): 0.0,
            ("hygiene", "structural"): 1.0,
            ("growth", "registry_count"): 1.0,
            ("growth", "test_coverage"): 0.0,
            ("project", "block_tests"): 1.0,
            ("project", "flow_tests"): 0.0,
            ("project", "connector_tests"): 1.0,
        }

        # Save originals and swap in lambdas.
        originals: dict[tuple[str, str], object] = {}
        for (tier, check), value in mock_scores.items():
            originals[(tier, check)] = TIERS[tier]["checks"][check]["fn"]
            TIERS[tier]["checks"][check]["fn"] = lambda v=value: v

        try:
            results = evaluate()
        finally:
            for (tier, check), fn in originals.items():
                TIERS[tier]["checks"][check]["fn"] = fn

        # Manually compute expected composite:
        # hygiene: 1.0*0.35 + 1.0*0.20 + 0.0*0.15 + 1.0*0.30 = 0.85
        # growth:  1.0*0.60 + 0.0*0.40 = 0.60
        # project: 1.0*0.35 + 0.0*0.40 + 1.0*0.25 = 0.60
        #
        # composite: 0.85*0.30 + 0.60*0.20 + 0.60*0.50
        #          = 0.255 + 0.12 + 0.30 = 0.675
        expected_composite = 0.675

        assert math.isclose(results["score"], expected_composite, rel_tol=1e-3), (
            f"Composite {results['score']} != expected {expected_composite}"
        )

        # Verify individual tier scores
        assert math.isclose(results["tiers"]["hygiene"]["score"], 0.85, rel_tol=1e-3)
        assert math.isclose(results["tiers"]["growth"]["score"], 0.60, rel_tol=1e-3)
        assert math.isclose(results["tiers"]["project"]["score"], 0.60, rel_tol=1e-3)
