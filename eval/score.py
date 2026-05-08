# SPDX-License-Identifier: Apache-2.0
"""Three-tier composite scoring script for SDG Hub.

Tiers:
  - Hygiene  (0.30): pytest, ruff, mypy, structural tests
  - Growth   (0.20): registered blocks+flows+connectors, test coverage
  - Project  (0.50): block tests, flow tests, connector tests

Usage:
  uv run python eval/score.py          # human-readable output
  uv run python eval/score.py --json   # JSON output
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path = ROOT) -> bool:
    """Run a subprocess and return True if it exits 0."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            timeout=300,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_pytest() -> float:
    """Run unit tests (excludes slow/integration)."""
    return (
        1.0
        if _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/blocks",
                "tests/connectors",
                "tests/flow",
                "tests/utils",
                "-m",
                "not (examples or slow)",
                "-q",
                "--tb=no",
            ]
        )
        else 0.0
    )


def check_ruff() -> float:
    """Run ruff linter."""
    return (
        1.0
        if _run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "src/",
                "tests/",
            ]
        )
        else 0.0
    )


def check_mypy() -> float:
    """Run mypy type checker."""
    return (
        1.0
        if _run(
            [
                sys.executable,
                "-m",
                "mypy",
                "src/sdg_hub",
            ]
        )
        else 0.0
    )


def check_structural() -> float:
    """Verify structural invariants: key directories and files exist."""
    required = [
        "src/sdg_hub/core/blocks",
        "src/sdg_hub/core/flow",
        "src/sdg_hub/core/connectors",
        "tests/blocks",
        "tests/flow",
        "tests/connectors",
        "pyproject.toml",
    ]
    return 1.0 if all((ROOT / p).exists() for p in required) else 0.0


def check_registry_count() -> float:
    """Count registered blocks, flows, and connectors (pass if >= 1 each)."""
    script = (
        "import sys, io; sys.stdout = io.StringIO(); "
        "from sdg_hub.core.blocks.registry import BlockRegistry; "
        "from sdg_hub.core.flow.registry import FlowRegistry; "
        "from sdg_hub.core.connectors.registry import ConnectorRegistry; "
        "import sdg_hub.core.connectors.agent; "
        "BlockRegistry.discover_blocks(); FlowRegistry.discover_flows(); "
        "b = len(BlockRegistry.list_blocks()); "
        "f = len(FlowRegistry.list_flows()); "
        "c = len(ConnectorRegistry.list_all()); "
        "sys.stdout = sys.__stdout__; "
        "print(f'{b},{f},{c}')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return 0.0
        parts = result.stdout.strip().split(",")
        b, f, c = int(parts[0]), int(parts[1]), int(parts[2])
        return 1.0 if b >= 1 and f >= 1 and c >= 1 else 0.0
    except Exception:
        return 0.0


def check_test_coverage() -> float:
    """Check that test directories cover block categories."""
    block_dirs = {
        d.name
        for d in (ROOT / "src/sdg_hub/core/blocks").iterdir()
        if d.is_dir() and d.name != "__pycache__"
    }
    test_block_dir = ROOT / "tests/blocks"
    if not test_block_dir.exists():
        return 0.0

    # Count block categories that have at least one test file anywhere
    # under tests/blocks/ (in subdirs or matching by name).
    test_subdirs = {
        d.name
        for d in test_block_dir.iterdir()
        if d.is_dir() and d.name != "__pycache__"
    }
    test_files = {f.stem for f in test_block_dir.rglob("test_*.py")}

    covered = 0
    for cat in block_dirs:
        # Direct subdir match or test file name contains the category
        has_subdir = cat in test_subdirs or any(cat in s for s in test_subdirs)
        has_test = any(cat in t for t in test_files)
        if has_subdir or has_test:
            covered += 1

    return 1.0 if covered / max(len(block_dirs), 1) >= 0.5 else 0.0


def check_block_tests() -> float:
    """Run block tests."""
    return (
        1.0
        if _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/blocks",
                "-m",
                "not (examples or slow)",
                "-q",
                "--tb=no",
            ]
        )
        else 0.0
    )


def check_flow_tests() -> float:
    """Run flow tests."""
    return (
        1.0
        if _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/flow",
                "-m",
                "not (examples or slow)",
                "-q",
                "--tb=no",
            ]
        )
        else 0.0
    )


def check_connector_tests() -> float:
    """Run connector tests."""
    return (
        1.0
        if _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/connectors",
                "-m",
                "not (examples or slow)",
                "-q",
                "--tb=no",
            ]
        )
        else 0.0
    )


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

TIERS: dict[str, dict] = {
    "hygiene": {
        "weight": 0.30,
        "checks": {
            "pytest": {"fn": check_pytest, "weight": 0.35},
            "ruff": {"fn": check_ruff, "weight": 0.20},
            "mypy": {"fn": check_mypy, "weight": 0.15},
            "structural": {"fn": check_structural, "weight": 0.30},
        },
    },
    "growth": {
        "weight": 0.20,
        "checks": {
            "registry_count": {"fn": check_registry_count, "weight": 0.60},
            "test_coverage": {"fn": check_test_coverage, "weight": 0.40},
        },
    },
    "project": {
        "weight": 0.50,
        "checks": {
            "block_tests": {"fn": check_block_tests, "weight": 0.35},
            "flow_tests": {"fn": check_flow_tests, "weight": 0.40},
            "connector_tests": {"fn": check_connector_tests, "weight": 0.25},
        },
    },
}


def evaluate() -> dict:
    """Run all tiers and return structured results."""
    results: dict = {"tiers": {}, "score": 0.0}
    composite = 0.0

    for tier_name, tier_def in TIERS.items():
        tier_result: dict = {"weight": tier_def["weight"], "checks": {}, "score": 0.0}
        tier_score = 0.0

        for check_name, check_def in tier_def["checks"].items():
            value = check_def["fn"]()
            tier_result["checks"][check_name] = {
                "weight": check_def["weight"],
                "score": value,
            }
            tier_score += value * check_def["weight"]

        tier_result["score"] = round(tier_score, 4)
        results["tiers"][tier_name] = tier_result
        composite += tier_score * tier_def["weight"]

    results["score"] = round(composite, 4)
    return results


def print_human(results: dict) -> None:
    """Print human-readable results."""
    print("=" * 60)  # noqa: T201
    print("  SDG Hub -- Three-Tier Composite Score")  # noqa: T201
    print("=" * 60)  # noqa: T201

    for tier_name, tier_data in results["tiers"].items():
        w = tier_data["weight"]
        s = tier_data["score"]
        print(f"\n  {tier_name.upper()} (weight: {w:.0%}): {s:.2%}")  # noqa: T201
        for check_name, check_data in tier_data["checks"].items():
            cw = check_data["weight"]
            cs = check_data["score"]
            status = "PASS" if cs >= 1.0 else "FAIL"
            print(f"    [{status}] {check_name} (weight: {cw:.0%})")  # noqa: T201

    print("\n" + "-" * 60)  # noqa: T201
    print(f"  COMPOSITE SCORE: {results['score']:.4f}")  # noqa: T201
    print("=" * 60)  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description="SDG Hub composite scoring")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    results = evaluate()

    if args.json:
        print(json.dumps(results, indent=2))  # noqa: T201
    else:
        print_human(results)

    sys.exit(0 if results["score"] >= 0.5 else 1)


if __name__ == "__main__":
    main()
