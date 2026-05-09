# SPDX-License-Identifier: Apache-2.0
"""Compare before/after quality scores and produce a Markdown report.

Reads two JSON files produced by ``eval/score.py --json`` and outputs a
Markdown comparison table.  Exits non-zero when the after score regresses
below the before score or drops below the absolute threshold.

Usage:
  python eval/compare_scores.py before_score.json after_score.json
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ABSOLUTE_THRESHOLD = 0.8


def _delta_icon(delta: float) -> str:
    # U+2705 / U+274C render in GitHub Markdown
    return "✅" if delta >= 0 else "❌"


def build_report(before: dict, after: dict) -> tuple[str, bool]:
    """Build the Markdown comparison table and return (report, passed)."""
    passed = True
    rows: list[str] = []

    for tier_name in before["tiers"]:
        b_score = before["tiers"][tier_name]["score"]
        b_pct = f"{b_score:.0%}"
        if tier_name not in after["tiers"]:
            # Tier removed or renamed between branches
            rows.append(f"| {tier_name} | {b_pct} | — | removed |")
            continue
        a_score = after["tiers"][tier_name]["score"]
        delta = a_score - b_score
        icon = _delta_icon(delta)
        if delta < 0:
            passed = False
        sign = "+" if delta >= 0 else ""
        rows.append(
            f"| {tier_name.capitalize()} "
            f"| {b_score:.0%} "
            f"| {a_score:.0%} "
            f"| {sign}{delta:.0%} {icon} |"
        )

    b_composite = before["score"]
    a_composite = after["score"]
    c_delta = a_composite - b_composite
    c_icon = _delta_icon(c_delta)
    if c_delta < 0:
        passed = False
    c_sign = "+" if c_delta >= 0 else ""

    rows.append(
        f"| **Composite** "
        f"| **{b_composite:.2f}** "
        f"| **{a_composite:.2f}** "
        f"| **{c_sign}{c_delta:.2f} {c_icon}** |"
    )

    # Absolute threshold check
    if a_composite < ABSOLUTE_THRESHOLD:
        passed = False

    lines = [
        "## Quality Score Gate",
        "",
        "| Tier | Before | After | Delta |",
        "|------|--------|-------|-------|",
        *rows,
        "",
    ]

    if a_composite < ABSOLUTE_THRESHOLD:
        lines.append(
            f"**Absolute threshold not met:** composite score "
            f"{a_composite:.2f} < {ABSOLUTE_THRESHOLD}"
        )
    if not passed:
        lines.append("")
        lines.append("**Result: FAILED**")
    else:
        lines.append("**Result: PASSED**")

    return "\n".join(lines), passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare quality scores")
    parser.add_argument("before", type=Path, help="Path to before_score.json")
    parser.add_argument("after", type=Path, help="Path to after_score.json")
    args = parser.parse_args()

    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())

    report, passed = build_report(before, after)
    print(report)  # noqa: T201
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
