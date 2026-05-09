# SPDX-License-Identifier: Apache-2.0
"""Agent PR revert rate monitoring script for SDG Hub.

Tracks agent PR merge/revert rates and flags agents whose revert rate
exceeds a configurable threshold (default 30%).

Usage:
  uv run python eval/revert_monitor.py              # human-readable output
  uv run python eval/revert_monitor.py --json        # JSON output
  uv run python eval/revert_monitor.py --threshold 0.2  # custom threshold
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT / ".factory" / "archive"
REPORT_PATH = ARCHIVE_DIR / "performance_report.json"

KNOWN_AGENTS = [
    "sdg-builder",
    "sdg-evaluator",
    "sdg-gardener",
    "sdg-lead",
]

DEFAULT_THRESHOLD = 0.30


def _gh(args: list[str]) -> str:
    """Run a gh CLI command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(
            ["gh", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(
                f"WARNING: gh {' '.join(args)} failed (exit {result.returncode})",
                file=sys.stderr,
            )
            return ""
        return result.stdout.strip()
    except FileNotFoundError:
        print("ERROR: 'gh' CLI not found", file=sys.stderr)
        return ""
    except subprocess.TimeoutExpired:
        print(f"WARNING: gh {' '.join(args)} timed out", file=sys.stderr)
        return ""


def fetch_merged_prs() -> list[dict]:
    """Fetch merged PRs with the agent-pr label."""
    raw = _gh(
        [
            "pr",
            "list",
            "--state",
            "merged",
            "--label",
            "agent-pr",
            "--json",
            "number,title,mergedAt,author",
            "--limit",
            "100",
        ]
    )
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"WARNING: Failed to parse gh output: {exc}", file=sys.stderr)
        return []


def fetch_reverted_prs() -> list[dict]:
    """Fetch merged PRs that look like reverts."""
    raw = _gh(
        [
            "pr",
            "list",
            "--state",
            "merged",
            "--search",
            "revert",
            "--json",
            "number,title",
            "--limit",
            "100",
        ]
    )
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"WARNING: Failed to parse gh output: {exc}", file=sys.stderr)
        return []


def classify_by_agent(
    merged_prs: list[dict],
    reverted_prs: list[dict],
) -> dict[str, dict[str, int]]:
    """Classify merged and reverted PRs per agent.

    Returns a dict mapping agent login to {"merged": N, "reverted": N}.
    """
    stats: dict[str, dict[str, int]] = {}
    for agent in KNOWN_AGENTS:
        stats[agent] = {"merged": 0, "reverted": 0}

    reverted_numbers: set[int] = set()
    pr_ref_pattern = re.compile(r"#(\d+)")
    for pr in reverted_prs:
        title = pr.get("title", "")
        for m in pr_ref_pattern.finditer(title):
            reverted_numbers.add(int(m.group(1)))

    for pr in merged_prs:
        author_login = ""
        author = pr.get("author")
        if isinstance(author, dict):
            author_login = author.get("login", "")
        elif isinstance(author, str):
            author_login = author

        if author_login not in stats:
            stats[author_login] = {"merged": 0, "reverted": 0}

        pr_number = pr.get("number", 0)
        if pr_number in reverted_numbers:
            stats[author_login]["reverted"] += 1
        else:
            stats[author_login]["merged"] += 1

    return stats


def compute_rates(
    stats: dict[str, dict[str, int]],
) -> dict[str, dict]:
    """Compute revert rates per agent."""
    rates: dict[str, dict] = {}
    for agent, counts in stats.items():
        merged = counts["merged"]
        reverted = counts["reverted"]
        total = merged + reverted
        rate = reverted / total if total > 0 else 0.0
        rates[agent] = {
            "merged": merged,
            "reverted": reverted,
            "rate": round(rate, 4),
        }
    return rates


def build_report(
    rates: dict[str, dict],
    threshold: float,
) -> dict:
    """Build the full performance report dict."""
    any_exceeded = any(
        agent_data["rate"] > threshold
        for agent_data in rates.values()
        if (agent_data["merged"] + agent_data["reverted"]) > 0
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agents": rates,
        "threshold": threshold,
        "status": "warning" if any_exceeded else "healthy",
    }


def save_report(report: dict) -> None:
    """Persist report to eval/data/performance_report.json."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")


def print_human(rates: dict[str, dict], threshold: float) -> None:
    """Print human-readable revert rate table."""
    print("Agent Revert Rate Report")
    print("=" * 40)
    for agent in sorted(rates):
        data = rates[agent]
        merged = data["merged"]
        reverted = data["reverted"]
        total = merged + reverted
        rate = data["rate"]

        if total == 0:
            line = f"{agent + ':':20s} 0/0 (no PRs)"
        else:
            pct = rate * 100
            status = "WARN" if rate > threshold else "OK"
            line = (
                f"{agent + ':':20s} "
                f"{merged}/{total} merged, "
                f"{reverted} reverted "
                f"({pct:.1f}%) {status}"
            )
        print(f"  {line}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent PR revert rate monitor")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Revert rate threshold (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    merged_prs = fetch_merged_prs()
    reverted_prs = fetch_reverted_prs()

    stats = classify_by_agent(merged_prs, reverted_prs)
    rates = compute_rates(stats)
    report = build_report(rates, args.threshold)

    save_report(report)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human(rates, args.threshold)
        print(f"  Report saved to {REPORT_PATH.relative_to(ROOT)}")
    # Check for any agent exceeding threshold and exit accordingly.
    exceeded = [
        agent
        for agent, data in rates.items()
        if data["rate"] > args.threshold and (data["merged"] + data["reverted"]) > 0
    ]

    if exceeded:
        print()
        for agent in exceeded:
            pct = rates[agent]["rate"] * 100
            print(
                f"  WARNING: {agent} revert rate {pct:.1f}% "
                f"exceeds {args.threshold:.0%} threshold."
            )
            print(
                f"  Suggestion: add 'needs-human-review' label to {agent}'s future PRs."
            )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
