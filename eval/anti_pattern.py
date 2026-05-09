# SPDX-License-Identifier: Apache-2.0
"""Anti-pattern detection for agent PRs.

Prevents agents from re-submitting changes that are too similar to
previously reverted PRs. Uses difflib.SequenceMatcher for comparison.

Usage:
  # Check a PR diff against known reverted patterns
  uv run python eval/anti_pattern.py check

  # Record a reverted PR's diff for future comparison
  uv run python eval/anti_pattern.py record <pr_number>

  # List all recorded patterns
  uv run python eval/anti_pattern.py list
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import difflib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
REVERTED_DIFFS_DIR = ROOT / "eval" / "data" / "reverted_diffs"

SIMILARITY_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return the completed process."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _get_current_diff() -> str:
    """Get the diff of the current branch against main."""
    result = _run(["git", "diff", "main...HEAD"])
    if result.returncode != 0:
        print(f"Error getting diff: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def _load_reverted_diffs() -> dict[str, str]:
    """Load all reverted diffs from the archive directory.

    Returns a mapping of PR number to diff content.
    """
    diffs: dict[str, str] = {}
    if not REVERTED_DIFFS_DIR.exists():
        return diffs
    for diff_file in sorted(REVERTED_DIFFS_DIR.glob("*.diff")):
        pr_number = diff_file.stem
        diffs[pr_number] = diff_file.read_text()
    return diffs


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_check() -> int:
    """Check the current PR diff against all recorded reverted diffs."""
    new_diff = _get_current_diff()
    if not new_diff.strip():
        print("No diff found between current branch and main.")
        return 0

    reverted_diffs = _load_reverted_diffs()
    if not reverted_diffs:
        print("No reverted diffs recorded. Nothing to compare against.")
        return 0

    blocked = False
    for pr_number, reverted_diff in reverted_diffs.items():
        ratio = difflib.SequenceMatcher(None, new_diff, reverted_diff).ratio()
        if ratio > SIMILARITY_THRESHOLD:
            # Load metadata if available
            meta_file = REVERTED_DIFFS_DIR / f"{pr_number}.json"
            title = ""
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                title = meta.get("title", "")

            label = f"PR #{pr_number}"
            if title:
                label += f" ({title})"

            print(
                f"BLOCKED: Current diff is {ratio:.0%} similar to "
                f"reverted {label}. Threshold is "
                f"{SIMILARITY_THRESHOLD:.0%}.",
                file=sys.stderr,
            )
            blocked = True

    if blocked:
        return 1

    print("No anti-pattern matches found. OK to proceed.")
    return 0


def cmd_record(pr_number: str) -> int:
    """Record a reverted PR's diff for future comparison."""
    REVERTED_DIFFS_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch the diff
    if not pr_number.isdigit():
        print(f"Error: PR number must be numeric, got '{pr_number}'", file=sys.stderr)
        return 1

    result = _run(["gh", "pr", "diff", pr_number])
    if result.returncode != 0:
        print(
            f"Error fetching diff for PR #{pr_number}: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return 1

    diff_content = result.stdout
    if not diff_content.strip():
        print(f"PR #{pr_number} has an empty diff.", file=sys.stderr)
        return 1

    # Save the diff
    diff_path = REVERTED_DIFFS_DIR / f"{pr_number}.diff"
    diff_path.write_text(diff_content)

    # Fetch and save metadata
    meta_result = _run(["gh", "pr", "view", pr_number, "--json", "title,closedAt,body"])
    meta: dict[str, str] = {
        "pr_number": pr_number,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if meta_result.returncode == 0:
        pr_data = json.loads(meta_result.stdout)
        meta["title"] = pr_data.get("title", "")
        meta["closed_at"] = pr_data.get("closedAt", "")
        meta["reason"] = pr_data.get("body", "")

    meta_path = REVERTED_DIFFS_DIR / f"{pr_number}.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Recorded PR #{pr_number} diff to {diff_path.relative_to(ROOT)}")
    print(f"Recorded PR #{pr_number} metadata to {meta_path.relative_to(ROOT)}")
    return 0


def cmd_list() -> int:
    """List all recorded reverted PR patterns."""
    reverted_diffs = _load_reverted_diffs()
    if not reverted_diffs:
        print("No reverted diffs recorded.")
        return 0

    print(f"Recorded reverted patterns ({len(reverted_diffs)}):\n")
    for pr_number in reverted_diffs:
        meta_file = REVERTED_DIFFS_DIR / f"{pr_number}.json"
        title = ""
        recorded_at = ""
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            title = meta.get("title", "")
            recorded_at = meta.get("recorded_at", "")

        line = f"  PR #{pr_number}"
        if title:
            line += f" - {title}"
        if recorded_at:
            line += f" (recorded: {recorded_at})"
        print(line)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anti-pattern detection for agent PRs.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check", help="Check current diff against reverted patterns")

    record_parser = sub.add_parser("record", help="Record a reverted PR diff")
    record_parser.add_argument("pr_number", help="PR number to record")

    sub.add_parser("list", help="List all recorded patterns")

    args = parser.parse_args()

    if args.command == "check":
        return cmd_check()
    elif args.command == "record":
        return cmd_record(args.pr_number)
    elif args.command == "list":
        return cmd_list()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
