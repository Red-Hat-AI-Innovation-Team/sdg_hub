# SPDX-License-Identifier: Apache-2.0
"""ACE Playbook system -- self-evolving behavioral rules for agents.

Manages DO/DON'T rules that evolve from experiment data.
Playbooks are stored as markdown at `eval/data/playbooks/{role}.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent.parent
PLAYBOOKS_DIR = ROOT / ".factory" / "playbooks"


@dataclass
class PlaybookEntry:
    """A single DO or DON'T rule in a playbook."""

    id: str
    category: str  # "DO" or "DONT"
    rule: str
    helpful: int = 0
    harmful: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def net_score(self) -> int:
        return self.helpful - self.harmful

    @property
    def observations(self) -> int:
        return self.helpful + self.harmful


# ---------------------------------------------------------------------------
# Parsing / serialization
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(
    r"^- \[(?P<id>[^\]]+)\] helpful=(?P<helpful>\d+) harmful=(?P<harmful>\d+)"
    r"(?: created=(?P<created>[^\s]+))? :: (?P<rule>.+)$"
)


def _parse_entry(line: str, category: str) -> PlaybookEntry | None:
    m = _ENTRY_RE.match(line.strip())
    if not m:
        return None
    return PlaybookEntry(
        id=m.group("id"),
        category=category,
        rule=m.group("rule"),
        helpful=int(m.group("helpful")),
        harmful=int(m.group("harmful")),
        created_at=m.group("created") or datetime.now(timezone.utc).isoformat(),
    )


def _format_entry(e: PlaybookEntry) -> str:
    return (
        f"- [{e.id}] helpful={e.helpful} harmful={e.harmful}"
        f" created={e.created_at} :: {e.rule}"
    )


def _playbook_path(role: str) -> Path:
    return PLAYBOOKS_DIR / f"{role}.md"


def _next_id(entries: list[PlaybookEntry], role: str) -> str:
    """Generate the next sequential ID for a role."""
    prefix = role[:5]
    max_num = 0
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
    for e in entries:
        m = pattern.match(e.id)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"{prefix}-{max_num + 1:04d}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_playbook(role: str) -> list[PlaybookEntry]:
    """Load a playbook from `eval/data/playbooks/{role}.md`."""
    path = _playbook_path(role)
    if not path.exists():
        return []

    entries: list[PlaybookEntry] = []
    current_category: str | None = None

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "### DO":
            current_category = "DO"
        elif stripped == "### DON'T":
            current_category = "DONT"
        elif stripped.startswith("- [") and current_category:
            entry = _parse_entry(stripped, current_category)
            if entry:
                entries.append(entry)

    return entries


def save_playbook(role: str, entries: list[PlaybookEntry]) -> None:
    """Write a playbook to `eval/data/playbooks/{role}.md`."""
    PLAYBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    do_entries = [e for e in entries if e.category == "DO"]
    dont_entries = [e for e in entries if e.category == "DONT"]

    lines = [
        "## Behavioral Playbook (auto-evolved from experiment data)",
        "",
        "### DO",
    ]
    for e in do_entries:
        lines.append(_format_entry(e))

    lines.append("")
    lines.append("### DON'T")
    for e in dont_entries:
        lines.append(_format_entry(e))

    lines.append("")
    _playbook_path(role).write_text("\n".join(lines))


def add_entry(role: str, category: str, rule: str) -> PlaybookEntry:
    """Add a new entry with helpful=0, harmful=0. Returns the new entry."""
    if category not in ("DO", "DONT"):
        raise ValueError(f"category must be 'DO' or 'DONT', got {category!r}")

    entries = load_playbook(role)
    entry = PlaybookEntry(
        id=_next_id(entries, role),
        category=category,
        rule=rule,
    )
    entries.append(entry)
    save_playbook(role, entries)
    return entry


def record_outcome(role: str, entry_id: str, outcome: str) -> None:
    """Increment helpful or harmful for an entry.

    Args:
        role: The playbook role.
        entry_id: The entry ID (e.g., "build-0001").
        outcome: "kept" increments helpful, "reverted" increments harmful.
    """
    if outcome not in ("kept", "reverted"):
        raise ValueError(f"outcome must be 'kept' or 'reverted', got {outcome!r}")

    entries = load_playbook(role)
    found = False
    for e in entries:
        if e.id == entry_id:
            if outcome == "kept":
                e.helpful += 1
            else:
                e.harmful += 1
            found = True
            break

    if not found:
        raise KeyError(f"entry {entry_id!r} not found in playbook {role!r}")

    save_playbook(role, entries)


def curate(
    role: str, min_observations: int = 3, max_entries: int = 15
) -> list[PlaybookEntry]:
    """Prune net-negative rules, deduplicate, and cap entries.

    1. Removes rules where harmful > helpful AND observations >= min_observations.
    2. Deduplicates rules with >75% text similarity (keeps the higher-scored one).
    3. Caps at max_entries sorted by net score descending.

    Returns the curated list.
    """
    entries = load_playbook(role)

    # Step 1: remove net-negative rules with sufficient observations
    entries = [
        e
        for e in entries
        if not (e.harmful > e.helpful and e.observations >= min_observations)
    ]

    # Step 2: deduplicate >75% similar rules
    deduplicated: list[PlaybookEntry] = []
    for entry in entries:
        is_dup = False
        for existing in deduplicated:
            if entry.category != existing.category:
                continue
            similarity = SequenceMatcher(
                None, entry.rule.lower(), existing.rule.lower()
            ).ratio()
            if similarity > 0.75:
                # Keep the one with the higher net score
                if entry.net_score > existing.net_score:
                    deduplicated.remove(existing)
                    deduplicated.append(entry)
                is_dup = True
                break
        if not is_dup:
            deduplicated.append(entry)

    # Step 3: cap at max_entries sorted by net score
    deduplicated.sort(key=lambda e: e.net_score, reverse=True)
    curated = deduplicated[:max_entries]

    save_playbook(role, curated)
    return curated


def format_for_prompt(role: str) -> str:
    """Return the playbook formatted for injection into an agent prompt."""
    entries = load_playbook(role)
    if not entries:
        return f"No playbook found for role '{role}'."

    do_rules = [e for e in entries if e.category == "DO"]
    dont_rules = [e for e in entries if e.category == "DONT"]

    lines = [f"## {role.title()} Behavioral Rules"]

    if do_rules:
        lines.append("")
        lines.append("DO:")
        for e in do_rules:
            sign = "+" if e.net_score >= 0 else ""
            lines.append(f"  - {e.rule} (score: {sign}{e.net_score})")

    if dont_rules:
        lines.append("")
        lines.append("DON'T:")
        for e in dont_rules:
            sign = "+" if e.net_score >= 0 else ""
            lines.append(f"  - {e.rule} (score: {sign}{e.net_score})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Seed playbooks
# ---------------------------------------------------------------------------

SEED_PLAYBOOKS: dict[str, list[PlaybookEntry]] = {
    "builder": [
        PlaybookEntry(
            id="build-0001",
            category="DO",
            rule="Always run ruff + mypy after making code changes",
        ),
        PlaybookEntry(
            id="build-0002",
            category="DO",
            rule="Use existing block patterns as reference when creating new blocks",
        ),
        PlaybookEntry(
            id="build-0003",
            category="DO",
            rule="Register new blocks with @BlockRegistry.register(name, category, description)",
        ),
        PlaybookEntry(
            id="build-0004",
            category="DONT",
            rule="Don't add type: ignore comments to suppress mypy errors",
        ),
        PlaybookEntry(
            id="build-0005",
            category="DONT",
            rule="Don't skip pre-commit hooks with --no-verify",
        ),
    ],
    "evaluator": [
        PlaybookEntry(
            id="evalu-0001",
            category="DO",
            rule="Always run structural tests before opening PRs",
        ),
        PlaybookEntry(
            id="evalu-0002",
            category="DO",
            rule="Test both success and error cases for every block",
        ),
        PlaybookEntry(
            id="evalu-0003",
            category="DO",
            rule="Mock LLM clients when testing LLM-powered blocks",
        ),
        PlaybookEntry(
            id="evalu-0004",
            category="DONT",
            rule="Don't rely on integration tests as the only test coverage",
        ),
    ],
    "reviewer": [
        PlaybookEntry(
            id="revie-0001",
            category="DO",
            rule="Verify conventional commit format on all PR commits",
        ),
        PlaybookEntry(
            id="revie-0002",
            category="DO",
            rule="Check that new blocks have corresponding test files under tests/blocks/",
        ),
        PlaybookEntry(
            id="revie-0003",
            category="DO",
            rule="Confirm flow YAML changes are covered by flow regression tests",
        ),
        PlaybookEntry(
            id="revie-0004",
            category="DONT",
            rule="Don't approve PRs that lower composite score below 0.5",
        ),
    ],
    "gardener": [
        PlaybookEntry(
            id="garde-0001",
            category="DO",
            rule="Keep uv.lock in sync with pyproject.toml after dependency changes",
        ),
        PlaybookEntry(
            id="garde-0002",
            category="DO",
            rule="Run uv pip install .[dev] to verify install after dependency updates",
        ),
        PlaybookEntry(
            id="garde-0003",
            category="DO",
            rule="Clean up __pycache__ and stale .pyc files periodically",
        ),
        PlaybookEntry(
            id="garde-0004",
            category="DONT",
            rule="Don't bump major versions without checking for breaking API changes",
        ),
    ],
    "lead": [
        PlaybookEntry(
            id="lead--0001",
            category="DO",
            rule="Use flow.dry_run() to validate pipeline changes before committing",
        ),
        PlaybookEntry(
            id="lead--0002",
            category="DO",
            rule="Run the composite score (eval/score.py) before merging feature branches",
        ),
        PlaybookEntry(
            id="lead--0003",
            category="DO",
            rule="Set model_config and agent_config before calling flow.generate()",
        ),
        PlaybookEntry(
            id="lead--0004",
            category="DONT",
            rule="Don't merge branches that fail any CI check in the required matrix",
        ),
        PlaybookEntry(
            id="lead--0005",
            category="DONT",
            rule="Don't commit .env files or API keys to the repository",
        ),
    ],
}


def init_playbooks() -> list[str]:
    """Create starter playbooks for all roles. Returns list of created roles."""
    created: list[str] = []
    for role, entries in SEED_PLAYBOOKS.items():
        if not _playbook_path(role).exists():
            save_playbook(role, entries)
            created.append(role)
    return created
