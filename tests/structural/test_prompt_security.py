# SPDX-License-Identifier: Apache-2.0
"""Verify prompt templates do not contain injection or exfiltration patterns.

SDG Hub prompt templates are fed directly to LLMs. Community-contributed
flows could accidentally (or deliberately) introduce patterns that
manipulate model behavior or exfiltrate data.

The red_team flow is explicitly allowlisted since it intentionally
generates adversarial content.

Security patterns ported from harness-eval-lab inspection rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.structural.security_patterns import ALL_SECURITY_PATTERNS

FLOWS_DIR = Path(__file__).resolve().parents[2] / "src" / "sdg_hub" / "flows"

ALLOWLISTED_DIRS = {"red_team"}


def _discover_prompt_yamls() -> list[Path]:
    """Return all prompt template YAMLs (excluding allowlisted directories)."""
    candidates = sorted(FLOWS_DIR.rglob("*.yaml")) + sorted(FLOWS_DIR.rglob("*.yml"))
    prompts: list[Path] = []
    for path in candidates:
        rel = path.relative_to(FLOWS_DIR)
        if rel.parts[0] in ALLOWLISTED_DIRS:
            continue
        # Prompt templates are YAML lists of role/content dicts.
        # Skip flow.yaml files (which have metadata + blocks keys).
        try:
            with open(path) as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError:
            continue
        if isinstance(data, list) and data and isinstance(data[0], dict) and "content" in data[0]:
            prompts.append(path)
    return prompts


def _prompt_id(path: Path) -> str:
    return str(path.relative_to(FLOWS_DIR))


PROMPT_YAMLS = _discover_prompt_yamls()


def _extract_content_fields(data: list[dict]) -> list[tuple[int, str]]:
    """Extract (index, content) pairs from a prompt template."""
    results = []
    for i, entry in enumerate(data):
        content = entry.get("content", "")
        if isinstance(content, str) and content.strip():
            results.append((i, content))
    return results


@pytest.mark.parametrize(
    "prompt_path", PROMPT_YAMLS, ids=[_prompt_id(p) for p in PROMPT_YAMLS]
)
def test_prompt_template_no_injection_patterns(prompt_path: Path) -> None:
    """Prompt templates must not contain patterns that manipulate LLM behavior."""
    with open(prompt_path) as fh:
        data = yaml.safe_load(fh)

    findings: list[str] = []

    for entry_idx, content in _extract_content_fields(data):
        for line_offset, line in enumerate(content.split("\n")):
            for label, pattern in ALL_SECURITY_PATTERNS:
                if pattern.search(line):
                    role = data[entry_idx].get("role", "unknown")
                    findings.append(
                        f"  [{role}] line {line_offset + 1}: "
                        f"'{label}' pattern detected"
                    )

    assert not findings, (
        f"Prompt template {prompt_path.relative_to(FLOWS_DIR)} contains "
        f"{len(findings)} security pattern(s):\n"
        + "\n".join(findings)
        + "\n\nIf this is intentional adversarial content, add the flow's "
        "top-level directory to ALLOWLISTED_DIRS in this test."
    )
