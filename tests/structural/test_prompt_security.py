# SPDX-License-Identifier: Apache-2.0
"""Verify prompt templates do not contain adversarial patterns.

SDG Hub prompt templates are fed directly to LLMs. As the flow catalog
grows with community contributions, templates could introduce patterns
that manipulate model behavior. These tests catch two classes of issues:

1. Injection patterns: adversarial instructions (jailbreak, role hijack,
   instruction override) that have no legitimate use in data generation
   prompt templates.

2. Template structure anomalies: Jinja2 control flow ({% if/for %}) in
   non-red-team templates. All existing templates use simple {{ variable }}
   substitution only; the red_team flow is the sole user of conditional
   logic. Control flow in other templates could indicate template logic
   that processes untrusted input.

The red_team flow is explicitly allowlisted since it intentionally
generates adversarial content.

Security patterns adapted from harness-eval-lab inspection rules,
curated for SDG Hub's prompt template context.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.structural.security_patterns import (
    INJECTION_PATTERNS,
    TEMPLATE_STRUCTURE_PATTERNS,
)

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
def test_no_injection_patterns(prompt_path: Path) -> None:
    """Prompt templates must not contain patterns that manipulate LLM behavior."""
    with open(prompt_path) as fh:
        data = yaml.safe_load(fh)

    findings: list[str] = []

    for entry_idx, content in _extract_content_fields(data):
        for line_offset, line in enumerate(content.split("\n")):
            for label, pattern in INJECTION_PATTERNS:
                if pattern.search(line):
                    role = data[entry_idx].get("role", "unknown")
                    findings.append(
                        f"  [{role}] line {line_offset + 1}: "
                        f"'{label}' pattern detected"
                    )

    assert not findings, (
        f"Prompt template {prompt_path.relative_to(FLOWS_DIR)} contains "
        f"{len(findings)} injection pattern(s):\n"
        + "\n".join(findings)
        + "\n\nIf this is intentional adversarial content, add the flow's "
        "top-level directory to ALLOWLISTED_DIRS in this test."
    )


@pytest.mark.parametrize(
    "prompt_path", PROMPT_YAMLS, ids=[_prompt_id(p) for p in PROMPT_YAMLS]
)
def test_no_template_structure_anomalies(prompt_path: Path) -> None:
    """Prompt templates should use simple {{ variable }} substitution only.

    Jinja2 control flow ({% if %}, {% for %}) in non-red-team templates
    is unexpected and may indicate template logic that processes untrusted
    input or bypasses the expected block pipeline.
    """
    with open(prompt_path) as fh:
        data = yaml.safe_load(fh)

    findings: list[str] = []

    for entry_idx, content in _extract_content_fields(data):
        for line_offset, line in enumerate(content.split("\n")):
            for label, pattern in TEMPLATE_STRUCTURE_PATTERNS:
                if pattern.search(line):
                    role = data[entry_idx].get("role", "unknown")
                    findings.append(
                        f"  [{role}] line {line_offset + 1}: "
                        f"'{label}' detected"
                    )

    assert not findings, (
        f"Prompt template {prompt_path.relative_to(FLOWS_DIR)} contains "
        f"{len(findings)} structural anomaly(ies):\n"
        + "\n".join(findings)
        + "\n\nAll non-red-team templates should use simple {{ variable }} "
        "substitution. If Jinja2 control flow is intentional, add the "
        "flow's top-level directory to ALLOWLISTED_DIRS in this test."
    )
