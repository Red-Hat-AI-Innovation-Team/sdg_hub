# SPDX-License-Identifier: Apache-2.0
"""Verify prompt templates do not contain adversarial patterns.

SDG Hub prompt templates are fed directly to LLMs. As the flow catalog
grows with community contributions, templates could introduce patterns
that manipulate model behavior. These tests catch two classes of issues:

1. Injection patterns: adversarial instructions (jailbreak, role hijack,
   instruction override) that have no legitimate use in data generation
   prompt templates.

2. Template structure anomalies: Jinja2 directives (control flow, file
   inclusion, macros) in non-red-team templates. All existing templates
   use simple {{ variable }} substitution only; the red_team flow is the
   sole user of Jinja2 control flow. Directives in other templates could
   indicate template logic that processes untrusted input.

The red_team flow's prompt_generation directory is explicitly allowlisted
since it intentionally generates adversarial content and uses Jinja2
conditionals for dynamic prompt construction.

Security patterns adapted from harness-eval-lab inspection rules,
curated for SDG Hub's prompt template context.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from tests.structural.security_patterns import (
    INJECTION_PATTERNS,
    TEMPLATE_STRUCTURE_PATTERNS,
)

logger = logging.getLogger(__name__)

FLOWS_DIR = Path(__file__).resolve().parents[2] / "src" / "sdg_hub" / "flows"

ALLOWLISTED_DIRS = {"red_team"}


def _discover_prompt_yamls() -> tuple[list[Path], list[tuple[Path, str]]]:
    """Return prompt template YAMLs and any files that failed to parse.

    Returns:
        A tuple of (prompt_paths, parse_errors) where parse_errors is a list
        of (path, error_message) for files that could not be parsed.
    """
    candidates = sorted(FLOWS_DIR.rglob("*.yaml")) + sorted(FLOWS_DIR.rglob("*.yml"))
    prompts: list[Path] = []
    parse_errors: list[tuple[Path, str]] = []
    for path in candidates:
        rel = path.relative_to(FLOWS_DIR)
        if rel.parts[0] in ALLOWLISTED_DIRS:
            continue
        try:
            with open(path) as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            parse_errors.append((path, str(exc)))
            continue
        if isinstance(data, list) and data and any(
            isinstance(item, dict) and "content" in item for item in data
        ):
            prompts.append(path)
    return prompts, parse_errors


def _prompt_id(path: Path) -> str:
    return str(path.relative_to(FLOWS_DIR))


PROMPT_YAMLS, _PARSE_ERRORS = _discover_prompt_yamls()


def _extract_content_fields(data: list[dict]) -> list[tuple[int, str]]:
    """Extract (index, content) pairs from a prompt template.

    Handles both string and list-of-strings content values.
    """
    results = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        content = entry.get("content", "")
        if isinstance(content, str) and content.strip():
            results.append((i, content))
        elif isinstance(content, list):
            joined = "\n".join(str(item) for item in content if isinstance(item, str))
            if joined.strip():
                results.append((i, joined))
    return results


def test_prompt_yamls_discovered() -> None:
    """Prompt template discovery must find templates.

    Guards against a broken discovery function silently producing
    zero test cases, which would make CI pass with no actual scanning.
    """
    assert PROMPT_YAMLS, (
        f"No prompt template YAMLs found under {FLOWS_DIR}. "
        "Check that _discover_prompt_yamls() and FLOWS_DIR are correct."
    )


def test_no_yaml_parse_errors() -> None:
    """All non-allowlisted YAML files must parse successfully.

    A malformed template that silently bypasses scanning is itself a
    security concern.
    """
    assert not _PARSE_ERRORS, (
        f"{len(_PARSE_ERRORS)} YAML file(s) failed to parse:\n"
        + "\n".join(
            f"  {path.relative_to(FLOWS_DIR)}: {err}"
            for path, err in _PARSE_ERRORS
        )
    )


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

    Jinja2 directives ({% if %}, {% for %}, {% include %}, etc.) in
    non-red-team templates are unexpected and may indicate template logic
    that processes untrusted input or bypasses the expected block pipeline.
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
