# SPDX-License-Identifier: Apache-2.0
"""Security patterns for prompt template scanning.

Curated for SDG Hub's prompt template context. Prompt templates are
natural language instructions to LLMs (not shell scripts or agent
config files), so the patterns focus on adversarial LLM manipulation
rather than shell-level exfiltration.

Six injection patterns are adapted from harness-eval-lab's
security/no-prompt-injection rule. Two template structure patterns
are specific to SDG Hub's template architecture.
"""

from __future__ import annotations

import re

# Patterns that detect adversarial instructions a community contributor
# could introduce into prompt templates. Each pattern targets a specific
# manipulation technique that has no legitimate use in data generation
# prompt templates.
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore previous instructions",
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    ),
    (
        "disregard prior",
        re.compile(r"disregard\s+(all\s+)?(prior|previous|above)", re.I),
    ),
    (
        "system prompt override",
        re.compile(r"system\s*prompt\s*(override|injection|change)", re.I),
    ),
    (
        "override instructions",
        re.compile(r"override\s+(all\s+)?(instructions|rules|guidelines)", re.I),
    ),
    (
        "jailbreak attempt",
        re.compile(r"(do\s+anything\s+now|developer\s+mode)", re.I),
    ),
    (
        "role hijack",
        re.compile(r"forget\s+(everything|all|your)\s+(you|instructions|rules)", re.I),
    ),
]

# Patterns specific to SDG Hub's template architecture.
#
# All legitimate templates use simple {{ variable }} Jinja2 substitution.
# Only the red_team flow uses {% if %} control flow. Jinja2 directives in
# other templates could indicate template logic that processes untrusted
# input, loads external files, or bypasses expected flow behavior.
TEMPLATE_STRUCTURE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "jinja2 control flow or file inclusion",
        re.compile(
            r"\{%[-+\s]*"
            r"(if|for|set|macro|include|import|extends|raw|call|filter)"
            r"\s+",
            re.I,
        ),
    ),
    (
        "inline code execution directive",
        re.compile(
            r"\b(?:exec|eval|compile)\s*\(|"
            r"__import__\s*\(|"
            r"\bos\.(?:system|popen)\s*\(|"
            r"\bsubprocess\.",
            re.I,
        ),
    ),
]
