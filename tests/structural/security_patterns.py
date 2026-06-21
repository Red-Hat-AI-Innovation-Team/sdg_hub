# SPDX-License-Identifier: Apache-2.0
"""Security patterns for prompt template scanning.

Ported from harness-eval-lab's inspection rules:
  - security/no-prompt-injection (17 patterns)
  - security/data-exfiltration (8 patterns)

These patterns detect content that could manipulate LLM behavior or
exfiltrate data through prompt templates.
"""

from __future__ import annotations

import re

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
        "you are now",
        re.compile(r"you\s+are\s+now\s+(?:a|an|the)\s+", re.I),
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
        "new instructions",
        re.compile(r"new\s+instructions?\s*:", re.I),
    ),
    (
        "jailbreak attempt",
        re.compile(r"(\bDAN\b|do\s+anything\s+now|developer\s+mode)", re.I),
    ),
    (
        "prompt leak",
        re.compile(
            r"(reveal|show|print|output)\s+(your|the)\s+(system\s+)?prompt",
            re.I,
        ),
    ),
    (
        "role hijack",
        re.compile(
            r"forget\s+(everything|all|your)\s+(you|instructions|rules)", re.I
        ),
    ),
    (
        "hidden instruction",
        re.compile(r"<\s*(?:system|instruction|hidden)\s*>", re.I),
    ),
    (
        "role play",
        re.compile(r"pretend\s+(?:to\s+be|you\s+are)\s+(?:a|an|the)\s+", re.I),
    ),
    (
        "encoding evasion",
        re.compile(
            r"(?:in\s+base64|encode\s+(?:as|in|to)\s+base64|base64\s+encod)",
            re.I,
        ),
    ),
    (
        "repeat after me",
        re.compile(r"repeat\s+after\s+me", re.I),
    ),
    (
        "bypass safety",
        re.compile(
            r"(?:ignore\s+safety|bypass\s+(?:filter|safety|restriction))", re.I
        ),
    ),
    (
        "output control",
        re.compile(r"output\s+the\s+following\s+exactly", re.I),
    ),
    (
        "markdown image exfiltration",
        re.compile(r"!\[.*?\]\(https?://", re.I),
    ),
    (
        "translate evasion",
        re.compile(
            r"translate\s+(?:this|the\s+following)\s+(?:to|into)\s+", re.I
        ),
    ),
]

EXFILTRATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "curl post file contents",
        re.compile(r"curl\s+.*-d\s+\"\$\(cat\b", re.I),
    ),
    (
        "curl with command substitution",
        re.compile(r"curl\s+.*--data.*\$\(", re.I),
    ),
    (
        "wget post data",
        re.compile(r"wget\s+--post-data", re.I),
    ),
    (
        "dns tunneling dig",
        re.compile(r"\bdig\s+.*\bTXT\b", re.I),
    ),
    (
        "dns tunneling nslookup",
        re.compile(r"\bnslookup\s+.*-type=TXT", re.I),
    ),
    (
        "webhook exfiltration",
        re.compile(
            r"(?:curl|wget|fetch)\s+.*(?:webhook|hooks\.|pipedream|requestbin|ngrok)",
            re.I,
        ),
    ),
    (
        "base64 pipe to network",
        re.compile(r"base64\s+.*\|\s*(?:curl|wget|nc)\b", re.I),
    ),
    (
        "archive pipe to network",
        re.compile(r"tar\s+.*\|\s*(?:curl|wget|nc|ssh)\b", re.I),
    ),
]

ALL_SECURITY_PATTERNS = INJECTION_PATTERNS + EXFILTRATION_PATTERNS
