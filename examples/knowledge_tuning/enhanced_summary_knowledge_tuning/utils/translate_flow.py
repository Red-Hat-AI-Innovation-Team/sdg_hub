#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Translate knowledge tuning flows to a new language.

Translates the Enhanced Multi-Summary QA flows (prompt YAMLs and flow definitions)
to a target language using an LLM translator and LLM verifier.

Usage:
    python utils/translate_flow.py --lang Spanish --lang-code es \\
        --translator openai/gpt-4o --verifier claude-sonnet-4-20250514

    python utils/translate_flow.py --lang French --lang-code fr \\
        --translator openai/gpt-4o --verifier openai/gpt-4o --max-retries 5

Environment variables (from .env):
    TRANSLATION_API_KEY  - API key for the translator model
    TRANSLATION_API_BASE - API base URL (optional, for custom endpoints)
    VERIFIER_API_KEY     - API key for the verifier model (if different from translator)
    VERIFIER_API_BASE    - API base URL for verifier (optional)
"""

from __future__ import annotations

import argparse
import copy
import logging
import re
import sys
from pathlib import Path

import litellm
import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML block-scalar dumper
# ---------------------------------------------------------------------------

class _BlockStyleDumper(yaml.SafeDumper):
    pass


def _str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_BlockStyleDumper.add_representer(str, _str_representer)

# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

TRANSLATION_SYSTEM_PROMPT = """You are an expert translator. Translate the following LLM prompt instructions to {target_language}.

Rules:
- Translate ALL instructional text, examples, descriptions, and natural language content
- DO NOT translate Jinja2 template variables: {{{{variable_name}}}} must remain exactly as-is
- DO NOT translate parsing/structural tags: [QUESTION], [ANSWER], [END], [Start of Context], [End of Context], [Start of Response], [End of Response], [Start of Explanation], [End of Explanation], [Start of Answer], [End of Answer] must remain in English
- DO NOT translate markdown heading markers (###, ##, etc.) but DO translate the heading text
- Preserve all markdown formatting (bold, italics, code blocks, lists)
- Preserve the exact whitespace and newline structure of the original
- Output ONLY the translated text, nothing else — no preamble, no explanation"""


def translate_text(
    text: str,
    target_language: str,
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
) -> str:
    """Translate a single text block using the configured LLM."""
    kwargs: dict = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.completion(
        messages=[
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT.format(target_language=target_language)},
            {"role": "user", "content": text},
        ],
        max_tokens=8192,
        temperature=0.1,
        **kwargs,
    )
    content = response.choices[0].message.content or ""
    log.debug("Translator response length: %d chars", len(content))
    if not content:
        log.warning("Translator returned empty response (model=%s)", model)
    return content


# ---------------------------------------------------------------------------
# Programmatic validation
# ---------------------------------------------------------------------------

_STRUCTURAL_TAGS = frozenset({
    "[QUESTION]",
    "[ANSWER]",
    "[END]",
    "[Document]",
    "[DOCUMENT]",
    "[Key Fact]",
    "[Start of Context]",
    "[End of Context]",
    "[Start of Response]",
    "[End of Response]",
    "[Start of Explanation]",
    "[End of Explanation]",
    "[Start of Answer]",
    "[End of Answer]",
})


def validate_translation(source: str, translated: str) -> list[str]:
    """Check that Jinja2 variables and structural tags are preserved."""
    issues = []

    # Check Jinja2 template variables
    source_vars = set(re.findall(r"\{\{\w+\}\}", source))
    translated_vars = set(re.findall(r"\{\{\w+\}\}", translated))
    missing_vars = source_vars - translated_vars
    extra_vars = translated_vars - source_vars
    if missing_vars:
        issues.append(f"Missing Jinja2 variables: {missing_vars}")
    if extra_vars:
        issues.append(f"Unexpected Jinja2 variables: {extra_vars}")

    # Check structural tags — only verify tags from the known allowlist
    source_tags = {tag for tag in _STRUCTURAL_TAGS if tag in source}
    missing_tags = {tag for tag in source_tags if tag not in translated}
    if missing_tags:
        issues.append(f"Missing structural tags: {missing_tags}")

    return issues


# ---------------------------------------------------------------------------
# LLM verification
# ---------------------------------------------------------------------------

VERIFICATION_SYSTEM_PROMPT = """You are verifying a translation of an LLM prompt from English to {target_language}.

You will receive the original English text inside <original_english> tags and the translated text inside <{target_language_lower}_translation> tags. These XML wrapper tags are NOT part of the content — ignore them entirely.

Check ONLY the content inside the tags for:
1. Semantic accuracy — does the translation convey the same meaning and instructions?
2. Completeness — are all instructions, examples, and guidelines translated?
3. Preserved elements — Jinja2 variables ({{{{var}}}}) and structural tags like [QUESTION], [END], [Start of Context], etc. must NOT be translated

Respond with ONLY one of:
- PASS — if the translation is accurate and complete
- FAIL: <brief reason> — if there are issues that need fixing"""


def verify_translation(
    source: str,
    translated: str,
    target_language: str,
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
) -> str:
    """Verify translation quality using a second LLM.

    Returns 'PASS' if the translation is accurate, or 'FAIL: <reason>' if issues
    are found.
    """
    kwargs: dict = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    user_message = (
        f"<original_english>\n{source}\n</original_english>\n\n"
        f"<{target_language.lower()}_translation>\n{translated}\n</{target_language.lower()}_translation>"
    )

    response = litellm.completion(
        messages=[
            {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT.format(target_language=target_language, target_language_lower=target_language.lower())},
            {"role": "user", "content": user_message},
        ],
        max_tokens=2048,
        reasoning_effort="low",
        **kwargs,
    )
    verdict = (response.choices[0].message.content or "").strip()
    log.debug("Verifier raw response: %r", verdict)
    if not verdict:
        log.warning("Verifier returned empty response (model=%s)", model)
        return "FAIL: verifier returned empty response"
    return verdict


# ---------------------------------------------------------------------------
# Clean content for YAML output
# ---------------------------------------------------------------------------

def _clean_content(text: str) -> str:
    """Strip trailing whitespace from each line for YAML block scalar compatibility."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


# ---------------------------------------------------------------------------
# Prompt YAML translation
# ---------------------------------------------------------------------------

def _translate_and_verify(
    content: str,
    target_language: str,
    translator_model: str,
    translator_api_key: str | None,
    translator_api_base: str | None,
    verifier_model: str,
    verifier_api_key: str | None,
    verifier_api_base: str | None,
    max_retries: int,
    label: str,
) -> tuple[str, list[str]]:
    """Translate content with a retry loop driven by the verifier.

    Returns (translated_content, list_of_unresolved_issues).
    """
    issues: list[str] = []

    for attempt in range(1, max_retries + 1):
        log.debug("%s: attempt %d/%d", label, attempt, max_retries)

        translated_content = translate_text(
            content, target_language, translator_model, translator_api_key, translator_api_base,
        )
        translated_content = _clean_content(translated_content)

        log.debug("%s: translated %d → %d chars", label, len(content), len(translated_content))

        # Programmatic validation
        prog_issues = validate_translation(content, translated_content)
        if prog_issues:
            log.debug("%s: programmatic issues: %s", label, prog_issues)

        # LLM verification
        verdict = verify_translation(
            content, translated_content, target_language,
            verifier_model, verifier_api_key, verifier_api_base,
        )
        log.debug("%s: verifier verdict: %r", label, verdict)

        passed = verdict.startswith("PASS") and not prog_issues

        if passed:
            if attempt > 1:
                print(f"      ✓ Passed on attempt {attempt}")
            return translated_content, []

        # Log failure and retry
        issues = [f"{label}: {i}" for i in prog_issues]
        if not verdict.startswith("PASS"):
            issues.append(f"{label} verifier: {verdict}")

        reason = verdict
        if prog_issues:
            reason = "; ".join(prog_issues)
            if not verdict.startswith("PASS"):
                reason += f" | verifier: {verdict}"

        if attempt < max_retries:
            print(f"      ⚠ Attempt {attempt} failed, retrying... ({reason})")
        else:
            print(f"      ✗ Failed after {max_retries} attempts ({reason})")

    return translated_content, issues


def translate_prompt_yaml(
    source_path: Path,
    output_path: Path,
    target_language: str,
    translator_model: str,
    translator_api_key: str | None,
    translator_api_base: str | None,
    verifier_model: str,
    verifier_api_key: str | None,
    verifier_api_base: str | None,
    max_retries: int = 3,
) -> list[str]:
    """Translate a prompt YAML file and return any unresolved validation issues."""
    with open(source_path, encoding="utf-8") as f:
        messages = yaml.safe_load(f)

    all_issues: list[str] = []
    translated_messages = []

    for msg in messages:
        translated_msg = dict(msg)
        content = msg.get("content", "").strip()
        if content:
            label = f"  {source_path.name} [{msg['role']}]"
            print(f"    Translating {msg['role']} message ({len(content)} chars)...")

            translated_content, issues = _translate_and_verify(
                content, target_language,
                translator_model, translator_api_key, translator_api_base,
                verifier_model, verifier_api_key, verifier_api_base,
                max_retries, label,
            )
            all_issues.extend(issues)
            translated_msg["content"] = translated_content
        translated_messages.append(translated_msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            translated_messages, f,
            Dumper=_BlockStyleDumper,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
            sort_keys=False,
        )
    print(f"    ✓ Saved: {output_path.name}")
    return all_issues


# ---------------------------------------------------------------------------
# Flow YAML adaptation
# ---------------------------------------------------------------------------

def adapt_flow_yaml(
    source_path: Path,
    output_path: Path,
    target_language: str,
    language_code: str,
) -> None:
    """Adapt a flow YAML for the target language."""
    with open(source_path, encoding="utf-8") as f:
        flow_def = yaml.safe_load(f)

    flow_def = copy.deepcopy(flow_def)
    meta = flow_def["metadata"]

    meta["name"] = f"{meta['name']} ({target_language})"
    meta["id"] = f"{meta['id']}-{language_code}"

    if "description" in meta:
        desc = meta["description"]
        if desc.endswith("."):
            desc = desc[:-1]
        meta["description"] = f"{desc} in {target_language}."

    if "tags" in meta:
        lang_tag = target_language.lower()
        if lang_tag not in meta["tags"]:
            meta["tags"].append(lang_tag)

    if "dataset_requirements" in meta and "description" in meta["dataset_requirements"]:
        req_desc = meta["dataset_requirements"]["description"]
        req_desc = req_desc.replace(
            "Input dataset should contain documents",
            f"Input dataset should contain {target_language} documents",
        )
        meta["dataset_requirements"]["description"] = req_desc

    for block in flow_def.get("blocks", []):
        config = block.get("block_config", {})
        if "prompt_config_path" in config:
            old_path = config["prompt_config_path"]
            config["prompt_config_path"] = old_path.replace(".yaml", f"_{language_code}.yaml")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            flow_def, f,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
            sort_keys=False,
        )
    print(f"  ✓ {output_path.relative_to(output_path.parent.parent)}")
    print(f"    Name: {meta['name']}")
    print(f"    ID:   {meta['id']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    import os

    parser = argparse.ArgumentParser(
        description="Translate knowledge tuning flows to a new language.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--lang", required=True, help="Target language name (e.g., 'Spanish', 'French')")
    parser.add_argument("--lang-code", required=True, help="ISO 639-1 language code (e.g., 'es', 'fr')")
    parser.add_argument("--translator", required=True, help="Translator model in litellm format (e.g., 'openai/gpt-4o')")
    parser.add_argument("--verifier", required=True, help="Verifier model in litellm format (e.g., 'claude-sonnet-4-20250514')")
    parser.add_argument("--max-retries", type=int, default=3, help="Max translation retries on verifier failure (default: 3)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging for translator/verifier responses")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ./translated_flows/<lang>/)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    target_language = args.lang
    language_code = args.lang_code
    translator_model = args.translator
    verifier_model = args.verifier
    max_retries = args.max_retries

    # API keys from env
    translator_api_key = os.getenv("TRANSLATION_API_KEY", "").strip() or None
    translator_api_base = os.getenv("TRANSLATION_API_BASE", "").strip() or None
    verifier_api_key = os.getenv("VERIFIER_API_KEY", "").strip() or os.getenv("TRANSLATION_API_KEY", "").strip() or None
    verifier_api_base = os.getenv("VERIFIER_API_BASE", "").strip() or None

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path("./translated_flows").resolve() / target_language.lower()

    # Locate source flows via FlowRegistry
    from sdg_hub import FlowRegistry

    FlowRegistry.discover_flows()
    english_flow_path = FlowRegistry.get_flow_path(
        "Extractive Summary Knowledge Tuning Dataset Generation Flow"
    )
    if not english_flow_path:
        print("ERROR: Could not find English source flows in registry", file=sys.stderr)
        sys.exit(1)

    source_dir = Path(english_flow_path).parent.parent

    # File mappings
    shared_prompts = {
        "evaluate_faithfulness.yaml": f"evaluate_faithfulness_{language_code}.yaml",
        "generate_answers.yaml": f"generate_answers_{language_code}.yaml",
        "generate_multiple_qa.yaml": f"generate_multiple_qa_{language_code}.yaml",
        "generate_question_list.yaml": f"generate_question_list_{language_code}.yaml",
    }
    flow_specific_prompts = {
        f"extractive_summary/extractive_summary.yaml": f"extractive_summary/extractive_summary_{language_code}.yaml",
        f"detailed_summary/detailed_summary.yaml": f"detailed_summary/detailed_summary_{language_code}.yaml",
        f"key_facts/key_facts_summary.yaml": f"key_facts/key_facts_summary_{language_code}.yaml",
    }
    flow_files = [
        "extractive_summary/flow.yaml",
        "detailed_summary/flow.yaml",
        "key_facts/flow.yaml",
        "doc_direct_qa/flow.yaml",
    ]
    all_prompts = {**shared_prompts, **flow_specific_prompts}

    print(f"Target language: {target_language} ({language_code})")
    print(f"Translator model: {translator_model}")
    print(f"Verifier model: {verifier_model}")
    print(f"Max retries: {max_retries}")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to create: {len(all_prompts)} prompts + {len(flow_files)} flows")
    print()

    # Step 1: Translate prompt YAMLs
    print("=" * 60)
    print("Step 1: Translating prompt files")
    print("=" * 60)
    all_issues: list[str] = []
    for source_rel, output_rel in all_prompts.items():
        source_path = source_dir / source_rel
        output_path = output_dir / output_rel
        print(f"\n  {source_rel}")
        issues = translate_prompt_yaml(
            source_path, output_path, target_language,
            translator_model, translator_api_key, translator_api_base,
            verifier_model, verifier_api_key, verifier_api_base,
            max_retries,
        )
        all_issues.extend(issues)

    print(f"\n✓ All {len(all_prompts)} prompt files translated")

    # Step 2: Adapt flow YAMLs
    print()
    print("=" * 60)
    print("Step 2: Creating flow definitions")
    print("=" * 60)
    for flow_rel in flow_files:
        source_path = source_dir / flow_rel
        output_path = output_dir / flow_rel
        adapt_flow_yaml(source_path, output_path, target_language, language_code)

    print(f"\n✓ All {len(flow_files)} flow files created")

    # Step 3: Verify with FlowRegistry
    print()
    print("=" * 60)
    print("Step 3: Verifying flow discovery")
    print("=" * 60)
    FlowRegistry.register_search_path(str(output_dir.parent))
    FlowRegistry._entries = {}
    FlowRegistry.discover_flows()

    lang_flows = FlowRegistry.search_flows(tag=target_language.lower())
    print(f"\nFound {len(lang_flows)} {target_language} flows:")
    for f in lang_flows:
        print(f"  - {f['name']} (ID: {f['id']})")

    if len(lang_flows) != 4:
        print(f"\n⚠ Expected 4 flows, found {len(lang_flows)}", file=sys.stderr)

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    if all_issues:
        print(f"\n⚠ {len(all_issues)} issue(s) found during validation:")
        for issue in all_issues:
            print(f"  {issue}")
        print("\nReview the flagged files before using them for data generation.")
    else:
        print("\n✓ All translations passed validation")

    print(f"\nFiles created in: {output_dir}")
    print(f"\nTo use these flows, set in your .env:")
    print(f"  SDG_LANG={target_language}")
    print(f"  SDG_LANG_CODE={language_code}")
    print(f"  TRANSLATED_FLOWS_DIR={output_dir.parent}")


if __name__ == "__main__":
    main()
