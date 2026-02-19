# SPDX-License-Identifier: Apache-2.0
"""Translate an SDG Hub flow and its prompt YAMLs to a target language.

Provides a generic ``translate_flow()`` function that works with **any** flow.
Prompt YAMLs and structural tags are auto-discovered from the flow YAML — no
hardcoded file lists.  Accepts a flow id, flow name, or filesystem path.

Usage as a module::

    from sdg_hub.core.utils.translation import translate_flow

    issues = translate_flow(
        flow="enhanced_summary_knowledge_tuning",
        lang="Spanish",
        lang_code="es",
    )

Usage as a script::

    python -m sdg_hub.core.utils.translation \\
        --flow enhanced_summary_knowledge_tuning \\
        --lang French --lang-code fr

Environment variables:
    SDG_TRANSLATION_API_KEY  - API key for the translator model
    SDG_TRANSLATION_API_BASE - API base URL (optional, for custom endpoints)
    SDG_TRANSLATION_VERIFIER_API_KEY     - API key for the verifier model (if different)
    SDG_TRANSLATION_VERIFIER_API_BASE    - API base URL for verifier (optional)
"""

from __future__ import annotations

from pathlib import Path
import argparse
import copy
import logging
import os
import re
import sys

import litellm
import yaml

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4.1-2025-04-14"


# ---------------------------------------------------------------------------
# YAML block-scalar dumper
# ---------------------------------------------------------------------------


class _BlockStyleDumper(yaml.SafeDumper):
    pass


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_BlockStyleDumper.add_representer(str, _str_representer)


# ---------------------------------------------------------------------------
# Flow resolution
# ---------------------------------------------------------------------------


def _resolve_flow_path(flow: str) -> Path:
    """Resolve *flow* to an absolute path to a single ``flow.yaml`` file.

    *flow* can be:
    1. A flow **id** or **name** registered with ``FlowRegistry``.
    2. A filesystem path to a ``flow.yaml`` file.
    """
    # Try FlowRegistry first (by id or name)
    try:
        from sdg_hub.core.flow.registry import FlowRegistry

        registry_path = FlowRegistry.get_flow_path(flow)
        if registry_path is not None:
            resolved = Path(registry_path).resolve()
            log.debug("Resolved flow %r via FlowRegistry -> %s", flow, resolved)
            return resolved
    except Exception:  # noqa: BLE001
        log.debug("FlowRegistry lookup failed for %r, trying filesystem", flow)

    # Fall back to filesystem path
    p = Path(flow).resolve()
    if p.is_file():
        return p

    raise SystemExit(
        f"ERROR: '{flow}' was not found in FlowRegistry and does not exist as "
        f"a filesystem path.\n"
        f"  Hint: use a registered flow id/name, or pass a path to a flow.yaml file."
    )


# ---------------------------------------------------------------------------
# Flow & prompt discovery
# ---------------------------------------------------------------------------


def _is_flow_yaml(path: Path) -> bool:
    """Return True if *path* looks like an SDG Hub flow YAML (has metadata + blocks)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return isinstance(data, dict) and "metadata" in data and "blocks" in data
    except Exception:  # noqa: BLE001
        return False


def discover_prompt_yamls(flow_yaml_path: Path) -> dict[Path, str]:
    """Extract ``prompt_config_path`` references from a single flow YAML.

    Returns a dict mapping **resolved absolute path** -> the basename of the
    prompt file.  All prompts are assumed to reside in the same directory as
    the flow YAML (flat prompt layout).
    """
    with open(flow_yaml_path, encoding="utf-8") as f:
        flow_def = yaml.safe_load(f)

    flow_dir = flow_yaml_path.parent
    prompts: dict[Path, str] = {}

    for block in flow_def.get("blocks", []):
        config = block.get("block_config", {})
        if "prompt_config_path" in config:
            rel_path = config["prompt_config_path"]
            abs_path = (flow_dir / rel_path).resolve()
            if abs_path not in prompts:
                # Store just the filename (flat layout assumption)
                prompts[abs_path] = abs_path.name

    return prompts


# ---------------------------------------------------------------------------
# Structural tag extraction
# ---------------------------------------------------------------------------


def extract_structural_tags(flow_yaml_path: Path) -> frozenset[str]:
    """Extract structural tags from ``TagParserBlock`` configs in a flow YAML.

    These are tags the parser uses to split LLM output into fields — they must
    **not** be translated.
    """
    with open(flow_yaml_path, encoding="utf-8") as f:
        flow_def = yaml.safe_load(f)

    tags: set[str] = set()
    for block in flow_def.get("blocks", []):
        if block.get("block_type") == "TagParserBlock":
            config = block.get("block_config", {})
            for tag in config.get("start_tags", []):
                if tag:
                    tags.add(tag)
            for tag in config.get("end_tags", []):
                if tag:
                    tags.add(tag)

    return frozenset(tags)


# ---------------------------------------------------------------------------
# Translation prompt builder
# ---------------------------------------------------------------------------


def _build_translation_system_prompt(
    target_language: str,
    structural_tags: frozenset[str],
) -> str:
    """Build the system prompt for the translator LLM, including detected tags."""
    if structural_tags:
        tag_list = ", ".join(sorted(structural_tags))
        tag_rule = (
            f"- DO NOT translate parsing/structural tags: "
            f"{tag_list} must remain exactly as-is"
        )
    else:
        tag_rule = "- There are no structural parsing tags to preserve in this flow"

    return (
        f"You are an expert translator. Translate the following LLM prompt "
        f"instructions to {target_language}.\n\n"
        f"Rules:\n"
        f"- Translate ALL instructional text, examples, descriptions, and "
        f"natural language content\n"
        f"- DO NOT translate Jinja2 template variables: "
        f"{{{{variable_name}}}} must remain exactly as-is\n"
        f"{tag_rule}\n"
        f"- DO NOT translate markdown heading markers (###, ##, etc.) "
        f"but DO translate the heading text\n"
        f"- Preserve all markdown formatting (bold, italics, code blocks, "
        f"lists)\n"
        f"- Preserve the exact whitespace and newline structure of the "
        f"original\n"
        f"- Output ONLY the translated text, nothing else — no preamble, "
        f"no explanation"
    )


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def translate_text(
    text: str,
    target_language: str,
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    *,
    system_prompt: str | None = None,
) -> str:
    """Translate a single text block using the configured LLM."""
    kwargs: dict = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    if system_prompt is None:
        system_prompt = _build_translation_system_prompt(target_language, frozenset())

    try:
        response = litellm.completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=8192,
            temperature=0.1,
            **kwargs,
        )
    except litellm.AuthenticationError:
        raise SystemExit(
            "Authentication failed for translator model. "
            "Please set SDG_TRANSLATION_API_KEY in your environment."
        ) from None
    if not response.choices:
        log.warning("Translator returned no choices (model=%s)", model)
        return ""
    content = response.choices[0].message.content or ""
    log.debug("Translator response length: %d chars", len(content))
    if not content:
        log.warning("Translator returned empty response (model=%s)", model)
    return content


# ---------------------------------------------------------------------------
# Programmatic validation
# ---------------------------------------------------------------------------


def validate_translation(
    source: str,
    translated: str,
    structural_tags: frozenset[str],
) -> list[str]:
    """Check that Jinja2 variables and structural tags are preserved."""
    issues: list[str] = []

    # Check Jinja2 template variables
    source_vars = set(re.findall(r"\{\{\w+\}\}", source))
    translated_vars = set(re.findall(r"\{\{\w+\}\}", translated))
    missing_vars = source_vars - translated_vars
    extra_vars = translated_vars - source_vars
    if missing_vars:
        issues.append(f"Missing Jinja2 variables: {missing_vars}")
    if extra_vars:
        issues.append(f"Unexpected Jinja2 variables: {extra_vars}")

    # Check structural tags — only verify tags present in source
    source_tags = {tag for tag in structural_tags if tag in source}
    missing_tags = {tag for tag in source_tags if tag not in translated}
    if missing_tags:
        issues.append(f"Missing structural tags: {missing_tags}")

    return issues


# ---------------------------------------------------------------------------
# LLM verification
# ---------------------------------------------------------------------------

_VERIFICATION_SYSTEM_PROMPT = """\
You are verifying a translation of an LLM prompt from English to \
{target_language}.

You will receive the original English text inside <original_english> tags \
and the translated text inside <{target_language_lower}_translation> tags. \
These XML wrapper tags are NOT part of the content — ignore them entirely.

Check ONLY the content inside the tags for:
1. Semantic accuracy — does the translation convey the same meaning?
2. Completeness — are all instructions, examples, and guidelines translated?
3. Preserved elements — Jinja2 variables ({{{{var}}}}) and structural tags \
{structural_tag_examples}must NOT be translated

Respond with ONLY one of:
- PASS — if the translation is accurate and complete
- FAIL: <brief reason> — if there are issues that need fixing"""


def verify_translation(
    source: str,
    translated: str,
    target_language: str,
    model: str,
    structural_tags: frozenset[str],
    api_key: str | None = None,
    api_base: str | None = None,
) -> str:
    """Verify translation quality using a second LLM.

    Returns ``'PASS'`` or ``'FAIL: <reason>'``.
    """
    kwargs: dict = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    if structural_tags:
        examples = "like " + ", ".join(sorted(structural_tags)[:5]) + " "
    else:
        examples = ""

    sys_prompt = _VERIFICATION_SYSTEM_PROMPT.format(
        target_language=target_language,
        target_language_lower=target_language.lower(),
        structural_tag_examples=examples,
    )

    user_message = (
        f"<original_english>\n{source}\n</original_english>\n\n"
        f"<{target_language.lower()}_translation>"
        f"\n{translated}\n"
        f"</{target_language.lower()}_translation>"
    )

    try:
        response = litellm.completion(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=256,
            temperature=0.0,
            **kwargs,
        )
    except litellm.AuthenticationError:
        raise SystemExit(
            "Authentication failed for verifier model. "
            "Please set SDG_TRANSLATION_VERIFIER_API_KEY (or SDG_TRANSLATION_API_KEY) "
            "in your environment."
        ) from None
    if not response.choices:
        log.warning("Verifier returned no choices (model=%s)", model)
        return "FAIL: verifier returned no choices"
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
    """Strip trailing whitespace per line for YAML block-scalar compat."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


# ---------------------------------------------------------------------------
# Translate-and-verify loop
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
    *,
    structural_tags: frozenset[str],
    system_prompt: str,
) -> tuple[str, list[str]]:
    """Translate with retry loop driven by programmatic + LLM verification."""
    issues: list[str] = []
    translated_content = content  # fallback if max_retries == 0

    for attempt in range(1, max_retries + 1):
        log.debug("%s: attempt %d/%d", label, attempt, max_retries)

        translated_content = translate_text(
            content,
            target_language,
            translator_model,
            translator_api_key,
            translator_api_base,
            system_prompt=system_prompt,
        )
        translated_content = _clean_content(translated_content)

        log.debug(
            "%s: translated %d -> %d chars",
            label,
            len(content),
            len(translated_content),
        )

        # Programmatic validation
        prog_issues = validate_translation(content, translated_content, structural_tags)
        if prog_issues:
            log.debug("%s: programmatic issues: %s", label, prog_issues)

        # LLM verification
        verdict = verify_translation(
            content,
            translated_content,
            target_language,
            verifier_model,
            structural_tags,
            verifier_api_key,
            verifier_api_base,
        )
        log.debug("%s: verifier verdict: %r", label, verdict)

        passed = verdict.startswith("PASS") and not prog_issues

        if passed:
            if attempt > 1:
                print(f"      ✓ Passed on attempt {attempt}")
            return translated_content, []

        # Build failure reason for logging
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


# ---------------------------------------------------------------------------
# Prompt YAML translation
# ---------------------------------------------------------------------------


def _translate_prompt_yaml(
    source_path: Path,
    output_path: Path,
    target_language: str,
    translator_model: str,
    translator_api_key: str | None,
    translator_api_base: str | None,
    verifier_model: str,
    verifier_api_key: str | None,
    verifier_api_base: str | None,
    max_retries: int,
    *,
    structural_tags: frozenset[str],
    system_prompt: str,
) -> list[str]:
    """Translate a prompt YAML file. Returns unresolved validation issues."""
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
                content,
                target_language,
                translator_model,
                translator_api_key,
                translator_api_base,
                verifier_model,
                verifier_api_key,
                verifier_api_base,
                max_retries,
                label,
                structural_tags=structural_tags,
                system_prompt=system_prompt,
            )
            all_issues.extend(issues)
            translated_msg["content"] = translated_content
        translated_messages.append(translated_msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            translated_messages,
            f,
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
    lang_code: str,
) -> None:
    """Adapt a flow YAML's metadata and prompt paths for *target_language*.

    Assumes flat prompt layout: all prompt files reside in the same directory
    as the flow YAML.  The ``prompt_config_path`` values are rewritten to
    just the translated filename (e.g. ``my_prompt_es.yaml``).
    """
    with open(source_path, encoding="utf-8") as f:
        flow_def = yaml.safe_load(f)

    flow_def = copy.deepcopy(flow_def)
    meta = flow_def["metadata"]

    meta["name"] = f"{meta['name']} ({target_language})"
    meta["id"] = f"{meta['id']}-{lang_code}"

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
            # Extract just the filename (strip any ../ or subdirectory prefix)
            # and rewrite to prompts/<name>_<lang_code>.yaml.
            old_basename = Path(config["prompt_config_path"]).name
            new_basename = old_basename.replace(".yaml", f"_{lang_code}.yaml")
            config["prompt_config_path"] = f"prompts/{new_basename}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            flow_def,
            f,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
            sort_keys=False,
        )
    print(f"  ✓ {output_path.name}")
    print(f"    Name: {meta['name']}")
    print(f"    ID:   {meta['id']}")


# ---------------------------------------------------------------------------
# Output path computation
# ---------------------------------------------------------------------------


def _compute_output_paths(
    flow_yaml: Path,
    prompt_yamls: dict[Path, str],
    output_dir: Path,
    lang_code: str,
) -> tuple[Path, dict[Path, Path]]:
    """Compute output paths for a single flow and its prompts.

    Output layout::

        output_dir/
        ├── flow.yaml
        └── prompts/
            ├── prompt_a_<lang_code>.yaml
            └── prompt_b_<lang_code>.yaml

    Returns (flow_out_path, prompt_src->out mapping).
    """
    flow_out = output_dir / flow_yaml.name
    prompts_dir = output_dir / "prompts"

    prompt_mapping: dict[Path, Path] = {}
    for prompt_src in prompt_yamls:
        stem = prompt_src.stem
        out_name = f"{stem}_{lang_code}.yaml"
        prompt_mapping[prompt_src] = prompts_dir / out_name

    return flow_out, prompt_mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def translate_flow(
    flow: str,
    lang: str,
    lang_code: str,
    translator_model: str = _DEFAULT_MODEL,
    verifier_model: str = _DEFAULT_MODEL,
    output_dir: str | None = None,
    *,
    translator_api_key: str | None = None,
    translator_api_base: str | None = None,
    verifier_api_key: str | None = None,
    verifier_api_base: str | None = None,
    max_retries: int = 3,
    verbose: bool = False,
    register: bool = True,
) -> list[str]:
    """Translate a single flow and its prompt YAMLs to a target language.

    Parameters
    ----------
    flow
        A registered flow **id** or **name** (looked up via ``FlowRegistry``),
        or a filesystem path to a ``flow.yaml`` file.
    lang
        Target language name (e.g. ``"Spanish"``).
    lang_code
        ISO 639-1 language code (e.g. ``"es"``).
    translator_model
        Model identifier for translation (default: ``"gpt-4.1-2025-04-14"``).
    verifier_model
        Model identifier for verification (default: ``"gpt-4.1-2025-04-14"``).
    output_dir
        Directory where translated flows will be written.  If ``None``
        (default), created in the current working directory as
        ``<source_flow_dir_name>_<lang_code>/``.
    translator_api_key, translator_api_base
        Optional API credentials for the translator model.
    verifier_api_key, verifier_api_base
        Optional API credentials for the verifier model.
    max_retries
        Maximum translation attempts per prompt message on verifier failure.
    verbose
        If ``True``, enable ``DEBUG``-level logging.
    register
        If ``True`` (default), register the output directory with
        ``FlowRegistry`` so the translated flows are immediately discoverable.

    Returns
    -------
    list[str]
        Unresolved validation issues (empty list = all passed).
    """
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s: %(message)s",
        )

    # Read API keys from environment if not provided as parameters
    if translator_api_key is None:
        translator_api_key = os.getenv("SDG_TRANSLATION_API_KEY", "").strip() or None
    if translator_api_base is None:
        translator_api_base = os.getenv("SDG_TRANSLATION_API_BASE", "").strip() or None
    if verifier_api_key is None:
        verifier_api_key = (
            os.getenv("SDG_TRANSLATION_VERIFIER_API_KEY", "").strip()
            or os.getenv("SDG_TRANSLATION_API_KEY", "").strip()
            or None
        )
    if verifier_api_base is None:
        verifier_api_base = (
            os.getenv("SDG_TRANSLATION_VERIFIER_API_BASE", "").strip() or None
        )

    # Resolve flow identifier to a filesystem path
    flow_yaml = _resolve_flow_path(flow)

    # Derive default output_dir from source flow's parent directory name.
    # Uses CWD to avoid writing into pip-installed / read-only paths.
    if output_dir is None:
        source_dir_name = flow_yaml.parent.name
        output_path = Path.cwd() / f"{source_dir_name}_{lang_code}"
    else:
        output_path = Path(output_dir).resolve()

    # 1. Discover prompt YAMLs referenced by this flow
    prompt_yamls = discover_prompt_yamls(flow_yaml)

    # 2. Auto-detect structural tags
    structural_tags = extract_structural_tags(flow_yaml)

    # 3. Build translation system prompt with detected tags
    system_prompt = _build_translation_system_prompt(lang, structural_tags)

    # 4. Compute output paths (flat layout)
    flow_out, prompt_mapping = _compute_output_paths(
        flow_yaml, prompt_yamls, output_path, lang_code
    )

    # Print summary
    print(f"Flow: {flow}")
    print(f"Source path: {flow_yaml}")
    print(f"Target language: {lang} ({lang_code})")
    print(f"Translator model: {translator_model}")
    print(f"Verifier model: {verifier_model}")
    print(f"Max retries: {max_retries}")
    print(f"Output directory: {output_path}")
    print(f"Files to process: {len(prompt_mapping)} prompts + 1 flow")
    if structural_tags:
        print(f"Structural tags detected: {sorted(structural_tags)}")
    else:
        print("Structural tags: none detected")
    print()

    # 5. Translate prompt YAMLs
    print("=" * 60)
    print("Step 1: Translating prompt files")
    print("=" * 60)
    all_issues: list[str] = []
    for source_path, out_path in prompt_mapping.items():
        print(f"\n  {source_path.name}")
        issues = _translate_prompt_yaml(
            source_path,
            out_path,
            lang,
            translator_model,
            translator_api_key,
            translator_api_base,
            verifier_model,
            verifier_api_key,
            verifier_api_base,
            max_retries,
            structural_tags=structural_tags,
            system_prompt=system_prompt,
        )
        all_issues.extend(issues)

    print(f"\n✓ All {len(prompt_mapping)} prompt files translated")

    # 6. Adapt flow YAML
    print()
    print("=" * 60)
    print("Step 2: Creating flow definition")
    print("=" * 60)
    adapt_flow_yaml(flow_yaml, flow_out, lang, lang_code)

    print("\n✓ Flow file created")

    # 7. Summary
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

    print(f"\nFiles created in: {output_path}")
    print("\nTo use these flows, set in your .env:")
    print(f"  SDG_LANG={lang}")
    print(f"  SDG_LANG_CODE={lang_code}")

    # 8. Register translated flows with FlowRegistry
    if register:
        from sdg_hub.core.flow.registry import FlowRegistry

        FlowRegistry.register_search_path(str(output_path))
        FlowRegistry._discover_flows(force_refresh=True)
        print(f"\n✓ Registered {output_path} with FlowRegistry")

    return all_issues


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``python -m sdg_hub.core.utils.translation``."""
    import os

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Translate SDG Hub flows to a new language.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--flow",
        required=True,
        help=(
            "Flow id, flow name, or path to a flow.yaml file "
            "(e.g., 'enhanced_summary_knowledge_tuning' or './my-flow/flow.yaml')"
        ),
    )
    parser.add_argument(
        "--lang",
        required=True,
        help="Target language name (e.g., 'Spanish', 'French')",
    )
    parser.add_argument(
        "--lang-code",
        required=True,
        help="ISO 639-1 language code (e.g., 'es', 'fr')",
    )
    parser.add_argument(
        "--translator-model",
        default=_DEFAULT_MODEL,
        help=f"Translator model (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--verifier-model",
        default=_DEFAULT_MODEL,
        help=f"Verifier model (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <source_flow_dir>_<lang_code>/)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max translation retries on verifier failure (default: 3)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging for translator/verifier responses",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Do not register translated flows with FlowRegistry",
    )
    args = parser.parse_args()

    if not args.verbose:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # API keys from env
    translator_api_key = os.getenv("SDG_TRANSLATION_API_KEY", "").strip() or None
    translator_api_base = os.getenv("SDG_TRANSLATION_API_BASE", "").strip() or None
    verifier_api_key = (
        os.getenv("SDG_TRANSLATION_VERIFIER_API_KEY", "").strip()
        or os.getenv("SDG_TRANSLATION_API_KEY", "").strip()
        or None
    )
    verifier_api_base = (
        os.getenv("SDG_TRANSLATION_VERIFIER_API_BASE", "").strip() or None
    )

    issues = translate_flow(
        flow=args.flow,
        lang=args.lang,
        lang_code=args.lang_code,
        translator_model=args.translator_model,
        verifier_model=args.verifier_model,
        output_dir=args.output_dir,
        translator_api_key=translator_api_key,
        translator_api_base=translator_api_base,
        verifier_api_key=verifier_api_key,
        verifier_api_base=verifier_api_base,
        max_retries=args.max_retries,
        verbose=args.verbose,
        register=not args.no_register,
    )

    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
