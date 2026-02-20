# SPDX-License-Identifier: Apache-2.0
"""Translate an SDG Hub flow and its prompt YAMLs to a target language.

Provides a generic ``translate_flow()`` function that works with **any** flow.
Prompt YAMLs and structural tags are auto-discovered from the flow YAML — no
hardcoded file lists.  Accepts a flow id or flow name.
"""

from __future__ import annotations

from pathlib import Path
import copy
import logging
import re

import litellm
import yaml

from ..blocks.llm.prompt_builder_block import PromptRenderer, PromptTemplateConfig
from ..flow.base import Flow
from ..flow.registry import FlowRegistry
from ..utils.error_handling import APIConnectionError
from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_TRANSLATION_PROMPT = PromptTemplateConfig(str(_PROMPTS_DIR / "translation.yaml"))
_VERIFICATION_PROMPT = PromptTemplateConfig(str(_PROMPTS_DIR / "verification.yaml"))


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
# Flow YAML parsing
# ---------------------------------------------------------------------------


def _parse_flow_yaml(
    flow_yaml_path: Path,
) -> tuple[dict[Path, str], frozenset[str]]:
    """Parse a flow YAML and return discovered prompts and structural tags.

    Returns (prompt_mapping, structural_tags) where prompt_mapping maps
    resolved absolute paths to prompt basenames, and structural_tags is the
    set of tags from ``TagParserBlock`` configs that must not be translated.
    """
    with open(flow_yaml_path, encoding="utf-8") as f:
        flow_def = yaml.safe_load(f)

    flow_dir = flow_yaml_path.parent
    flow_dir_resolved = flow_dir.resolve()
    prompts: dict[Path, str] = {}
    tags: set[str] = set()

    for block in flow_def.get("blocks", []):
        config = block.get("block_config", {})

        # Discover prompt paths
        if "prompt_config_path" in config:
            rel_path = config["prompt_config_path"]
            abs_path = (flow_dir / rel_path).resolve()
            if not abs_path.is_relative_to(flow_dir_resolved):
                logger.warning(
                    "Skipping prompt_config_path %r: resolves outside flow directory",
                    rel_path,
                )
                continue
            if abs_path not in prompts:
                prompts[abs_path] = abs_path.name

        # Extract structural tags
        if block.get("block_type") == "TagParserBlock":
            for tag in config.get("start_tags", []):
                if tag:
                    tags.add(tag)
            for tag in config.get("end_tags", []):
                if tag:
                    tags.add(tag)

    return prompts, frozenset(tags)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _build_tag_rule(structural_tags: frozenset[str]) -> str:
    """Build the tag-rule string for the translation system prompt."""
    if structural_tags:
        tag_list = ", ".join(sorted(structural_tags))
        return (
            f"- DO NOT translate parsing/structural tags: "
            f"{tag_list} must remain exactly as-is"
        )
    return "- There are no structural parsing tags to preserve in this flow"


def _render_prompt(
    config: PromptTemplateConfig, template_vars: dict[str, str]
) -> list[dict[str, str]]:
    """Render a prompt template to a list of message dicts for litellm."""
    renderer = PromptRenderer(config.get_message_templates())
    messages = renderer.render_messages(template_vars)
    return [{"role": m.role, "content": m.content} for m in messages]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _llm_call(
    messages: list[dict[str, str]],
    model: str,
    api_key: str | None,
    api_base: str | None,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    """Make a single litellm completion call and return the response text."""
    kwargs: dict = {"model": model}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if api_base is not None:
        kwargs["api_base"] = api_base

    try:
        response = litellm.completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
    except litellm.AuthenticationError as exc:
        raise APIConnectionError(
            f"Authentication failed for model {model}. Please check your api_key."
        ) from exc

    if not response.choices:
        logger.warning("LLM returned no choices (model=%s)", model)
        return ""
    content = (response.choices[0].message.content or "").strip()
    if not content:
        logger.warning("LLM returned empty response (model=%s)", model)
    return content


# ---------------------------------------------------------------------------
# Translation & verification
# ---------------------------------------------------------------------------


def _translate_text(
    text: str,
    target_language: str,
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    *,
    tag_rule: str | None = None,
) -> str:
    """Translate a single text block using the configured LLM."""
    if tag_rule is None:
        tag_rule = _build_tag_rule(frozenset())

    messages = _render_prompt(
        _TRANSLATION_PROMPT,
        {"target_language": target_language, "tag_rule": tag_rule, "text": text},
    )
    return _llm_call(
        messages, model, api_key, api_base, max_tokens=8192, temperature=0.1
    )


def _validate_translation(
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


def _verify_translation(
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
    if structural_tags:
        examples = "like " + ", ".join(sorted(structural_tags)[:5]) + " "
    else:
        examples = ""

    messages = _render_prompt(
        _VERIFICATION_PROMPT,
        {
            "target_language": target_language,
            "target_language_lower": target_language.lower(),
            "structural_tag_examples": examples,
            "source": source,
            "translated": translated,
        },
    )
    result = _llm_call(
        messages, model, api_key, api_base, max_tokens=256, temperature=0.0
    )
    return result or "FAIL: verifier returned empty response"


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
    tag_rule: str,
) -> tuple[str, list[str]]:
    """Translate with retry loop driven by programmatic + LLM verification."""
    issues: list[str] = []
    translated_content = content  # fallback if max_retries == 0

    for attempt in range(1, max_retries + 1):
        logger.debug("%s: attempt %d/%d", label, attempt, max_retries)

        translated_content = _translate_text(
            content,
            target_language,
            translator_model,
            translator_api_key,
            translator_api_base,
            tag_rule=tag_rule,
        )
        translated_content = _clean_content(translated_content)

        logger.debug(
            "%s: translated %d -> %d chars",
            label,
            len(content),
            len(translated_content),
        )

        # Programmatic validation
        prog_issues = _validate_translation(
            content, translated_content, structural_tags
        )
        if prog_issues:
            logger.debug("%s: programmatic issues: %s", label, prog_issues)

        # LLM verification
        verdict = _verify_translation(
            content,
            translated_content,
            target_language,
            verifier_model,
            structural_tags,
            verifier_api_key,
            verifier_api_base,
        )
        logger.debug("%s: verifier verdict: %r", label, verdict)

        passed = verdict.startswith("PASS") and not prog_issues

        if passed:
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
            logger.warning(
                "%s: attempt %d failed, retrying (%s)", label, attempt, reason
            )
        else:
            logger.warning(
                "%s: failed after %d attempts (%s)", label, max_retries, reason
            )

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
    tag_rule: str,
) -> list[str]:
    """Translate a prompt YAML file. Returns unresolved validation issues."""
    with open(source_path, encoding="utf-8") as f:
        messages = yaml.safe_load(f)

    if not isinstance(messages, list):
        raise ValueError(
            f"Prompt YAML {source_path} must be a list of messages, "
            f"got {type(messages).__name__}"
        )

    all_issues: list[str] = []
    translated_messages = []

    for msg in messages:
        translated_msg = dict(msg)
        content = msg.get("content", "").strip()
        if content:
            label = f"{source_path.name} [{msg['role']}]"
            logger.info("Translating %s (%d chars)", label, len(content))

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
                tag_rule=tag_rule,
            )
            all_issues.extend(issues)
            translated_msg["content"] = translated_content
        translated_messages.append(translated_msg)

    if all_issues:
        logger.warning(
            "Skipping write of %s due to validation issues", output_path.name
        )
        return all_issues

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
    logger.info("Saved %s", output_path.name)
    return all_issues


# ---------------------------------------------------------------------------
# Flow YAML adaptation
# ---------------------------------------------------------------------------


def _adapt_flow_yaml(
    source_path: Path,
    output_path: Path,
    target_language: str,
    lang_code: str,
    translated_prompts: set[str],
) -> None:
    """Create a translated copy of a flow YAML.

    Updates the flow id/name for uniqueness and rewrites
    ``prompt_config_path`` values to point at the translated prompt files.
    Only rewrites paths whose basename is in *translated_prompts*.
    """
    with open(source_path, encoding="utf-8") as f:
        flow_def = yaml.safe_load(f)

    flow_def = copy.deepcopy(flow_def)
    if "metadata" not in flow_def:
        raise ValueError(f"Flow YAML {source_path} is missing 'metadata' section")
    meta = flow_def["metadata"]
    meta["name"] = f"{meta['name']} ({target_language})"
    meta["id"] = f"{meta['id']}-{lang_code}"

    for block in flow_def.get("blocks", []):
        config = block.get("block_config", {})
        if "prompt_config_path" in config:
            old_basename = Path(config["prompt_config_path"]).name
            if old_basename not in translated_prompts:
                continue
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
    logger.info("Created %s (id=%s)", meta["name"], meta["id"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def translate_flow(
    flow: str,
    lang: str,
    lang_code: str,
    translator_model: str = "gpt-5.2",
    verifier_model: str = "gpt-5.2",
    output_dir: str | None = None,
    translator_api_key: str | None = None,
    translator_api_base: str | None = None,
    verifier_api_key: str | None = None,
    verifier_api_base: str | None = None,
    max_retries: int = 3,
    verbose: bool = False,
    register: bool = True,
) -> "Flow":
    """Translate a single flow and its prompt YAMLs to a target language.

    Parameters
    ----------
    flow
        A registered flow **id** or **name** (looked up via ``FlowRegistry``).
    lang
        Target language name (e.g. ``"Spanish"``).
    lang_code
        ISO 639-1 language code (e.g. ``"es"``).
    translator_model
        Model identifier for translation.
    verifier_model
        Model identifier for verification.
    output_dir
        Directory where translated flows will be written.  If ``None``
        (default), created in the current working directory as
        ``<source_flow_dir_name>_<lang_code>/``.
    translator_api_key, translator_api_base
        API credentials for the translator model.
    verifier_api_key, verifier_api_base
        API credentials for the verifier model.
    max_retries
        Maximum translation attempts per prompt message on verifier failure.
    verbose
        If ``True``, enable ``DEBUG``-level logging.
    register
        If ``True`` (default), register the output directory with
        ``FlowRegistry`` so the translated flows are immediately discoverable.

    Returns
    -------
    Flow
        The translated flow loaded from the generated ``flow.yaml``.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Resolve flow identifier to a filesystem path
    flow_yaml = Path(FlowRegistry.get_flow_path_safe(flow)).resolve()

    # Derive default output_dir from source flow's parent directory name
    if output_dir is None:
        output_path = Path.cwd() / f"{flow_yaml.parent.name}_{lang_code}"
    else:
        output_path = Path(output_dir).resolve()

    # Skip if already translated — check registry and output directory
    translated_id = f"{flow}-{lang_code}"
    if FlowRegistry.get_flow_path(translated_id) is not None:
        logger.info("Flow '%s' already registered, skipping translation", translated_id)
        return Flow.from_yaml(FlowRegistry.get_flow_path_safe(translated_id))
    if output_path.exists() and (output_path / flow_yaml.name).exists():
        logger.info(
            "Output directory '%s' already exists, skipping translation", output_path
        )
        return Flow.from_yaml(str(output_path / flow_yaml.name))

    # Parse flow YAML once — discover prompts and structural tags together
    prompt_yamls, structural_tags = _parse_flow_yaml(flow_yaml)
    tag_rule = _build_tag_rule(structural_tags)

    # Compute output paths — check for basename collisions
    flow_out = output_path / flow_yaml.name
    prompts_dir = output_path / "prompts"
    stems = [src.stem for src in prompt_yamls]
    if len(stems) != len(set(stems)):
        raise ValueError(
            f"Duplicate prompt basenames detected: {stems}. "
            "All prompt files must have unique names."
        )
    prompt_mapping = {
        src: prompts_dir / f"{src.stem}_{lang_code}.yaml" for src in prompt_yamls
    }

    logger.info(
        "Translating flow '%s' to %s (%s) — %d prompt(s)",
        flow,
        lang,
        lang_code,
        len(prompt_mapping),
    )

    # Translate prompt YAMLs
    all_issues: list[str] = []
    for source_path, out_path in prompt_mapping.items():
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
            tag_rule=tag_rule,
        )
        all_issues.extend(issues)

    # Adapt flow YAML — only rewrite paths for prompts we actually translated
    translated_basenames = {src.name for src in prompt_yamls}
    _adapt_flow_yaml(flow_yaml, flow_out, lang, lang_code, translated_basenames)

    # Summary
    if all_issues:
        for issue in all_issues:
            logger.warning("Validation issue: %s", issue)
    logger.info("Translation complete — output: %s", output_path)

    # Register translated flows with FlowRegistry
    if register:
        FlowRegistry.register_search_path(str(output_path))
        FlowRegistry._discover_flows(force_refresh=True)

    return Flow.from_yaml(str(flow_out))
