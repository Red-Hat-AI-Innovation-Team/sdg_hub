# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for applying configuration parameters to flow blocks.

Both ``model_config.py`` and ``agent_config.py`` need identical logic to
iterate over target blocks, set attributes (via ``hasattr`` or the Pydantic
``extra == "allow"`` fallback), redact sensitive values in logs and build a
summary.  This module extracts that loop so it only exists once.
"""

# Standard
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

# Local
from .logger_config import setup_logger

if TYPE_CHECKING:
    from ..flow.base import Flow

logger = setup_logger(__name__)

# Parameter names whose values must never appear in log output.
# Intentionally the superset of what model_config and agent_config each
# previously defined so that both code-paths share a single set.
DEFAULT_SENSITIVE_PARAMS: frozenset[str] = frozenset(
    {"api_key", "agent_api_key", "token", "password", "secret"}
)


def resolve_target_blocks(
    flow: "Flow",
    blocks: list[str] | None,
    auto_detect_fn: Callable[[Flow], list[str]],
    config_label: str,
) -> set[str]:
    """Validate explicit block names or auto-detect target blocks.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    blocks : list[str] | None
        Explicit block names supplied by the caller, or ``None`` to
        auto-detect.
    auto_detect_fn : Callable[[Flow], list[str]]
        A function that returns the list of auto-detected block names
        (e.g. ``detect_llm_blocks`` or ``detect_agent_blocks``).
    config_label : str
        Human-readable label used in log messages (e.g. ``"LLM"`` or
        ``"agent"``).

    Returns
    -------
    set[str]
        The resolved set of target block names.

    Raises
    ------
    ValueError
        If any explicitly named block does not exist in the flow.
    """
    if blocks is not None:
        existing_block_names = {block.block_name for block in flow.blocks}
        invalid_blocks = set(blocks) - existing_block_names
        if invalid_blocks:
            raise ValueError(
                f"Specified blocks not found in flow: {sorted(invalid_blocks)}. "
                f"Available blocks: {sorted(existing_block_names)}"
            )
        target_block_names = set(blocks)
        logger.info(
            f"Targeting specific blocks for {config_label} configuration: "
            f"{sorted(target_block_names)}"
        )
    else:
        target_block_names = set(auto_detect_fn(flow))
        logger.info(
            f"Auto-detected {len(target_block_names)} {config_label} blocks "
            f"for configuration: {sorted(target_block_names)}"
        )

    return target_block_names


def apply_config_to_blocks(
    flow: "Flow",
    config_params: dict[str, Any],
    target_block_names: set[str],
    config_label: str,
    sensitive_params: frozenset[str] = DEFAULT_SENSITIVE_PARAMS,
) -> int:
    """Apply configuration parameters to the target blocks of a flow.

    For each target block, every key in *config_params* is applied using the
    following precedence:

    1. If the block already has the attribute (``hasattr``), set it directly.
    2. Else if the block's Pydantic ``model_config`` allows extras, set it.
    3. Otherwise log a warning and skip the parameter.

    Sensitive values (as defined by *sensitive_params*) are redacted in all log
    output.

    Parameters
    ----------
    flow : Flow
        The flow instance whose blocks will be mutated.
    config_params : dict[str, Any]
        Mapping of parameter names to values to apply.
    target_block_names : set[str]
        Block names that should receive the configuration.
    config_label : str
        Human-readable label for log messages (e.g. ``"LLM"`` or ``"agent"``).
    sensitive_params : frozenset[str]
        Parameter names whose values must be redacted in logs.

    Returns
    -------
    int
        The number of blocks that were actually modified.
    """
    modified_count = 0

    for block in flow.blocks:
        if block.block_name not in target_block_names:
            continue

        block_modified = False
        for param_name, param_value in config_params.items():
            if hasattr(block, param_name):
                setattr(block, param_name, param_value)
                block_modified = True
            elif block.model_config.get("extra") == "allow":
                setattr(block, param_name, param_value)
                block_modified = True
            else:
                logger.warning(
                    f"Block '{block.block_name}' ({block.__class__.__name__}) "
                    f"does not have attribute '{param_name}' - skipping"
                )
                continue

            # Log what was set, redacting sensitive values.
            if param_name in sensitive_params:
                logger.debug(f"Block '{block.block_name}': {param_name} set (redacted)")
            else:
                logger.debug(
                    f"Block '{block.block_name}': {param_name} set to '{param_value}'"
                )

        if block_modified:
            modified_count += 1

    # Summary logging
    if modified_count > 0:
        param_summary = []
        for param_name, param_value in config_params.items():
            if param_name in sensitive_params:
                param_summary.append(f"{param_name}: (redacted)")
            else:
                param_summary.append(f"{param_name}: '{param_value}'")

        logger.info(
            f"Successfully configured {modified_count} {config_label} blocks "
            f"with: {', '.join(param_summary)}"
        )
        logger.info(f"Configured blocks: {sorted(target_block_names)}")
    else:
        logger.warning(
            f"No blocks were modified - check block names or {config_label} "
            f"block detection"
        )

    return modified_count
