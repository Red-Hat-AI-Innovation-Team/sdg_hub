# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for applying configuration parameters to flow blocks.

Used by model_config.py and agent_config.py to avoid duplicating the
hasattr / extra=="allow" / warning logic and sensitive-param redaction.
"""

from typing import TYPE_CHECKING, Any
import logging

if TYPE_CHECKING:
    from ..blocks.base import BaseBlock


def apply_config_to_blocks(
    blocks: "list[BaseBlock]",
    target_block_names: set[str],
    config_params: dict[str, Any],
    sensitive_params: set[str],
    config_label: str,
    block_logger: logging.Logger,
) -> int:
    """Apply configuration parameters to targeted blocks.

    Iterates over blocks, matches by name, and sets each config parameter
    using hasattr/setattr with a fallback to Pydantic's extra=="allow".
    Sensitive parameters are redacted in log output.

    Parameters
    ----------
    blocks : list
        The flow's block list (flow.blocks).
    target_block_names : set[str]
        Block names to apply configuration to.
    config_params : dict[str, Any]
        Parameter name-value pairs to set on each target block.
    sensitive_params : set[str]
        Parameter names whose values should be redacted in logs.
    config_label : str
        Human-readable label for log messages (e.g., "LLM", "agent").
    block_logger : logging.Logger
        Logger instance to use for all log output.

    Returns
    -------
    int
        Number of blocks that were modified.
    """
    modified_count = 0
    for block in blocks:
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
                block_logger.warning(
                    f"Block '{block.block_name}' ({block.__class__.__name__}) "
                    f"does not have attribute '{param_name}' - skipping"
                )
                continue

            if param_name in sensitive_params:
                block_logger.debug(
                    f"Block '{block.block_name}': {param_name} set (redacted)"
                )
            else:
                block_logger.debug(
                    f"Block '{block.block_name}': {param_name} set to '{param_value}'"
                )

        if block_modified:
            modified_count += 1

    if modified_count > 0:
        param_summary = []
        for param_name, param_value in config_params.items():
            if param_name in sensitive_params:
                param_summary.append(f"{param_name}: (redacted)")
            else:
                param_summary.append(f"{param_name}: '{param_value}'")

        block_logger.info(
            f"Successfully configured {modified_count} {config_label} "
            f"blocks with: {', '.join(param_summary)}"
        )
        block_logger.info(f"Configured blocks: {sorted(target_block_names)}")
    else:
        block_logger.warning(
            f"No blocks were modified - check block names or "
            f"{config_label} block detection"
        )

    return modified_count
