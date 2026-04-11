# SPDX-License-Identifier: Apache-2.0
"""Agent configuration helper functions for Flow class."""

# Standard
from typing import TYPE_CHECKING, Any, Optional

# Local
from ..utils.logger_config import setup_logger

if TYPE_CHECKING:
    from .base import Flow

logger = setup_logger(__name__)

# Default Langflow agent endpoint URL for local development.
DEFAULT_AGENT_URL = "http://localhost:7860/api/v1/run/my-flow"


def detect_agent_blocks(flow: "Flow") -> list[str]:
    """Detect blocks with block_type='agent'.

    Parameters
    ----------
    flow : Flow
        The flow instance.

    Returns
    -------
    list[str]
        List of block names that are agent blocks.
    """
    return [block.block_name for block in flow.blocks if block.block_type == "agent"]


def is_agent_config_required(flow: "Flow") -> bool:
    """Check if agent configuration is required for this flow.

    Parameters
    ----------
    flow : Flow
        The flow instance.

    Returns
    -------
    bool
        True if flow has agent blocks and needs agent configuration.
    """
    return len(detect_agent_blocks(flow)) > 0


def is_agent_config_set(flow: "Flow") -> bool:
    """Check if agent configuration has been set.

    Parameters
    ----------
    flow : Flow
        The flow instance.

    Returns
    -------
    bool
        True if agent configuration has been set or is not required.
    """
    return (not is_agent_config_required(flow)) or flow._agent_config_set


def reset_agent_config(flow: "Flow") -> None:
    """Reset agent configuration flag (useful for testing or reconfiguration).

    After calling this, set_agent_config() must be called again before generate().

    Parameters
    ----------
    flow : Flow
        The flow instance to reset.
    """
    if is_agent_config_required(flow):
        flow._agent_config_set = False
        logger.info(
            "Agent configuration flag reset - call set_agent_config() before generate()"
        )


def set_agent_config(
    flow: "Flow",
    agent_framework: Optional[str] = None,
    agent_url: Optional[str] = None,
    agent_api_key: Optional[str] = None,
    blocks: Optional[list[str]] = None,
    **kwargs: Any,
) -> None:
    """Configure agent settings for agent blocks in this flow (in-place).

    This function is designed to work with credential-free flow definitions where
    agent blocks don't have hardcoded URLs or API keys in the YAML. Instead,
    agent settings are configured at runtime using this function.

    By default, auto-detects all agent blocks in the flow and applies configuration
    to them. Optionally allows targeting specific blocks only.

    Parameters
    ----------
    flow : Flow
        The flow instance to configure.
    agent_framework : Optional[str]
        Agent framework connector name (e.g., 'langflow').
    agent_url : Optional[str]
        Agent API endpoint URL to configure.
    agent_api_key : Optional[str]
        Agent API key to configure.
    blocks : Optional[list[str]]
        Specific block names to target. If None, auto-detects all agent blocks.
    **kwargs : Any
        Additional agent parameters (e.g., timeout, max_retries).

    Examples
    --------
    >>> flow = Flow.from_yaml("path/to/flow.yaml")
    >>> from sdg_hub.core.flow.agent_config import DEFAULT_AGENT_URL
    >>> flow.set_agent_config(
    ...     agent_framework="langflow",
    ...     agent_url=DEFAULT_AGENT_URL,
    ...     agent_api_key="your_key",
    ... )
    >>> result = flow.generate(dataset)

    >>> # Configure only specific blocks
    >>> flow.set_agent_config(
    ...     agent_url=DEFAULT_AGENT_URL,
    ...     blocks=["my_agent_block"]
    ... )

    Raises
    ------
    ValueError
        If no configuration parameters are provided or if specified blocks don't exist.
    """
    # Build the configuration parameters dictionary
    config_params: dict[str, Any] = {}
    if agent_framework is not None:
        config_params["agent_framework"] = agent_framework
    if agent_url is not None:
        config_params["agent_url"] = agent_url
    if agent_api_key is not None:
        config_params["agent_api_key"] = agent_api_key

    # Add any additional kwargs (timeout, max_retries, etc.)
    config_params.update(kwargs)

    # Validate that at least one parameter is provided
    if not config_params:
        raise ValueError(
            "At least one configuration parameter must be provided "
            "(agent_framework, agent_url, agent_api_key, or **kwargs)"
        )

    # Determine target blocks
    if blocks is not None:
        # Validate that specified blocks exist in the flow
        existing_block_names = {block.block_name for block in flow.blocks}
        invalid_blocks = set(blocks) - existing_block_names
        if invalid_blocks:
            raise ValueError(
                f"Specified blocks not found in flow: {sorted(invalid_blocks)}. "
                f"Available blocks: {sorted(existing_block_names)}"
            )
        target_block_names = set(blocks)
        logger.info(
            f"Targeting specific blocks for agent configuration: {sorted(target_block_names)}"
        )
    else:
        # Auto-detect agent blocks
        target_block_names = set(detect_agent_blocks(flow))
        logger.info(
            f"Auto-detected {len(target_block_names)} agent blocks for configuration: "
            f"{sorted(target_block_names)}"
        )

    # Sensitive parameter names that should not be logged
    sensitive_params = {"agent_api_key", "api_key", "token", "password", "secret"}

    # Apply configuration to target blocks
    modified_count = 0
    for block in flow.blocks:
        if block.block_name in target_block_names:
            block_modified = False
            for param_name, param_value in config_params.items():
                if hasattr(block, param_name):
                    setattr(block, param_name, param_value)
                    block_modified = True
                    # Don't log sensitive values
                    if param_name in sensitive_params:
                        logger.debug(
                            f"Block '{block.block_name}': {param_name} set (redacted)"
                        )
                    else:
                        logger.debug(
                            f"Block '{block.block_name}': {param_name} "
                            f"set to '{param_value}'"
                        )
                # check if allow extra
                elif block.model_config.get("extra") == "allow":
                    setattr(block, param_name, param_value)
                    block_modified = True
                    if param_name in sensitive_params:
                        logger.debug(
                            f"Block '{block.block_name}': {param_name} set (redacted)"
                        )
                    else:
                        logger.debug(
                            f"Block '{block.block_name}': {param_name} "
                            f"set to '{param_value}'"
                        )
                else:
                    logger.warning(
                        f"Block '{block.block_name}' ({block.__class__.__name__}) "
                        f"does not have attribute '{param_name}' - skipping"
                    )

            if block_modified:
                modified_count += 1

    if modified_count > 0:
        # Enhanced logging showing what was configured
        param_summary = []
        for param_name, param_value in config_params.items():
            if param_name in sensitive_params:
                param_summary.append(f"{param_name}: (redacted)")
            else:
                param_summary.append(f"{param_name}: '{param_value}'")

        logger.info(
            f"Successfully configured {modified_count} agent blocks with: "
            f"{', '.join(param_summary)}"
        )
        logger.info(f"Configured blocks: {sorted(target_block_names)}")

        # Mark that agent configuration has been set
        flow._agent_config_set = True
    else:
        logger.warning(
            "No blocks were modified - check block names or agent block detection"
        )
