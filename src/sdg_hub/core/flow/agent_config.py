# SPDX-License-Identifier: Apache-2.0
"""Agent configuration helper functions for Flow class."""

# Standard
from typing import TYPE_CHECKING, Any, Optional

# Local
from ..utils.config_helpers import apply_config_to_blocks, resolve_target_blocks
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

    # Resolve which blocks to target
    target_block_names = resolve_target_blocks(
        flow, blocks, detect_agent_blocks, config_label="agent"
    )

    # Apply configuration and mark the flow if any blocks were modified
    modified_count = apply_config_to_blocks(
        flow, config_params, target_block_names, config_label="agent"
    )

    if modified_count > 0:
        flow._agent_config_set = True
