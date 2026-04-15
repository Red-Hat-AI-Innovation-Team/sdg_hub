# SPDX-License-Identifier: Apache-2.0
"""Model configuration helper functions for Flow class."""

# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from pydantic import SecretStr

# Local
from ..utils.config_helpers import apply_config_to_blocks, resolve_target_blocks
from ..utils.logger_config import setup_logger

if TYPE_CHECKING:
    from .base import Flow

logger = setup_logger(__name__)

# Default API base URL for local vLLM / OpenAI-compatible model servers.
DEFAULT_API_BASE = "http://localhost:8101/v1"


def detect_llm_blocks(flow: "Flow") -> list[str]:
    """Detect blocks with block_type='llm'.

    Parameters
    ----------
    flow : Flow
        The flow instance.

    Returns
    -------
    list[str]
        List of block names that are LLM blocks.
    """
    return [block.block_name for block in flow.blocks if block.block_type == "llm"]


def is_model_config_required(flow: "Flow") -> bool:
    """Check if model configuration is required for this flow.

    Parameters
    ----------
    flow : Flow
        The flow instance.

    Returns
    -------
    bool
        True if flow has LLM blocks and needs model configuration.
    """
    return len(detect_llm_blocks(flow)) > 0


def is_model_config_set(flow: "Flow") -> bool:
    """Check if model configuration has been set.

    Parameters
    ----------
    flow : Flow
        The flow instance.

    Returns
    -------
    bool
        True if model configuration has been set or is not required.
    """
    return (not is_model_config_required(flow)) or flow._model_config_set


def reset_model_config(flow: "Flow") -> None:
    """Reset model configuration flag (useful for testing or reconfiguration).

    After calling this, set_model_config() must be called again before generate().

    Parameters
    ----------
    flow : Flow
        The flow instance to reset.
    """
    if is_model_config_required(flow):
        flow._model_config_set = False
        logger.info(
            "Model configuration flag reset - call set_model_config() before generate()"
        )


def get_default_model(flow: "Flow") -> Optional[str]:
    """Get the default recommended model for this flow.

    Parameters
    ----------
    flow : Flow
        The flow instance.

    Returns
    -------
    Optional[str]
        Default model name, or None if no models specified.

    Examples
    --------
    >>> flow = Flow.from_yaml("path/to/flow.yaml")
    >>> default_model = flow.get_default_model()
    >>> print(f"Default model: {default_model}")
    """
    if not flow.metadata.recommended_models:
        return None
    return flow.metadata.recommended_models.default


def get_model_recommendations(flow: "Flow") -> dict[str, Any]:
    """Get a clean summary of model recommendations for this flow.

    Parameters
    ----------
    flow : Flow
        The flow instance.

    Returns
    -------
    dict[str, Any]
        Dictionary with model recommendations in user-friendly format.

    Examples
    --------
    >>> flow = Flow.from_yaml("path/to/flow.yaml")
    >>> recommendations = flow.get_model_recommendations()
    >>> print("Model recommendations:")
    >>> print(f"  Default: {recommendations['default']}")
    >>> print(f"  Compatible: {recommendations['compatible']}")
    >>> print(f"  Experimental: {recommendations['experimental']}")
    """
    if not flow.metadata.recommended_models:
        return {
            "default": None,
            "compatible": [],
            "experimental": [],
        }

    return {
        "default": flow.metadata.recommended_models.default,
        "compatible": flow.metadata.recommended_models.compatible,
        "experimental": flow.metadata.recommended_models.experimental,
    }


def set_model_config(
    flow: "Flow",
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    blocks: Optional[list[str]] = None,
    **kwargs: Any,
) -> None:
    """Configure model settings for LLM blocks in this flow (in-place).

    This function is designed to work with model-agnostic flow definitions where
    LLM blocks don't have hardcoded model configurations in the YAML. Instead,
    model settings are configured at runtime using this function.

    Based on LiteLLM's basic usage pattern, this function focuses on the core
    parameters (model, api_base, api_key) with additional parameters passed via kwargs.

    By default, auto-detects all LLM blocks in the flow and applies configuration to them.
    Optionally allows targeting specific blocks only.

    Parameters
    ----------
    flow : Flow
        The flow instance to configure.
    model : Optional[str]
        Model name to configure (e.g., "hosted_vllm/openai/gpt-oss-120b").
    api_base : Optional[str]
        API base URL to configure (e.g., DEFAULT_API_BASE).
    api_key : Optional[str]
        API key to configure.
    blocks : Optional[list[str]]
        Specific block names to target. If None, auto-detects all LLM blocks.
    **kwargs : Any
        Additional model parameters (e.g., temperature, max_tokens, top_p, etc.).

    Examples
    --------
    >>> # Recommended workflow: discover -> initialize -> set_model_config -> generate
    >>> flow = Flow.from_yaml("path/to/flow.yaml")  # Initialize flow
    >>> from sdg_hub.core.flow.model_config import DEFAULT_API_BASE
    >>> flow.set_model_config(  # Configure model settings
    ...     model="hosted_vllm/openai/gpt-oss-120b",
    ...     api_base=DEFAULT_API_BASE,
    ...     api_key="your_key",
    ...     temperature=0.7,
    ...     max_tokens=2048
    ... )
    >>> result = flow.generate(dataset)  # Generate data

    >>> # Configure only specific blocks
    >>> flow.set_model_config(
    ...     model="hosted_vllm/openai/gpt-oss-120b",
    ...     api_base=DEFAULT_API_BASE,
    ...     blocks=["gen_detailed_summary", "knowledge_generation"]
    ... )

    Raises
    ------
    ValueError
        If no configuration parameters are provided or if specified blocks don't exist.
    """
    # Build the configuration parameters dictionary
    config_params: dict[str, Any] = {}
    if model is not None:
        config_params["model"] = model
    if api_base is not None:
        config_params["api_base"] = api_base
    if api_key is not None:
        # Convert string api_key to SecretStr for automatic redaction in logs
        api_key_secret = SecretStr(api_key)
        config_params["api_key"] = api_key_secret

    # Add any additional kwargs (temperature, max_tokens, etc.)
    config_params.update(kwargs)

    # Validate that at least one parameter is provided
    if not config_params:
        raise ValueError(
            "At least one configuration parameter must be provided "
            "(model, api_base, api_key, or **kwargs)"
        )

    # Resolve which blocks to target
    target_block_names = resolve_target_blocks(
        flow, blocks, detect_llm_blocks, config_label="LLM"
    )

    # Apply configuration and mark the flow if any blocks were modified
    modified_count = apply_config_to_blocks(
        flow, config_params, target_block_names, config_label="LLM"
    )

    if modified_count > 0:
        flow._model_config_set = True
