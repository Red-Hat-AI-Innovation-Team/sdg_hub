# SPDX-License-Identifier: Apache-2.0
"""Model configuration helper functions for Flow class."""

# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from pydantic import SecretStr

# Local
from ..utils.logger_config import setup_logger

if TYPE_CHECKING:
    from .base import Flow

logger = setup_logger(__name__)


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
        API base URL to configure (e.g., "http://localhost:8101/v1").
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
    >>> flow.set_model_config(  # Configure model settings
    ...     model="hosted_vllm/openai/gpt-oss-120b",
    ...     api_base="http://localhost:8101/v1",
    ...     api_key="your_key",
    ...     temperature=0.7,
    ...     max_tokens=2048
    ... )
    >>> result = flow.generate(dataset)  # Generate data

    >>> # Configure only specific blocks
    >>> flow.set_model_config(
    ...     model="hosted_vllm/openai/gpt-oss-120b",
    ...     api_base="http://localhost:8101/v1",
    ...     blocks=["gen_detailed_summary", "knowledge_generation"]
    ... )

    Raises
    ------
    ValueError
        If no configuration parameters are provided or if specified blocks don't exist.
    """
    # Build the configuration parameters dictionary
    config_params = {}
    if model is not None:
        config_params["model"] = model
    if api_base is not None:
        config_params["api_base"] = api_base
    if api_key is not None:
        # Convert string api_key to SecretStr for automatic redaction in logs
        config_params["api_key"] = (
            SecretStr(api_key) if isinstance(api_key, str) else api_key
        )

    # Add any additional kwargs (temperature, max_tokens, etc.)
    config_params.update(kwargs)

    # Validate that at least one parameter is provided
    if not config_params:
        raise ValueError(
            "At least one configuration parameter must be provided "
            "(model, api_base, api_key, or **kwargs)"
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
            f"Targeting specific blocks for configuration: {sorted(target_block_names)}"
        )
    else:
        # Auto-detect LLM blocks
        target_block_names = set(detect_llm_blocks(flow))
        logger.info(
            f"Auto-detected {len(target_block_names)} LLM blocks for configuration: {sorted(target_block_names)}"
        )

    # Sensitive parameter names that should not be logged
    sensitive_params = {"api_key", "token", "password", "secret"}

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
        # Apply same redaction rules as per-block logging
        param_summary = []
        for param_name, param_value in config_params.items():
            if param_name in sensitive_params:
                param_summary.append(f"{param_name}: (redacted)")
            else:
                param_summary.append(f"{param_name}: '{param_value}'")

        logger.info(
            f"Successfully configured {modified_count} LLM blocks with: {', '.join(param_summary)}"
        )
        logger.info(f"Configured blocks: {sorted(target_block_names)}")

        # Mark that model configuration has been set
        flow._model_config_set = True
    else:
        logger.warning(
            "No blocks were modified - check block names or LLM block detection"
        )
