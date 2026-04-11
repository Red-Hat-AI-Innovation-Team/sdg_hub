# SPDX-License-Identifier: Apache-2.0
"""YAML loading and saving helper functions for Flow class."""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

# Third Party
import yaml

# Local
from ..blocks.base import BaseBlock
from ..blocks.registry import BlockRegistry
from ..utils.error_handling import FlowValidationError
from ..utils.logger_config import setup_logger
from ..utils.path_resolution import resolve_path
from ..utils.yaml_utils import save_flow_yaml
from .metadata import FlowMetadata
from .validation import FlowValidator

if TYPE_CHECKING:
    from .base import Flow

logger = setup_logger(__name__)


def load_flow_from_yaml(flow_cls: type["Flow"], yaml_path: str) -> "Flow":
    """Load flow from YAML configuration file.

    Parameters
    ----------
    flow_cls : type[Flow]
        The Flow class to instantiate.
    yaml_path : str
        Path to the YAML flow configuration file.

    Returns
    -------
    Flow
        Validated Flow instance.

    Raises
    ------
    FlowValidationError
        If ``yaml_path`` is None or the flow configuration/metadata is invalid.
    FileNotFoundError
        If the YAML file does not exist at the given path.
    """
    if yaml_path is None:
        raise FlowValidationError(
            "Flow path cannot be None. Please provide a valid YAML file path or check that the flow exists in the registry."
        )

    yaml_path = resolve_path(yaml_path, [])
    yaml_dir = Path(yaml_path).parent

    logger.info(f"Loading flow from: {yaml_path}")

    flow_config = _parse_yaml_file(yaml_path)
    metadata = _extract_metadata(flow_config, yaml_path)
    blocks = _instantiate_blocks(flow_config, yaml_dir)

    return _assemble_flow(flow_cls, blocks, metadata, flow_config, yaml_path)


def _parse_yaml_file(yaml_path: str) -> dict[str, Any]:
    """Parse and validate the YAML file, returning the flow config dict.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML flow configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed and structurally validated flow configuration.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    FlowValidationError
        If the YAML is malformed or structurally invalid.
    """
    try:
        with open(yaml_path, encoding="utf-8") as f:
            flow_config = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Flow file not found: {yaml_path}") from exc
    except yaml.YAMLError as exc:
        raise FlowValidationError(f"Invalid YAML in {yaml_path}: {exc}") from exc

    # Check that YAML root is a dict (not None, list, or scalar)
    if not isinstance(flow_config, dict):
        raise FlowValidationError(
            f"Invalid flow configuration in {yaml_path}: "
            f"expected a YAML mapping at root, got {type(flow_config).__name__}"
        )

    # Validate YAML structure
    validator = FlowValidator()
    validation_errors = validator.validate_yaml_structure(flow_config)
    if validation_errors:
        raise FlowValidationError(
            "Invalid flow configuration:\n" + "\n".join(validation_errors)
        )

    return flow_config


def _extract_metadata(flow_config: dict[str, Any], yaml_path: str) -> "FlowMetadata":
    """Extract and validate metadata from the flow configuration.

    Parameters
    ----------
    flow_config : dict[str, Any]
        Parsed flow configuration.
    yaml_path : str
        Path to the YAML file (used as fallback name).

    Returns
    -------
    FlowMetadata
        Validated metadata instance.

    Raises
    ------
    FlowValidationError
        If the metadata configuration is invalid.
    """
    metadata_dict = flow_config.get("metadata", {})
    if "name" not in metadata_dict:
        metadata_dict["name"] = Path(yaml_path).stem

    try:
        return FlowMetadata(**metadata_dict)
    except Exception as exc:
        raise FlowValidationError(f"Invalid metadata configuration: {exc}") from exc


def _instantiate_blocks(flow_config: dict[str, Any], yaml_dir: Path) -> list[BaseBlock]:
    """Create block instances from the flow configuration.

    Parameters
    ----------
    flow_config : dict[str, Any]
        Parsed flow configuration.
    yaml_dir : Path
        Directory containing the flow YAML file.

    Returns
    -------
    list[BaseBlock]
        List of validated block instances.

    Raises
    ------
    FlowValidationError
        If any block fails to instantiate.
    """
    blocks = []
    block_configs = flow_config.get("blocks", [])

    for i, block_config in enumerate(block_configs):
        try:
            block = create_block_from_config(block_config, yaml_dir)
            blocks.append(block)
        except Exception as exc:
            raise FlowValidationError(
                f"Failed to create block at index {i}: {exc}"
            ) from exc

    return blocks


def _assemble_flow(
    flow_cls: type["Flow"],
    blocks: list[BaseBlock],
    metadata: "FlowMetadata",
    flow_config: dict[str, Any],
    yaml_path: str,
) -> "Flow":
    """Create the Flow instance and set configuration flags.

    Parameters
    ----------
    flow_cls : type[Flow]
        The Flow class to instantiate.
    blocks : list[BaseBlock]
        Validated block instances.
    metadata : FlowMetadata
        Validated metadata.
    flow_config : dict[str, Any]
        Original parsed flow configuration (used for id persistence).
    yaml_path : str
        Path to the YAML file (used for id persistence).

    Returns
    -------
    Flow
        Fully configured Flow instance.

    Raises
    ------
    FlowValidationError
        If flow validation fails.
    """
    # Import here to avoid circular imports
    from .agent_config import detect_agent_blocks
    from .model_config import detect_llm_blocks

    try:
        flow = flow_cls(blocks=blocks, metadata=metadata)
        # Persist generated id back to the YAML file (only on initial load)
        # If the file had no metadata.id originally, update and rewrite
        if not flow_config.get("metadata", {}).get("id"):
            flow_config.setdefault("metadata", {})["id"] = flow.metadata.id
            save_flow_yaml(
                yaml_path,
                flow_config,
                f"added generated id: {flow.metadata.id}",
            )
        else:
            logger.debug(f"Flow already had id: {flow.metadata.id}")
        # Check if this is a flow without LLM blocks
        llm_blocks = detect_llm_blocks(flow)
        if not llm_blocks:
            # No LLM blocks, so no model config needed
            flow._model_config_set = True
        else:
            # LLM blocks present - user must call set_model_config()
            flow._model_config_set = False

        # Check if this is a flow without agent blocks
        agent_blocks = detect_agent_blocks(flow)
        if not agent_blocks:
            # No agent blocks, so no agent config needed
            flow._agent_config_set = True
        else:
            # Agent blocks present - user must call set_agent_config()
            flow._agent_config_set = False

        return flow
    except FlowValidationError:
        raise
    except Exception as exc:
        raise FlowValidationError(f"Flow validation failed: {exc}") from exc


def create_block_from_config(
    block_config: dict[str, Any],
    yaml_dir: Path,
) -> BaseBlock:
    """Create a block instance from configuration with validation.

    Parameters
    ----------
    block_config : dict[str, Any]
        Block configuration from YAML.
    yaml_dir : Path
        Directory containing the flow YAML file.

    Returns
    -------
    BaseBlock
        Validated block instance.

    Raises
    ------
    FlowValidationError
        If block creation fails.
    """
    # Validate block configuration structure
    if not isinstance(block_config, dict):
        raise FlowValidationError("Block configuration must be a dictionary")

    block_type_name = block_config.get("block_type")
    if not block_type_name:
        raise FlowValidationError("Block configuration missing 'block_type'")

    # Get block class from registry
    try:
        block_class = BlockRegistry._get(block_type_name)
    except KeyError as exc:
        # Get all available blocks from all categories
        all_blocks = BlockRegistry.list_blocks()
        available_blocks = ", ".join(all_blocks)
        raise FlowValidationError(
            f"Block type '{block_type_name}' not found in registry. "
            f"Available blocks: {available_blocks}"
        ) from exc

    # Process block configuration
    config = block_config.get("block_config", {})
    if not isinstance(config, dict):
        raise FlowValidationError("'block_config' must be a dictionary")

    config = config.copy()

    # Resolve config file paths relative to YAML directory
    for path_key in ["config_path", "config_paths", "prompt_config_path"]:
        if path_key in config:
            config[path_key] = resolve_config_paths(config[path_key], yaml_dir)

    # Create block instance with Pydantic validation
    try:
        return block_class(**config)
    except Exception as exc:
        # Don't include full config in error - it may contain secrets
        raise FlowValidationError(
            f"Failed to create block '{block_type_name}' "
            f"with config keys {list(config.keys())}: {exc}"
        ) from exc


def resolve_config_paths(
    paths: Union[str, list[str], dict[str, str]], yaml_dir: Path
) -> Union[str, list[str], dict[str, str]]:
    """Resolve configuration file paths relative to YAML directory.

    Parameters
    ----------
    paths : Union[str, list[str], dict[str, str]]
        Path(s) to resolve.
    yaml_dir : Path
        Directory containing the YAML file.

    Returns
    -------
    Union[str, list[str], dict[str, str]]
        Resolved path(s).
    """
    if isinstance(paths, str):
        return str(yaml_dir / paths)
    elif isinstance(paths, list):
        return [str(yaml_dir / path) for path in paths]
    elif isinstance(paths, dict):
        return {key: str(yaml_dir / path) for key, path in paths.items()}
    return paths


def save_flow_to_yaml(flow: "Flow", output_path: str) -> None:
    """Save flow configuration to YAML file.

    Note: This creates a basic YAML structure. For exact reproduction
    of original YAML, save the original file separately.

    Parameters
    ----------
    flow : Flow
        The flow instance to save.
    output_path : str
        Path to save the YAML file.
    """
    config = {
        "metadata": flow.metadata.model_dump(),
        "blocks": [
            {
                "block_type": block.__class__.__name__,
                "block_config": block.model_dump(),
            }
            for block in flow.blocks
        ],
    }

    save_flow_yaml(output_path, config)
