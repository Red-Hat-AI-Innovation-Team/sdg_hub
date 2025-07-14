# SPDX-License-Identifier: Apache-2.0
"""Redesigned Flow class for managing data generation pipelines."""

# Standard
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third Party
from datasets import Dataset
import yaml

# Local
from ..blocks.base import BaseBlock
from ..blocks.registry import BlockRegistry
from ..logger_config import setup_logger
from ..utils.error_handling import EmptyDatasetError, FlowValidationError
from ..utils.path_resolution import resolve_path
from .metadata import FlowMetadata, FlowParameter, DatasetRequirements
from .validation import FlowValidator

logger = setup_logger(__name__)


class Flow:
    """A metadata-driven flow for chaining data generation blocks.

    Flow provides a clean abstraction for chaining blocks together with support
    for both YAML-based configuration and programmatic creation. Includes
    metadata support for versioning and open source contributions.

    Parameters
    ----------
    blocks : List[BaseBlock], optional
        List of initialized block instances for programmatic creation.
    metadata : FlowMetadata, optional
        Flow metadata. If not provided, minimal metadata will be created.
    parameters : Dict[str, FlowParameter], optional
        Runtime parameters that can be overridden.
    **metadata_kwargs : Any
        Keyword arguments to create FlowMetadata (name, description, etc.).

    Attributes
    ----------
    blocks : List[BaseBlock]
        Ordered list of blocks in the flow.
    metadata : FlowMetadata
        Flow metadata including name, version, etc.
    parameters : Dict[str, FlowParameter]
        Runtime parameters that can be overridden.
    """

    def __init__(
        self,
        blocks: Optional[List[BaseBlock]] = None,
        metadata: Optional[FlowMetadata] = None,
        parameters: Optional[Dict[str, FlowParameter]] = None,
        **metadata_kwargs: Any,
    ) -> None:
        """Initialize Flow with blocks and metadata."""
        self._blocks = blocks or []

        # Create metadata from kwargs if not provided
        if metadata is None:
            # Set default name if not provided
            if "name" not in metadata_kwargs:
                metadata_kwargs["name"] = f"Flow_{len(self._blocks)}_blocks"
            metadata = FlowMetadata(**metadata_kwargs)

        self._metadata = metadata
        self._parameters = parameters or {}

        # Validate blocks
        self._validate_blocks()

        logger.info(
            f"Initialized Flow '{self.metadata.name}' v{self.metadata.version} "
            f"with {len(self.blocks)} blocks"
        )

    @property
    def blocks(self) -> List[BaseBlock]:
        """Get the list of blocks (immutable)."""
        return self._blocks.copy()

    @property
    def metadata(self) -> FlowMetadata:
        """Get flow metadata."""
        return self._metadata

    @property
    def parameters(self) -> Dict[str, FlowParameter]:
        """Get runtime parameters."""
        return self._parameters.copy()

    def _validate_blocks(self) -> None:
        """Validate that all blocks are BaseBlock instances."""
        for i, block in enumerate(self._blocks):
            if not isinstance(block, BaseBlock):
                raise FlowValidationError(
                    f"Block at index {i} is not a BaseBlock instance: {type(block)}"
                )

    @classmethod
    def from_yaml(cls, yaml_path: str, **runtime_kwargs: Any) -> "Flow":
        """Load flow from YAML configuration file.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML flow configuration file.
        **runtime_kwargs : Any
            Runtime arguments to inject into block configurations.

        Returns
        -------
        Flow
            Initialized Flow instance.

        Raises
        ------
        FileNotFoundError
            If the YAML file cannot be found.
        FlowValidationError
            If the YAML structure is invalid.
        KeyError
            If a required block type is not found in the registry.
        """
        yaml_path = resolve_path(yaml_path, [])
        yaml_dir = Path(yaml_path).parent

        logger.info(f"Loading flow from: {yaml_path}")

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                flow_config = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Flow file not found: {yaml_path}") from exc
        except yaml.YAMLError as exc:
            raise FlowValidationError(f"Invalid YAML in {yaml_path}: {exc}") from exc

        # Validate YAML structure
        validator = FlowValidator()
        validation_errors = validator.validate_yaml_structure(flow_config)
        if validation_errors:
            raise FlowValidationError(
                f"Invalid flow configuration:\n" + "\n".join(validation_errors)
            )

        # Extract metadata
        metadata_dict = flow_config.get("metadata", {})
        if "name" not in metadata_dict:
            metadata_dict["name"] = Path(yaml_path).stem
        metadata = FlowMetadata.from_dict(metadata_dict)

        # No need for flow-level parameters anymore

        # Create blocks
        blocks = []
        block_configs = flow_config.get("blocks", [])

        for block_config in block_configs:
            block = cls._create_block_from_config(
                block_config, yaml_dir
            )
            # Store the original config on the block for later serialization
            block._original_config = block_config.copy()
            blocks.append(block)

        return cls(blocks=blocks, metadata=metadata)

    @classmethod
    def _create_block_from_config(
        cls,
        block_config: Dict[str, Any],
        yaml_dir: Path,
    ) -> BaseBlock:
        """Create a block instance from configuration.

        Parameters
        ----------
        block_config : Dict[str, Any]
            Block configuration from YAML.
        yaml_dir : Path
            Directory containing the flow YAML file.

        Returns
        -------
        BaseBlock
            Initialized block instance.
        """
        block_type_name = block_config.get("block_type")
        if not block_type_name:
            raise FlowValidationError("Block configuration missing 'block_type'")

        # Get block class from registry
        try:
            block_class = BlockRegistry.get(block_type_name)
        except KeyError as exc:
            raise FlowValidationError(
                f"Block type '{block_type_name}' not found in registry"
            ) from exc

        # Process block configuration
        config = block_config.get("block_config", {}).copy()

        # Resolve config file paths
        for path_key in ["config_path", "config_paths"]:
            if path_key in config:
                config[path_key] = cls._resolve_config_paths(config[path_key], yaml_dir)

        # No need to apply runtime overrides here - they're handled at generation time

        # Create block instance
        try:
            return block_class(**config)
        except Exception as exc:
            raise FlowValidationError(
                f"Failed to create block '{block_type_name}': {exc}"
            ) from exc

    @classmethod
    def _resolve_config_paths(
        cls, paths: Union[str, List[str], Dict[str, str]], yaml_dir: Path
    ) -> Union[str, List[str], Dict[str, str]]:
        """Resolve configuration file paths relative to YAML directory."""
        if isinstance(paths, str):
            return str(yaml_dir / paths)
        if isinstance(paths, list):
            return [str(yaml_dir / path) for path in paths]
        if isinstance(paths, dict):
            return {key: str(yaml_dir / path) for key, path in paths.items()}
        return paths

    def generate(
        self, 
        dataset: Dataset, 
        runtime_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dataset:
        """Generate data by executing the flow blocks in sequence.

        Parameters
        ----------
        dataset : Dataset
            Input dataset to process.
        runtime_params : Optional[Dict[str, Dict[str, Any]]], optional
            Runtime parameters organized by block name. Format:
            {
                "block_name": {"param1": value1, "param2": value2},
                "other_block": {"param3": value3}
            }

        Returns
        -------
        Dataset
            Processed dataset after all blocks have been executed.

        Raises
        ------
        EmptyDatasetError
            If any block produces an empty dataset.
        FlowValidationError
            If flow validation fails.
        """
        if not self._blocks:
            raise FlowValidationError("Cannot generate with empty flow")

        if len(dataset) == 0:
            raise EmptyDatasetError("Input dataset is empty")

        # Validate dataset requirements if specified
        if self.metadata.dataset_requirements:
            dataset_errors = self.metadata.dataset_requirements.validate_dataset(
                dataset.column_names, len(dataset)
            )
            if dataset_errors:
                raise FlowValidationError(
                    "Dataset validation failed:\n" + "\n".join(dataset_errors)
                )

        logger.info(
            f"Starting flow '{self.metadata.name}' with {len(dataset)} samples "
            f"across {len(self._blocks)} blocks"
        )

        current_dataset = dataset

        for i, block in enumerate(self._blocks):
            logger.info(
                f"Executing block {i + 1}/{len(self._blocks)}: {block.block_name}"
            )

            # Prepare block kwargs with runtime overrides
            block_kwargs = self._prepare_block_kwargs(block, runtime_params or {})

            try:
                # Execute block
                current_dataset = block(current_dataset, **block_kwargs)

                if len(current_dataset) == 0:
                    raise EmptyDatasetError(
                        f"Block '{block.block_name}' produced empty dataset"
                    )

                logger.info(
                    f"Block '{block.block_name}' completed: "
                    f"{len(current_dataset)} samples, "
                    f"{len(current_dataset.column_names)} columns"
                )

            except Exception as exc:
                logger.error(f"Block '{block.block_name}' failed: {exc}")
                raise

        logger.info(
            f"Flow '{self.metadata.name}' completed successfully: "
            f"{len(current_dataset)} final samples"
        )

        return current_dataset

    def _prepare_block_kwargs(
        self, block: BaseBlock, runtime_params: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare kwargs for block execution with runtime overrides."""
        block_name = block.block_name
        
        # Get runtime parameters specific to this block
        if block_name in runtime_params:
            return runtime_params[block_name].copy()
        
        return {}

    def validate(self, dataset: Dataset) -> List[str]:
        """Validate that the flow can be executed with the given dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset to validate against.

        Returns
        -------
        List[str]
            List of validation error messages. Empty if validation passes.
        """
        errors = []

        # Validate dataset requirements if specified
        if self.metadata.dataset_requirements:
            dataset_errors = self.metadata.dataset_requirements.validate_dataset(
                dataset.column_names, len(dataset)
            )
            errors.extend(dataset_errors)

        # Let blocks handle their own validation during execution
        # We only validate high-level dataset compatibility here

        return errors

    def add_block(self, block: BaseBlock) -> "Flow":
        """Add a block to the flow (creates new Flow instance).

        Parameters
        ----------
        block : BaseBlock
            Block to add to the flow.

        Returns
        -------
        Flow
            New Flow instance with the added block.
        """
        new_blocks = self._blocks + [block]
        return Flow(
            blocks=new_blocks, metadata=self._metadata, parameters=self._parameters
        )

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the flow.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing flow information.
        """
        return {
            "metadata": self.metadata.to_dict(),
            "blocks": [
                getattr(block, '_original_config', {
                    "block_type": block.__class__.__name__,
                    "block_config": {"block_name": block.block_name}
                })
                for block in self._blocks
            ],
            "total_blocks": len(self._blocks),
        }

    def to_yaml(self, output_path: str) -> None:
        """Save flow configuration to YAML file.

        Parameters
        ----------
        output_path : str
            Path where to save the YAML file.
        """
        config = {
            "metadata": self.metadata.to_dict(),
            "blocks": [
                getattr(block, '_original_config', {
                    "block_type": block.__class__.__name__,
                    "block_config": {"block_name": block.block_name}
                })
                for block in self._blocks
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Flow configuration saved to: {output_path}")

    def __repr__(self) -> str:
        """String representation of the flow."""
        return (
            f"Flow(name='{self.metadata.name}', "
            f"version='{self.metadata.version}', "
            f"blocks={len(self._blocks)})"
        )

    def __len__(self) -> int:
        """Number of blocks in the flow."""
        return len(self._blocks)
