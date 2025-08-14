# SPDX-License-Identifier: Apache-2.0
"""Migration utilities for backward compatibility with old flow formats."""

# Standard
from pathlib import Path
from typing import Any, Union

# Local
from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)


class FlowMigration:
    """Utility class for migrating old flow formats to new format."""

    @staticmethod
    def is_old_format(flow_config: Union[list[dict[str, Any]], dict[str, Any]]) -> bool:
        """Detect if a flow configuration is in the old format.

        Parameters
        ----------
        flow_config : Union[List[Dict[str, Any]], Dict[str, Any]]
            The loaded YAML configuration.

        Returns
        -------
        bool
            True if the configuration is in old format, False otherwise.
        """
        # Old format: Direct array of blocks
        # New format: Dictionary with 'metadata' and 'blocks' keys
        if isinstance(flow_config, list):
            return True

        if isinstance(flow_config, dict):
            # Check if it has the new format structure
            has_metadata = "metadata" in flow_config
            has_blocks = "blocks" in flow_config

            # If it has both metadata and blocks, it's new format
            if has_metadata and has_blocks:
                return False

            # If it doesn't have the expected new format structure but is a dict,
            # check if it looks like old format (all keys are block configs)
            if not has_metadata and not has_blocks:
                # Check first few items to see if they look like old block configs
                for value in flow_config.values():
                    if isinstance(value, dict) and "block_type" in value:
                        return True
                # If it's a dict but doesn't look like blocks, assume new format
                return False

        # If we can't determine, assume new format
        return False

    @staticmethod
    def migrate_to_new_format(
        flow_config: list[dict[str, Any]], yaml_path: str
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Migrate old format flow configuration to new format.

        Parameters
        ----------
        flow_config : List[Dict[str, Any]]
            Old format flow configuration (array of blocks).
        yaml_path : str
            Path to the original YAML file for generating metadata.

        Returns
        -------
        tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]
            Tuple of (new format flow configuration, extracted runtime_params).
        """
        logger.info(f"Migrating old flow format from: {yaml_path}")

        # Generate default metadata
        flow_name = Path(yaml_path).stem
        metadata = FlowMigration._generate_default_metadata(flow_name)

        # Process blocks and extract runtime parameters
        migrated_blocks = []
        runtime_params = {}

        for i, block_config in enumerate(flow_config):
            try:
                migrated_block, block_runtime_params = (
                    FlowMigration._migrate_block_config(block_config)
                )
                migrated_blocks.append(migrated_block)

                # Add block's runtime params if any
                if block_runtime_params:
                    block_name = migrated_block.get("block_config", {}).get(
                        "block_name"
                    )
                    if block_name:
                        runtime_params[block_name] = block_runtime_params

            except Exception as exc:
                logger.warning(f"Failed to migrate block at index {i}: {exc}")
                # Keep original block config as fallback
                migrated_blocks.append(block_config)

        # Create new format structure
        new_config = {"metadata": metadata, "blocks": migrated_blocks}

        logger.info(f"Successfully migrated flow with {len(migrated_blocks)} blocks")
        logger.info(f"Extracted runtime_params for {len(runtime_params)} blocks")

        return new_config, runtime_params

    @staticmethod
    def _generate_default_metadata(flow_name: str) -> dict[str, Any]:
        """Generate default metadata for migrated flows."""
        # Import here to avoid circular import
        from ..utils.flow_identifier import get_flow_identifier

        metadata = {
            "name": flow_name,
            "description": f"Migrated flow: {flow_name}",
            "version": "1.0.0",
            "author": "SDG_Hub",
            "tags": ["migrated"],
            "recommended_models": {
                "default": "meta-llama/Llama-3.3-70B-Instruct",
                "compatible": [],
                "experimental": [],
            },
        }

        # Generate id for migrated flows
        flow_id = get_flow_identifier(flow_name)
        if flow_id:
            metadata["id"] = flow_id
            logger.debug(f"Generated id for migrated flow: {flow_id}")

        return metadata

    @staticmethod
    def _migrate_block_config(
        block_config: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Migrate individual block configuration from old to new format.

        Parameters
        ----------
        block_config : Dict[str, Any]
            Old format block configuration.

        Returns
        -------
        tuple[Dict[str, Any], Dict[str, Any]]
            Tuple of (migrated block configuration, extracted runtime_params).
        """
        if not isinstance(block_config, dict):
            return block_config, {}

        # Start with the original config
        migrated_config = block_config.copy()
        runtime_params = {}

        # Extract gen_kwargs as runtime_params
        if "gen_kwargs" in migrated_config:
            runtime_params = migrated_config.pop("gen_kwargs")
            logger.debug(f"Extracted gen_kwargs as runtime_params: {runtime_params}")

        # Remove unsupported fields
        for unsupported_field in ["drop_columns", "drop_duplicates", "batch_kwargs"]:
            if unsupported_field in migrated_config:
                migrated_config.pop(unsupported_field)
                logger.debug(
                    f"Ignoring {unsupported_field} as it's not supported in new flow format"
                )

        # Handle parser_kwargs for LLMBlock (keep in block_config)
        if migrated_config.get("block_type") == "LLMBlock":
            block_config_section = migrated_config.get("block_config", {})
            if "parser_kwargs" in block_config_section:
                parser_kwargs = block_config_section["parser_kwargs"]
                logger.debug(f"Preserving parser_kwargs for LLMBlock: {parser_kwargs}")

        # Handle operator string conversion for FilterByValueBlock
        if migrated_config.get("block_type") == "FilterByValueBlock":
            block_config_section = migrated_config.get("block_config", {})
            if "operation" in block_config_section:
                operation = block_config_section["operation"]
                if isinstance(operation, str) and operation.startswith("operator."):
                    # Convert "operator.eq" to "eq"
                    block_config_section["operation"] = operation.replace(
                        "operator.", ""
                    )
                    logger.debug(
                        f"Converted operation from {operation} to {block_config_section['operation']}"
                    )

        return migrated_config, runtime_params
