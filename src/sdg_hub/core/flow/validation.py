# SPDX-License-Identifier: Apache-2.0
"""Flow validation utilities."""

# Standard
from typing import TYPE_CHECKING, Any

# Third Party
import pandas as pd

if TYPE_CHECKING:
    # Local
    from .base import Flow


class FlowValidator:
    """Validator for flow configurations and execution readiness."""

    def validate_yaml_structure(self, flow_config: dict[str, Any]) -> list[str]:
        """Validate the structure of a flow YAML configuration.

        Parameters
        ----------
        flow_config : Dict[str, Any]
            The loaded YAML configuration.

        Returns
        -------
        List[str]
            List of validation error messages. Empty if valid.
        """
        errors = []

        # Check required top-level keys
        if "blocks" not in flow_config:
            errors.append("Flow configuration must contain 'blocks' section")
            return errors  # Can't continue without blocks

        blocks = flow_config["blocks"]
        if not isinstance(blocks, list):
            errors.append("'blocks' must be a list")
            return errors

        if not blocks:
            errors.append("Flow must contain at least one block")
            return errors

        # Validate each block configuration
        for i, block_config in enumerate(blocks):
            block_errors = self._validate_block_config(block_config, i)
            errors.extend(block_errors)

        # Validate metadata if present
        if "metadata" in flow_config:
            metadata_errors = self._validate_metadata_config(flow_config["metadata"])
            errors.extend(metadata_errors)

            # Validate output_columns against block outputs if declared
            metadata = flow_config["metadata"]
            if isinstance(metadata, dict) and "output_columns" in metadata:
                output_cols = metadata["output_columns"]
                if isinstance(output_cols, list) and all(
                    isinstance(c, str) for c in output_cols
                ):
                    oc_errors = self._validate_output_columns_against_blocks(
                        output_cols,
                        flow_config.get("blocks", []),
                        metadata=metadata,
                    )
                    errors.extend(oc_errors)

        # Validate parameters if present
        if "parameters" in flow_config:
            param_errors = self._validate_parameters_config(flow_config["parameters"])
            errors.extend(param_errors)

        return errors

    def _validate_block_config(
        self, block_config: dict[str, Any], index: int
    ) -> list[str]:
        """Validate a single block configuration."""
        errors = []
        prefix = f"Block {index}"

        if not isinstance(block_config, dict):
            errors.append(f"{prefix}: Block configuration must be a dictionary")
            return errors

        # Check required fields
        if "block_type" not in block_config:
            errors.append(f"{prefix}: Missing required field 'block_type'")

        if "block_config" not in block_config:
            errors.append(f"{prefix}: Missing required field 'block_config'")
        else:
            # Validate block_config structure
            inner_config = block_config["block_config"]
            if not isinstance(inner_config, dict):
                errors.append(f"{prefix}: 'block_config' must be a dictionary")
            elif "block_name" not in inner_config:
                errors.append(f"{prefix}: 'block_config' must contain 'block_name'")

        # Validate optional fields
        if "runtime_overrides" in block_config:
            overrides = block_config["runtime_overrides"]
            if not isinstance(overrides, list):
                errors.append(f"{prefix}: 'runtime_overrides' must be a list")
            elif not all(isinstance(item, str) for item in overrides):
                errors.append(
                    f"{prefix}: All 'runtime_overrides' items must be strings"
                )

        return errors

    def _validate_metadata_config(self, metadata: dict[str, Any]) -> list[str]:
        """Validate metadata configuration."""
        errors = []

        if not isinstance(metadata, dict):
            errors.append("'metadata' must be a dictionary")
            return errors

        # Check required name field
        if "name" not in metadata:
            errors.append("Metadata must contain 'name' field")
        elif not isinstance(metadata["name"], str) or not metadata["name"].strip():
            errors.append("Metadata 'name' must be a non-empty string")

        # Validate id if present
        if "id" in metadata:
            flow_id = metadata["id"]
            if not isinstance(flow_id, str):
                errors.append("Metadata: 'id' must be a string")
            elif flow_id and not flow_id.islower():
                errors.append("Metadata: 'id' must be lowercase")
            elif flow_id and not flow_id.replace("-", "").isalnum():
                errors.append(
                    "Metadata: 'id' must contain only alphanumeric characters and hyphens"
                )

        # Validate optional fields
        string_fields = [
            "description",
            "version",
            "author",
            "recommended_model",
            "license",
        ]
        for field in string_fields:
            if field in metadata and not isinstance(metadata[field], str):
                errors.append(f"Metadata '{field}' must be a string")

        if "tags" in metadata:
            tags = metadata["tags"]
            if not isinstance(tags, list):
                errors.append("Metadata 'tags' must be a list")
            elif not all(isinstance(tag, str) for tag in tags):
                errors.append("All metadata 'tags' must be strings")

        if "output_columns" in metadata:
            output_cols = metadata["output_columns"]
            if output_cols is not None:
                if not isinstance(output_cols, list):
                    errors.append("Metadata 'output_columns' must be a list")
                elif not all(isinstance(col, str) for col in output_cols):
                    errors.append("All metadata 'output_columns' must be strings")

        return errors

    def _validate_parameters_config(self, parameters: dict[str, Any]) -> list[str]:
        """Validate parameters configuration."""
        errors = []

        if not isinstance(parameters, dict):
            errors.append("'parameters' must be a dictionary")
            return errors

        for param_name, param_config in parameters.items():
            if not isinstance(param_name, str):
                errors.append("Parameter names must be strings")
                continue

            if isinstance(param_config, dict):
                # Full parameter specification
                if "default" not in param_config:
                    errors.append(f"Parameter '{param_name}' must have 'default' value")

                # Validate optional fields
                if "description" in param_config and not isinstance(
                    param_config["description"], str
                ):
                    errors.append(
                        f"Parameter '{param_name}' description must be a string"
                    )

                if "required" in param_config and not isinstance(
                    param_config["required"], bool
                ):
                    errors.append(
                        f"Parameter '{param_name}' required field must be boolean"
                    )

        return errors

    def validate_flow_execution(self, flow: "Flow", dataset: pd.DataFrame) -> list[str]:
        """Validate that a flow can be executed with the given dataset.

        Parameters
        ----------
        flow : Flow
            The flow to validate.
        dataset : pd.DataFrame
            Dataset to validate against.

        Returns
        -------
        List[str]
            List of validation error messages. Empty if validation passes.
        """
        errors = []

        if not flow.blocks:
            errors.append("Flow contains no blocks")
            return errors

        if len(dataset) == 0:
            errors.append("Dataset is empty")
            return errors

        # Track available columns as we progress through blocks
        current_columns = set(dataset.columns.tolist())

        for _i, block in enumerate(flow.blocks):
            block_name = block.block_name

            # Check input columns
            if hasattr(block, "input_cols") and block.input_cols:
                missing_cols = self._check_missing_columns(
                    block.input_cols, current_columns
                )
                if missing_cols:
                    errors.append(
                        f"Block '{block_name}' missing input columns: {missing_cols}"
                    )

            # Update available columns for next block
            if hasattr(block, "output_cols") and block.output_cols:
                new_columns = self._extract_column_names(block.output_cols)
                current_columns.update(new_columns)

        return errors

    def _check_missing_columns(
        self, required_cols: Any, available_cols: set[str]
    ) -> list[str]:
        """Check which required columns are missing."""
        if isinstance(required_cols, (list, dict)):
            return [col for col in required_cols if col not in available_cols]
        return []

    def _extract_column_names(self, output_cols: Any) -> list[str]:
        """Extract column names from output specification."""
        if isinstance(output_cols, str):
            return [output_cols]
        if isinstance(output_cols, list):
            return output_cols
        if isinstance(output_cols, dict):
            return list(output_cols.keys())
        return []

    def _validate_output_columns_against_blocks(
        self,
        output_columns: list[str],
        block_configs: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Validate output_columns against block configs from raw YAML.

        Traces column availability through the block chain to verify that all
        declared output_columns are producible. Seeds the available column set
        from dataset_requirements to avoid false positives on input columns
        that pass through untouched.

        Parameters
        ----------
        output_columns : list[str]
            Declared output columns from metadata.
        block_configs : list[dict[str, Any]]
            Raw block configurations from YAML.
        metadata : dict[str, Any] | None
            Full metadata dict, used to read dataset_requirements.

        Returns
        -------
        list[str]
            Validation error messages.
        """
        if not output_columns:
            return []

        # Seed with known input columns from dataset_requirements so that
        # passthrough columns are not flagged as missing.
        available_columns: set[str] = set()
        if metadata:
            dataset_req = metadata.get("dataset_requirements", {})
            if isinstance(dataset_req, dict):
                for key in ("required_columns", "optional_columns"):
                    cols = dataset_req.get(key, [])
                    if isinstance(cols, list):
                        available_columns.update(c for c in cols if isinstance(c, str))

        for block_config in block_configs:
            if not isinstance(block_config, dict):
                continue

            block_type = block_config.get("block_type", "")
            config = block_config.get("block_config", {})
            if not isinstance(config, dict):
                continue

            output_cols = config.get("output_cols")

            if block_type == "RenameColumnsBlock":
                input_cols = config.get("input_cols")
                if isinstance(input_cols, dict):
                    for old_name, new_name in input_cols.items():
                        available_columns.discard(old_name)
                        if isinstance(new_name, str):
                            available_columns.add(new_name)
            else:
                if output_cols:
                    available_columns.update(self._extract_column_names(output_cols))

        missing = [col for col in output_columns if col not in available_columns]
        if missing:
            return [
                f"Declared output_columns {missing} cannot be traced to any "
                f"block output or dataset requirement. "
                f"Available columns: {sorted(available_columns)}"
            ]

        return []

    def validate_block_chain(self, blocks: list[Any]) -> list[str]:
        """Validate that blocks can be chained together.

        Parameters
        ----------
        blocks : List[Any]
            List of block instances to validate.

        Returns
        -------
        List[str]
            List of validation error messages.
        """
        errors = []

        if not blocks:
            errors.append("Block chain is empty")
            return errors

        # Check that all blocks have unique names
        block_names = []
        for i, block in enumerate(blocks):
            if hasattr(block, "block_name"):
                name = block.block_name
                if name in block_names:
                    errors.append(f"Duplicate block name '{name}' at index {i}")
                block_names.append(name)
            else:
                errors.append(f"Block at index {i} missing 'block_name' attribute")

        return errors
