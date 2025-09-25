# SPDX-License-Identifier: Apache-2.0
"""Pydantic-based Flow class for managing data generation pipelines."""

# Standard
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
import time
import uuid

# Third Party
from datasets import Dataset
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
import datasets
import yaml

# Local
from ..blocks.base import BaseBlock
from ..blocks.registry import BlockRegistry
from ..utils.datautils import safe_concatenate_with_validation, validate_no_duplicates
from ..utils.error_handling import EmptyDatasetError, FlowValidationError
from ..utils.flow_metrics import display_metrics_summary, save_metrics_to_json
from ..utils.logger_config import setup_logger
from ..utils.path_resolution import resolve_path
from ..utils.yaml_utils import save_flow_yaml
from .checkpointer import FlowCheckpointer
from .metadata import DatasetRequirements, FlowMetadata, FlowParameter
from .migration import FlowMigration
from .validation import FlowValidator

logger = setup_logger(__name__)


class Flow(BaseModel):
    """Pydantic-based flow for chaining data generation blocks.

    A Flow represents a complete data generation pipeline with proper validation,
    metadata tracking, and execution capabilities. All configuration is validated
    using Pydantic models for type safety and better error messages.

    Attributes
    ----------
    blocks : List[BaseBlock]
        Ordered list of blocks to execute in the flow.
    metadata : FlowMetadata
        Flow metadata including name, version, author, etc.
    parameters : Dict[str, FlowParameter]
        Runtime parameters that can be overridden during execution.
    """

    blocks: list[BaseBlock] = Field(
        default_factory=list,
        description="Ordered list of blocks to execute in the flow",
    )
    metadata: FlowMetadata = Field(
        description="Flow metadata including name, version, author, etc."
    )
    parameters: dict[str, FlowParameter] = Field(
        default_factory=dict,
        description="Runtime parameters that can be overridden during execution",
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Private attributes (not serialized)
    _migrated_runtime_params: dict[str, dict[str, Any]] = {}
    _llm_client: Any = None  # Only used for backward compatibility with old YAMLs
    _model_config_set: bool = False  # Track if model configuration has been set
    _block_metrics: list[dict[str, Any]] = PrivateAttr(
        default_factory=list
    )  # Track block execution metrics

    @field_validator("blocks")
    @classmethod
    def validate_blocks(cls, v: list[BaseBlock]) -> list[BaseBlock]:
        """Validate that all blocks are BaseBlock instances."""
        if not v:
            return v

        for i, block in enumerate(v):
            if not isinstance(block, BaseBlock):
                raise ValueError(
                    f"Block at index {i} is not a BaseBlock instance: {type(block)}"
                )

        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls, v: dict[str, FlowParameter]
    ) -> dict[str, FlowParameter]:
        """Validate parameter names and ensure they are FlowParameter instances."""
        if not v:
            return v

        validated = {}
        for param_name, param_value in v.items():
            if not isinstance(param_name, str) or not param_name.strip():
                raise ValueError(
                    f"Parameter name must be a non-empty string: {param_name}"
                )

            if not isinstance(param_value, FlowParameter):
                raise ValueError(
                    f"Parameter '{param_name}' must be a FlowParameter instance, "
                    f"got: {type(param_value)}"
                )

            validated[param_name.strip()] = param_value

        return validated

    @model_validator(mode="after")
    def validate_block_names_unique(self) -> "Flow":
        """Ensure all block names are unique within the flow."""
        if not self.blocks:
            return self

        seen_names = set()
        for i, block in enumerate(self.blocks):
            if block.block_name in seen_names:
                raise ValueError(
                    f"Duplicate block name '{block.block_name}' at index {i}. "
                    f"All block names must be unique within a flow."
                )
            seen_names.add(block.block_name)

        return self

    @classmethod
    def from_yaml(cls, yaml_path: str, client: Any = None) -> "Flow":
        """Load flow from YAML configuration file.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML flow configuration file.
        client : Any, optional
            LLM client instance. Required for backward compatibility with old format YAMLs
            that use deprecated LLMBlocks. Ignored for new format YAMLs.

        Returns
        -------
        Flow
            Validated Flow instance.

        Raises
        ------
        FlowValidationError
            If yaml_path is None or the file doesn't exist.
        """
        if yaml_path is None:
            raise FlowValidationError(
                "Flow path cannot be None. Please provide a valid YAML file path or check that the flow exists in the registry."
            )

        yaml_path = resolve_path(yaml_path, [])
        yaml_dir = Path(yaml_path).parent

        logger.info(f"Loading flow from: {yaml_path}")

        # Load YAML file
        try:
            with open(yaml_path, encoding="utf-8") as f:
                flow_config = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Flow file not found: {yaml_path}") from exc
        except yaml.YAMLError as exc:
            raise FlowValidationError(f"Invalid YAML in {yaml_path}: {exc}") from exc

        # Check if this is an old format flow and migrate if necessary
        migrated_runtime_params = None
        is_old_format = FlowMigration.is_old_format(flow_config)
        if is_old_format:
            logger.info(f"Detected old format flow, migrating: {yaml_path}")
            if client is None:
                logger.warning(
                    "Old format YAML detected but no client provided. LLMBlocks may fail."
                )
            flow_config, migrated_runtime_params = FlowMigration.migrate_to_new_format(
                flow_config, yaml_path
            )
            # Save migrated config back to YAML to persist id
            save_flow_yaml(yaml_path, flow_config, "migrated to new format")

        # Validate YAML structure
        validator = FlowValidator()
        validation_errors = validator.validate_yaml_structure(flow_config)
        if validation_errors:
            raise FlowValidationError(
                "Invalid flow configuration:\n" + "\n".join(validation_errors)
            )

        # Extract and validate metadata
        metadata_dict = flow_config.get("metadata", {})
        if "name" not in metadata_dict:
            metadata_dict["name"] = Path(yaml_path).stem

        # Note: Old format compatibility removed - only new RecommendedModels format supported

        try:
            metadata = FlowMetadata(**metadata_dict)
        except Exception as exc:
            raise FlowValidationError(f"Invalid metadata configuration: {exc}") from exc

        # Extract and validate parameters
        parameters = {}
        params_dict = flow_config.get("parameters", {})
        for param_name, param_config in params_dict.items():
            try:
                parameters[param_name] = FlowParameter(**param_config)
            except Exception as exc:
                raise FlowValidationError(
                    f"Invalid parameter '{param_name}': {exc}"
                ) from exc

        # Create blocks with validation
        blocks = []
        block_configs = flow_config.get("blocks", [])

        for i, block_config in enumerate(block_configs):
            try:
                # Inject client for deprecated LLMBlocks if this is an old format flow
                if (
                    is_old_format
                    and block_config.get("block_type") == "LLMBlock"
                    and client is not None
                ):
                    if "block_config" not in block_config:
                        block_config["block_config"] = {}
                    block_config["block_config"]["client"] = client
                    logger.debug(
                        f"Injected client for deprecated LLMBlock: {block_config['block_config'].get('block_name')}"
                    )

                block = cls._create_block_from_config(block_config, yaml_dir)
                blocks.append(block)
            except Exception as exc:
                raise FlowValidationError(
                    f"Failed to create block at index {i}: {exc}"
                ) from exc

        # Create and validate the flow
        try:
            flow = cls(blocks=blocks, metadata=metadata, parameters=parameters)
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
            # Store migrated runtime params and client for backward compatibility
            if migrated_runtime_params:
                flow._migrated_runtime_params = migrated_runtime_params
            if is_old_format and client is not None:
                flow._llm_client = client

            # Check if this is a flow without LLM blocks
            llm_blocks = flow._detect_llm_blocks()
            if not llm_blocks:
                # No LLM blocks, so no model config needed
                flow._model_config_set = True
            else:
                # LLM blocks present - user must call set_model_config()
                flow._model_config_set = False

            return flow
        except Exception as exc:
            raise FlowValidationError(f"Flow validation failed: {exc}") from exc

    @classmethod
    def _create_block_from_config(
        cls,
        block_config: dict[str, Any],
        yaml_dir: Path,
    ) -> BaseBlock:
        """Create a block instance from configuration with validation.

        Parameters
        ----------
        block_config : Dict[str, Any]
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
                config[path_key] = cls._resolve_config_paths(config[path_key], yaml_dir)

        # Create block instance with Pydantic validation
        try:
            return block_class(**config)
        except Exception as exc:
            raise FlowValidationError(
                f"Failed to create block '{block_type_name}' with config {config}: {exc}"
            ) from exc

    @classmethod
    def _resolve_config_paths(
        cls, paths: Union[str, list[str], dict[str, str]], yaml_dir: Path
    ) -> Union[str, list[str], dict[str, str]]:
        """Resolve configuration file paths relative to YAML directory."""
        if isinstance(paths, str):
            return str(yaml_dir / paths)
        elif isinstance(paths, list):
            return [str(yaml_dir / path) for path in paths]
        elif isinstance(paths, dict):
            return {key: str(yaml_dir / path) for key, path in paths.items()}
        return paths

    def generate(
        self,
        dataset: Dataset,
        runtime_params: Optional[dict[str, dict[str, Any]]] = None,
        checkpoint_dir: Optional[str] = None,
        save_freq: Optional[int] = None,
        log_dir: Optional[str] = None,
        max_concurrency: Optional[int] = None,
    ) -> Dataset:
        """Execute the flow blocks in sequence to generate data.

        Note: For flows with LLM blocks, set_model_config() must be called first
        to configure model settings before calling generate().

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
        checkpoint_dir : Optional[str], optional
            Directory to save/load checkpoints. If provided, enables checkpointing.
        save_freq : Optional[int], optional
            Number of completed samples after which to save a checkpoint.
            If None, only saves final results when checkpointing is enabled.
        log_dir : Optional[str], optional
            Directory to save execution logs. If provided, logs will be written to both
            console and a log file in this directory. Maintains backward compatibility
            when None.
        max_concurrency : Optional[int], optional
            Maximum number of concurrent requests across all blocks.
            Controls async request concurrency to prevent overwhelming servers.

        Returns
        -------
        Dataset
            Processed dataset after all blocks have been executed.

        Raises
        ------
        EmptyDatasetError
            If input dataset is empty or any block produces an empty dataset.
        FlowValidationError
            If flow validation fails or if model configuration is required but not set.
        """
        # Validate save_freq parameter early to prevent range() errors
        if save_freq is not None and save_freq <= 0:
            raise FlowValidationError(
                f"save_freq must be greater than 0, got {save_freq}"
            )

        # Set up file logging if log_dir is provided
        flow_logger = logger  # Use global logger by default
        if log_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            flow_name = self.metadata.name.replace(" ", "_").lower()
            log_filename = f"{flow_name}_{timestamp}.log"

            # Create a flow-specific logger for this execution
            unique_id = str(uuid.uuid4())[:8]  # Short unique ID
            flow_logger_name = f"{__name__}.flow_{flow_name}_{timestamp}_{unique_id}"
            flow_logger = setup_logger(
                flow_logger_name, log_dir=log_dir, log_filename=log_filename
            )
            flow_logger.propagate = False
            flow_logger.info(
                f"Flow logging enabled - logs will be saved to: {log_dir}/{log_filename}"
            )
        # Validate max_concurrency parameter
        if max_concurrency is not None:
            # Explicitly reject boolean values (bool is a subclass of int in Python)
            if isinstance(max_concurrency, bool) or not isinstance(
                max_concurrency, int
            ):
                raise FlowValidationError(
                    f"max_concurrency must be an int, got {type(max_concurrency).__name__}"
                )
            if max_concurrency <= 0:
                raise FlowValidationError(
                    f"max_concurrency must be greater than 0, got {max_concurrency}"
                )

        # Validate preconditions
        if not self.blocks:
            raise FlowValidationError("Cannot generate with empty flow")

        if len(dataset) == 0:
            raise EmptyDatasetError("Input dataset is empty")

        validate_no_duplicates(dataset)

        # Check if model configuration has been set for flows with LLM blocks
        llm_blocks = self._detect_llm_blocks()
        if llm_blocks and not self._model_config_set:
            raise FlowValidationError(
                f"Model configuration required before generate(). "
                f"Found {len(llm_blocks)} LLM blocks: {sorted(llm_blocks)}. "
                f"Call flow.set_model_config() first."
            )

        # Validate dataset requirements
        dataset_errors = self.validate_dataset(dataset)
        if dataset_errors:
            raise FlowValidationError(
                "Dataset validation failed:\n" + "\n".join(dataset_errors)
            )

        # Log concurrency control if specified
        if max_concurrency is not None:
            logger.info(f"Using max_concurrency={max_concurrency} for LLM requests")

        # Initialize checkpointer if enabled
        checkpointer = None
        completed_dataset = None
        if checkpoint_dir:
            checkpointer = FlowCheckpointer(
                checkpoint_dir=checkpoint_dir,
                save_freq=save_freq,
                flow_id=self.metadata.id,
            )

            # Load existing progress
            remaining_dataset, completed_dataset = checkpointer.load_existing_progress(
                dataset
            )

            if len(remaining_dataset) == 0:
                flow_logger.info(
                    "All samples already completed, returning existing results"
                )
                if log_dir is not None and flow_logger is not logger:
                    for h in list(getattr(flow_logger, "handlers", [])):
                        try:
                            h.flush()
                            h.close()
                        except Exception:
                            pass
                        finally:
                            flow_logger.removeHandler(h)

                return completed_dataset

            dataset = remaining_dataset
            flow_logger.info(f"Resuming with {len(dataset)} remaining samples")

        flow_logger.info(
            f"Starting flow '{self.metadata.name}' v{self.metadata.version} "
            f"with {len(dataset)} samples across {len(self.blocks)} blocks"
            + (f" (max_concurrency={max_concurrency})" if max_concurrency else "")
        )

        # Reset metrics for this execution
        self._block_metrics = []
        run_start = time.perf_counter()

        # Merge migrated runtime params with provided ones (provided ones take precedence)
        merged_runtime_params = self._migrated_runtime_params.copy()
        if runtime_params:
            merged_runtime_params.update(runtime_params)
        runtime_params = merged_runtime_params

        # Execute flow with metrics capture, ensuring metrics are always displayed/saved
        final_dataset = None
        execution_successful = False

        try:
            # Process dataset in chunks if checkpointing with save_freq
            if checkpointer and save_freq:
                all_processed = []

                # Process in chunks of save_freq
                for i in range(0, len(dataset), save_freq):
                    chunk_end = min(i + save_freq, len(dataset))
                    chunk_dataset = dataset.select(range(i, chunk_end))

                    flow_logger.info(
                        f"Processing chunk {i // save_freq + 1}: samples {i} to {chunk_end - 1}"
                    )

                    # Execute all blocks on this chunk
                    processed_chunk = self._execute_blocks_on_dataset(
                        chunk_dataset, runtime_params, flow_logger, max_concurrency
                    )
                    all_processed.append(processed_chunk)

                    # Save checkpoint after chunk completion
                    checkpointer.add_completed_samples(processed_chunk)

                # Save final checkpoint for any remaining samples
                checkpointer.save_final_checkpoint()

                # Combine all processed chunks
                final_dataset = safe_concatenate_with_validation(
                    all_processed, "processed chunks from flow execution"
                )

                # Combine with previously completed samples if any
                if checkpointer and completed_dataset:
                    final_dataset = safe_concatenate_with_validation(
                        [completed_dataset, final_dataset],
                        "completed checkpoint data with newly processed data",
                    )

            else:
                # Process entire dataset at once
                final_dataset = self._execute_blocks_on_dataset(
                    dataset, runtime_params, flow_logger, max_concurrency
                )

                # Save final checkpoint if checkpointing enabled
                if checkpointer:
                    checkpointer.add_completed_samples(final_dataset)
                    checkpointer.save_final_checkpoint()

                    # Combine with previously completed samples if any
                    if completed_dataset:
                        final_dataset = safe_concatenate_with_validation(
                            [completed_dataset, final_dataset],
                            "completed checkpoint data with newly processed data",
                        )

            execution_successful = True

        finally:
            # Always display metrics and save JSON, even if execution failed
            display_metrics_summary(
                self._block_metrics, self.metadata.name, final_dataset
            )

            # Save metrics to JSON if log_dir is provided
            if log_dir is not None:
                # Ensure necessary variables exist
                if "timestamp" not in locals() or "flow_name" not in locals():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    flow_name = self.metadata.name.replace(" ", "_").lower()

                save_metrics_to_json(
                    self._block_metrics,
                    self.metadata.name,
                    self.metadata.version,
                    execution_successful,
                    run_start,
                    log_dir,
                    timestamp,
                    flow_name,
                    flow_logger,
                )

        # Keep a basic log entry for file logs (only if execution was successful)
        if execution_successful and final_dataset is not None:
            flow_logger.info(
                f"Flow '{self.metadata.name}' completed successfully: "
                f"{len(final_dataset)} final samples, "
                f"{len(final_dataset.column_names)} final columns"
            )

        # Close file handlers if we opened a flow-specific logger
        if log_dir is not None and flow_logger is not logger:
            for h in list(getattr(flow_logger, "handlers", [])):
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
                finally:
                    flow_logger.removeHandler(h)

        return final_dataset

    def _execute_blocks_on_dataset(
        self,
        dataset: Dataset,
        runtime_params: dict[str, dict[str, Any]],
        flow_logger=None,
        max_concurrency: Optional[int] = None,
    ) -> Dataset:
        """Execute all blocks in sequence on the given dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset to process through all blocks.
        runtime_params : Dict[str, Dict[str, Any]]
            Runtime parameters for block execution.
        flow_logger : logging.Logger, optional
            Logger to use for this execution. Falls back to global logger if None.
        max_concurrency : Optional[int], optional
            Maximum concurrency for LLM requests across blocks.

        Returns
        -------
        Dataset
            Dataset after processing through all blocks.
        """
        # Use provided logger or fall back to global logger
        exec_logger = flow_logger if flow_logger is not None else logger
        current_dataset = dataset

        # Execute blocks in sequence
        for i, block in enumerate(self.blocks):
            exec_logger.info(
                f"Executing block {i + 1}/{len(self.blocks)}: "
                f"{block.block_name} ({block.__class__.__name__})"
            )

            # Prepare block execution parameters
            block_kwargs = self._prepare_block_kwargs(block, runtime_params)

            # Add max_concurrency to block kwargs if provided
            if max_concurrency is not None:
                block_kwargs["_flow_max_concurrency"] = max_concurrency

            # Capture metrics before execution
            start_time = time.perf_counter()
            input_rows = len(current_dataset)
            input_cols = set(current_dataset.column_names)

            try:
                # Check if this is a deprecated block and skip validations
                is_deprecated_block = (
                    hasattr(block, "__class__")
                    and hasattr(block.__class__, "__module__")
                    and "deprecated_blocks" in block.__class__.__module__
                )

                if is_deprecated_block:
                    exec_logger.debug(
                        f"Skipping validations for deprecated block: {block.block_name}"
                    )
                    # Call generate() directly to skip validations, but keep the runtime params
                    current_dataset = block.generate(current_dataset, **block_kwargs)
                else:
                    # Execute block with validation and logging
                    current_dataset = block(current_dataset, **block_kwargs)

                # Validate output
                if len(current_dataset) == 0:
                    raise EmptyDatasetError(
                        f"Block '{block.block_name}' produced empty dataset"
                    )

                # Capture metrics after successful execution
                execution_time = time.perf_counter() - start_time
                output_rows = len(current_dataset)
                output_cols = set(current_dataset.column_names)
                added_cols = output_cols - input_cols
                removed_cols = input_cols - output_cols

                # Store block metrics
                self._block_metrics.append(
                    {
                        "block_name": block.block_name,
                        "block_type": block.__class__.__name__,
                        "execution_time": execution_time,
                        "input_rows": input_rows,
                        "output_rows": output_rows,
                        "added_cols": list(added_cols),
                        "removed_cols": list(removed_cols),
                        "status": "success",
                    }
                )

                exec_logger.info(
                    f"Block '{block.block_name}' completed successfully: "
                    f"{len(current_dataset)} samples, "
                    f"{len(current_dataset.column_names)} columns"
                )

            except Exception as exc:
                # Capture metrics for failed execution
                execution_time = time.perf_counter() - start_time
                self._block_metrics.append(
                    {
                        "block_name": block.block_name,
                        "block_type": block.__class__.__name__,
                        "execution_time": execution_time,
                        "input_rows": input_rows,
                        "output_rows": 0,
                        "added_cols": [],
                        "removed_cols": [],
                        "status": "failed",
                        "error": str(exc),
                    }
                )

                exec_logger.error(
                    f"Block '{block.block_name}' failed during execution: {exc}"
                )
                raise FlowValidationError(
                    f"Block '{block.block_name}' execution failed: {exc}"
                ) from exc

        return current_dataset

    def _prepare_block_kwargs(
        self, block: BaseBlock, runtime_params: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Prepare execution parameters for a block."""
        return runtime_params.get(block.block_name, {})

    def set_model_config(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        blocks: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Configure model settings for LLM blocks in this flow (in-place).

        This method is designed to work with model-agnostic flow definitions where
        LLM blocks don't have hardcoded model configurations in the YAML. Instead,
        model settings are configured at runtime using this method.

        Based on LiteLLM's basic usage pattern, this method focuses on the core
        parameters (model, api_base, api_key) with additional parameters passed via kwargs.

        By default, auto-detects all LLM blocks in the flow and applies configuration to them.
        Optionally allows targeting specific blocks only.

        Parameters
        ----------
        model : Optional[str]
            Model name to configure (e.g., "hosted_vllm/openai/gpt-oss-120b").
        api_base : Optional[str]
            API base URL to configure (e.g., "http://localhost:8101/v1").
        api_key : Optional[str]
            API key to configure.
        blocks : Optional[List[str]]
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
            config_params["api_key"] = api_key

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
            existing_block_names = {block.block_name for block in self.blocks}
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
            target_block_names = set(self._detect_llm_blocks())
            logger.info(
                f"Auto-detected {len(target_block_names)} LLM blocks for configuration: {sorted(target_block_names)}"
            )

        # Apply configuration to target blocks
        modified_count = 0
        for block in self.blocks:
            if block.block_name in target_block_names:
                for param_name, param_value in config_params.items():
                    if hasattr(block, param_name):
                        old_value = getattr(block, param_name)
                        setattr(block, param_name, param_value)
                        logger.debug(
                            f"Block '{block.block_name}': {param_name} "
                            f"'{old_value}' -> '{param_value}'"
                        )
                    ## check if allow extra
                    elif block.model_config["extra"] == "allow":
                        setattr(block, param_name, param_value)
                        logger.debug(
                            f"Block '{block.block_name}': {param_name} "
                            f"'{old_value}' -> '{param_value}'"
                        )
                    else:
                        logger.warning(
                            f"Block '{block.block_name}' ({block.__class__.__name__}) "
                            f"does not have attribute '{param_name}' - skipping"
                        )

                modified_count += 1

        if modified_count > 0:
            # Enhanced logging showing what was configured
            param_summary = []
            for param_name, param_value in config_params.items():
                if param_name == "model":
                    param_summary.append(f"model: '{param_value}'")
                elif param_name == "api_base":
                    param_summary.append(f"api_base: '{param_value}'")
                else:
                    param_summary.append(f"{param_name}: {param_value}")

            logger.info(
                f"Successfully configured {modified_count} LLM blocks with: {', '.join(param_summary)}"
            )
            logger.info(f"Configured blocks: {sorted(target_block_names)}")

            # Mark that model configuration has been set
            self._model_config_set = True
        else:
            logger.warning(
                "No blocks were modified - check block names or LLM block detection"
            )

    def _detect_llm_blocks(self) -> list[str]:
        """Detect LLM blocks in the flow by checking for model-related attribute existence.

        LLM blocks are identified by having model, api_base, or api_key attributes,
        regardless of their values (they may be None until set_model_config() is called).

        Returns
        -------
        List[str]
            List of block names that have LLM-related attributes.
        """
        llm_blocks = []

        for block in self.blocks:
            block_type = block.__class__.__name__
            block_name = block.block_name

            # Check by attribute existence (not value) - LLM blocks have these attributes even if None
            has_model_attr = hasattr(block, "model")
            has_api_base_attr = hasattr(block, "api_base")
            has_api_key_attr = hasattr(block, "api_key")

            # A block is considered an LLM block if it has any LLM-related attributes
            is_llm_block = has_model_attr or has_api_base_attr or has_api_key_attr

            if is_llm_block:
                llm_blocks.append(block_name)
                logger.debug(
                    f"Detected LLM block '{block_name}' ({block_type}): "
                    f"has_model_attr={has_model_attr}, has_api_base_attr={has_api_base_attr}, has_api_key_attr={has_api_key_attr}"
                )

        return llm_blocks

    def is_model_config_required(self) -> bool:
        """Check if model configuration is required for this flow.

        Returns
        -------
        bool
            True if flow has LLM blocks and needs model configuration.
        """
        return len(self._detect_llm_blocks()) > 0

    def is_model_config_set(self) -> bool:
        """Check if model configuration has been set.

        Returns
        -------
        bool
            True if model configuration has been set or is not required.
        """
        return self._model_config_set

    def reset_model_config(self) -> None:
        """Reset model configuration flag (useful for testing or reconfiguration).

        After calling this, set_model_config() must be called again before generate().
        """
        if self.is_model_config_required():
            self._model_config_set = False
            logger.info(
                "Model configuration flag reset - call set_model_config() before generate()"
            )

    def get_default_model(self) -> Optional[str]:
        """Get the default recommended model for this flow.

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
        if not self.metadata.recommended_models:
            return None
        return self.metadata.recommended_models.default

    def get_model_recommendations(self) -> dict[str, Any]:
        """Get a clean summary of model recommendations for this flow.

        Returns
        -------
        Dict[str, Any]
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
        if not self.metadata.recommended_models:
            return {
                "default": None,
                "compatible": [],
                "experimental": [],
            }

        return {
            "default": self.metadata.recommended_models.default,
            "compatible": self.metadata.recommended_models.compatible,
            "experimental": self.metadata.recommended_models.experimental,
        }

    def validate_dataset(self, dataset: Dataset) -> list[str]:
        """Validate dataset against flow requirements."""
        errors = []

        if len(dataset) == 0:
            errors.append("Dataset is empty")

        if self.metadata.dataset_requirements:
            errors.extend(
                self.metadata.dataset_requirements.validate_dataset(
                    dataset.column_names, len(dataset)
                )
            )

        return errors

    def dry_run(
        self,
        dataset: Dataset,
        sample_size: int = 2,
        runtime_params: Optional[dict[str, dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Perform a dry run of the flow with a subset of data.

        Parameters
        ----------
        dataset : Dataset
            Input dataset to test with.
        sample_size : int, default=2
            Number of samples to use for dry run testing.
        runtime_params : Optional[Dict[str, Dict[str, Any]]], optional
            Runtime parameters organized by block name.

        Returns
        -------
        Dict[str, Any]
            Dry run results with execution info and sample outputs.

        Raises
        ------
        EmptyDatasetError
            If input dataset is empty.
        FlowValidationError
            If any block fails during dry run execution.
        """
        # Validate preconditions
        if not self.blocks:
            raise FlowValidationError("Cannot dry run empty flow")

        if len(dataset) == 0:
            raise EmptyDatasetError("Input dataset is empty")

        validate_no_duplicates(dataset)

        # Use smaller sample size if dataset is smaller
        actual_sample_size = min(sample_size, len(dataset))

        logger.info(
            f"Starting dry run for flow '{self.metadata.name}' "
            f"with {actual_sample_size} samples"
        )

        # Create subset dataset
        sample_dataset = dataset.select(range(actual_sample_size))

        # Initialize dry run results
        dry_run_results = {
            "flow_name": self.metadata.name,
            "flow_version": self.metadata.version,
            "sample_size": actual_sample_size,
            "original_dataset_size": len(dataset),
            "input_columns": dataset.column_names,
            "blocks_executed": [],
            "final_dataset": None,
            "execution_successful": True,
            "execution_time_seconds": 0,
        }

        start_time = time.perf_counter()

        try:
            # Execute the flow with sample data
            current_dataset = sample_dataset
            runtime_params = runtime_params or {}

            for i, block in enumerate(self.blocks):
                block_start_time = time.perf_counter()
                input_rows = len(current_dataset)

                logger.info(
                    f"Dry run executing block {i + 1}/{len(self.blocks)}: "
                    f"{block.block_name} ({block.__class__.__name__})"
                )

                # Prepare block execution parameters
                block_kwargs = self._prepare_block_kwargs(block, runtime_params)

                # Check if this is a deprecated block and skip validations
                is_deprecated_block = (
                    hasattr(block, "__class__")
                    and hasattr(block.__class__, "__module__")
                    and "deprecated_blocks" in block.__class__.__module__
                )

                if is_deprecated_block:
                    logger.debug(
                        f"Dry run: Skipping validations for deprecated block: {block.block_name}"
                    )
                    # Call generate() directly to skip validations, but keep the runtime params
                    current_dataset = block.generate(current_dataset, **block_kwargs)
                else:
                    # Execute block with validation and logging
                    current_dataset = block(current_dataset, **block_kwargs)

                block_execution_time = time.time() - block_start_time

                # Record block execution info
                block_info = {
                    "block_name": block.block_name,
                    "block_type": block.__class__.__name__,
                    "execution_time_seconds": block_execution_time,
                    "input_rows": input_rows,
                    "output_rows": len(current_dataset),
                    "output_columns": current_dataset.column_names,
                    "parameters_used": block_kwargs,
                }

                dry_run_results["blocks_executed"].append(block_info)

                logger.info(
                    f"Dry run block '{block.block_name}' completed: "
                    f"{len(current_dataset)} samples, "
                    f"{len(current_dataset.column_names)} columns, "
                    f"{block_execution_time:.2f}s"
                )

            # Store final results
            dry_run_results["final_dataset"] = {
                "rows": len(current_dataset),
                "columns": current_dataset.column_names,
                "sample_data": current_dataset.to_dict()
                if len(current_dataset) > 0
                else {},
            }

            execution_time = time.perf_counter() - start_time
            dry_run_results["execution_time_seconds"] = execution_time

            logger.info(
                f"Dry run completed successfully for flow '{self.metadata.name}' "
                f"in {execution_time:.2f}s"
            )

            return dry_run_results

        except Exception as exc:
            execution_time = time.perf_counter() - start_time
            dry_run_results["execution_successful"] = False
            dry_run_results["execution_time_seconds"] = execution_time
            dry_run_results["error"] = str(exc)

            logger.error(f"Dry run failed for flow '{self.metadata.name}': {exc}")

            raise FlowValidationError(f"Dry run failed: {exc}") from exc

    def add_block(self, block: BaseBlock) -> "Flow":
        """Add a block to the flow, returning a new Flow instance.

        Parameters
        ----------
        block : BaseBlock
            Block to add to the flow.

        Returns
        -------
        Flow
            New Flow instance with the added block.

        Raises
        ------
        ValueError
            If the block is invalid or creates naming conflicts.
        """
        if not isinstance(block, BaseBlock):
            raise ValueError(f"Block must be a BaseBlock instance, got: {type(block)}")

        # Check for name conflicts
        existing_names = {b.block_name for b in self.blocks}
        if block.block_name in existing_names:
            raise ValueError(
                f"Block name '{block.block_name}' already exists in flow. "
                f"Block names must be unique."
            )

        # Create new flow with added block
        new_blocks = self.blocks + [block]

        return Flow(
            blocks=new_blocks, metadata=self.metadata, parameters=self.parameters
        )

    def get_info(self) -> dict[str, Any]:
        """Get information about the flow."""
        return {
            "metadata": self.metadata.model_dump(),
            "parameters": {
                name: param.model_dump() for name, param in self.parameters.items()
            },
            "blocks": [
                {
                    "block_type": block.__class__.__name__,
                    "block_name": block.block_name,
                    "input_cols": getattr(block, "input_cols", None),
                    "output_cols": getattr(block, "output_cols", None),
                }
                for block in self.blocks
            ],
            "total_blocks": len(self.blocks),
            "block_names": [block.block_name for block in self.blocks],
        }

    def get_dataset_requirements(self) -> Optional[DatasetRequirements]:
        """Get the dataset requirements for this flow.

        Returns
        -------
        Optional[DatasetRequirements]
            Dataset requirements object or None if not defined.

        Examples
        --------
        >>> flow = Flow.from_yaml("path/to/flow.yaml")
        >>> requirements = flow.get_dataset_requirements()
        >>> if requirements:
        ...     print(f"Required columns: {requirements.required_columns}")
        """
        return self.metadata.dataset_requirements

    def get_dataset_schema(self) -> Dataset:
        """Get an empty dataset with the correct schema for this flow.

        Returns
        -------
        Dataset
            Empty HuggingFace Dataset with the correct schema/features for this flow.
            Users can add data to this dataset or use it to validate their own dataset schema.

        Examples
        --------
        >>> flow = Flow.from_yaml("path/to/flow.yaml")
        >>> schema_dataset = flow.get_dataset_schema()
        >>>
        >>> # Add your data
        >>> schema_dataset = schema_dataset.add_item({
        ...     "document": "Your document text",
        ...     "domain": "Computer Science",
        ...     "icl_document": "Example document"
        ... })
        >>>
        >>> # Or validate your existing dataset schema
        >>> my_dataset = Dataset.from_dict(my_data)
        >>> if my_dataset.features == schema_dataset.features:
        ...     print("Schema matches!")
        """

        requirements = self.get_dataset_requirements()

        if requirements is None:
            # Return empty dataset with no schema requirements
            return Dataset.from_dict({})

        # Build schema features
        schema_features = {}

        # Process required columns
        for col_name in requirements.required_columns:
            col_type = requirements.column_types.get(col_name, "string")
            schema_features[col_name] = self._map_column_type_to_feature(col_type)

        # Process optional columns
        for col_name in requirements.optional_columns:
            col_type = requirements.column_types.get(col_name, "string")
            schema_features[col_name] = self._map_column_type_to_feature(col_type)

        # Create empty dataset with the correct features
        features = datasets.Features(schema_features)
        empty_data = {col_name: [] for col_name in schema_features.keys()}

        return Dataset.from_dict(empty_data, features=features)

    def _map_column_type_to_feature(self, col_type: str):
        """Map column type string to HuggingFace feature type."""
        # Map common type names to HuggingFace types
        if col_type in ["str", "string", "text"]:
            return datasets.Value("string")
        elif col_type in ["int", "integer"]:
            return datasets.Value("int64")
        elif col_type in ["float", "number"]:
            return datasets.Value("float64")
        elif col_type in ["bool", "boolean"]:
            return datasets.Value("bool")
        else:
            # Default to string for unknown types
            return datasets.Value("string")

    def print_info(self) -> None:
        """
        Print an interactive summary of the Flow in the console.

        The summary contains:
        1. Flow metadata (name, version, author, description)
        2. Defined runtime parameters with type hints and defaults
        3. A table of all blocks with their input and output columns

        Notes
        -----
        Uses the `rich` library for colourised output; install with
        `pip install rich` if not already present.

        Returns
        -------
        None
        """

        console = Console()

        # Create main tree structure
        flow_tree = Tree(
            f"[bold bright_blue]{self.metadata.name}[/bold bright_blue] Flow"
        )

        # Metadata section
        metadata_branch = flow_tree.add(
            "[bold bright_green]Metadata[/bold bright_green]"
        )
        metadata_branch.add(
            f"Version: [bright_cyan]{self.metadata.version}[/bright_cyan]"
        )
        metadata_branch.add(
            f"Author: [bright_cyan]{self.metadata.author}[/bright_cyan]"
        )
        if self.metadata.description:
            metadata_branch.add(
                f"Description: [white]{self.metadata.description}[/white]"
            )

        # Parameters section
        if self.parameters:
            params_branch = flow_tree.add(
                "[bold bright_yellow]Parameters[/bold bright_yellow]"
            )
            for name, param in self.parameters.items():
                param_info = f"[bright_cyan]{name}[/bright_cyan]: [white]{param.type_hint}[/white]"
                if param.default is not None:
                    param_info += f" = [bright_white]{param.default}[/bright_white]"
                params_branch.add(param_info)

        # Blocks overview
        flow_tree.add(
            f"[bold bright_magenta]Blocks[/bold bright_magenta] ({len(self.blocks)} total)"
        )

        # Create blocks table
        blocks_table = Table(show_header=True, header_style="bold bright_white")
        blocks_table.add_column("Block Name", style="bright_cyan")
        blocks_table.add_column("Type", style="bright_green")
        blocks_table.add_column("Input Cols", style="bright_yellow")
        blocks_table.add_column("Output Cols", style="bright_red")

        for block in self.blocks:
            input_cols = getattr(block, "input_cols", None)
            output_cols = getattr(block, "output_cols", None)

            blocks_table.add_row(
                block.block_name,
                block.__class__.__name__,
                str(input_cols) if input_cols else "[bright_black]None[/bright_black]",
                str(output_cols)
                if output_cols
                else "[bright_black]None[/bright_black]",
            )

        # Print everything
        console.print()
        console.print(
            Panel(
                flow_tree,
                title="[bold bright_white]Flow Information[/bold bright_white]",
                border_style="bright_blue",
            )
        )
        console.print()
        console.print(
            Panel(
                blocks_table,
                title="[bold bright_white]Block Details[/bold bright_white]",
                border_style="bright_magenta",
            )
        )
        console.print()

    def to_yaml(self, output_path: str) -> None:
        """Save flow configuration to YAML file.

        Note: This creates a basic YAML structure. For exact reproduction
        of original YAML, save the original file separately.
        """
        config = {
            "metadata": self.metadata.model_dump(),
            "blocks": [
                {
                    "block_type": block.__class__.__name__,
                    "block_config": block.model_dump(),
                }
                for block in self.blocks
            ],
        }

        if self.parameters:
            config["parameters"] = {
                name: param.model_dump() for name, param in self.parameters.items()
            }

        save_flow_yaml(output_path, config)

    def __len__(self) -> int:
        """Number of blocks in the flow."""
        return len(self.blocks)

    def __repr__(self) -> str:
        """String representation of the flow."""
        return (
            f"Flow(name='{self.metadata.name}', "
            f"version='{self.metadata.version}', "
            f"blocks={len(self.blocks)})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        block_names = [block.block_name for block in self.blocks]
        return (
            f"Flow '{self.metadata.name}' v{self.metadata.version}\n"
            f"Blocks: {' -> '.join(block_names) if block_names else 'None'}\n"
            f"Author: {self.metadata.author or 'Unknown'}\n"
            f"Description: {self.metadata.description or 'No description'}"
        )
