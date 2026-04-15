# SPDX-License-Identifier: Apache-2.0
"""Flow execution helper functions for Flow class."""

# Standard
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union
import time
import uuid

# Third Party
import datasets
import pandas as pd

# Local
from ..blocks.base import BaseBlock
from ..utils.datautils import safe_concatenate_with_validation, validate_no_duplicates
from ..utils.error_handling import EmptyDatasetError, FlowValidationError
from ..utils.flow_metrics import (
    display_metrics_summary,
    display_time_estimation_summary,
    save_metrics_to_json,
)
from ..utils.logger_config import setup_logger
from ..utils.time_estimator import estimate_execution_time
from .checkpointer import FlowCheckpointer
from .column_tracker import ColumnDependencyTracker

if TYPE_CHECKING:
    from .base import Flow

logger = setup_logger(__name__)


def _validate_max_concurrency(max_concurrency: Optional[int]) -> None:
    """Validate the max_concurrency parameter.

    Parameters
    ----------
    max_concurrency : Optional[int]
        Maximum concurrency value to validate.

    Raises
    ------
    FlowValidationError
        If max_concurrency is invalid (not an int, bool, or <= 0).
    """
    if max_concurrency is not None:
        # Explicitly reject boolean values (bool is a subclass of int in Python)
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
            raise FlowValidationError(
                f"max_concurrency must be an int, got {type(max_concurrency).__name__}"
            )
        if max_concurrency <= 0:
            raise FlowValidationError(
                f"max_concurrency must be greater than 0, got {max_concurrency}"
            )


def _close_flow_logger(flow_logger, module_logger) -> None:
    """Close file handlers on a flow-specific logger.

    Parameters
    ----------
    flow_logger : logging.Logger
        The flow-specific logger to close.
    module_logger : logging.Logger
        The module-level logger (to check if flow_logger is different).
    """
    if flow_logger is not module_logger:
        for h in list(getattr(flow_logger, "handlers", [])):
            try:
                h.flush()
                h.close()
            except Exception:
                # Ignore errors during cleanup - handler may already be closed
                # or in an invalid state. We still want to remove it.
                pass
            finally:
                flow_logger.removeHandler(h)


def convert_to_dataframe(
    dataset: Union[pd.DataFrame, datasets.Dataset],
) -> tuple[pd.DataFrame, bool]:
    """Convert datasets.Dataset to pd.DataFrame if needed (backwards compatibility).

    Parameters
    ----------
    dataset : Union[pd.DataFrame, datasets.Dataset]
        Input dataset in either format.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        Tuple of (converted DataFrame, was_dataset flag).
        was_dataset is True if input was a datasets.Dataset, False if it was already a DataFrame.
    """
    if isinstance(dataset, datasets.Dataset):
        logger.info("Converting datasets.Dataset to pd.DataFrame for processing")
        return dataset.to_pandas(), True
    return dataset, False


def convert_from_dataframe(
    df: pd.DataFrame, should_convert: bool
) -> Union[pd.DataFrame, datasets.Dataset]:
    """Convert pd.DataFrame back to datasets.Dataset if needed (backwards compatibility).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to potentially convert.
    should_convert : bool
        If True, convert to datasets.Dataset. If False, return as-is.

    Returns
    -------
    Union[pd.DataFrame, datasets.Dataset]
        Original DataFrame or converted Dataset, matching the input type.
    """
    if should_convert:
        logger.info(
            "Converting pd.DataFrame back to datasets.Dataset to match input type"
        )
        return datasets.Dataset.from_pandas(df)
    return df


def validate_flow_dataset(
    flow: "Flow", dataset: Union[pd.DataFrame, datasets.Dataset]
) -> list[str]:
    """Validate dataset against flow requirements.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    dataset : Union[pd.DataFrame, datasets.Dataset]
        Dataset to validate. Can be either pandas DataFrame or HuggingFace Dataset
        (will be automatically converted to DataFrame for backwards compatibility).

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid).
    """
    # Convert to DataFrame if needed (backwards compatibility)
    dataset, _ = convert_to_dataframe(dataset)

    errors = []

    if len(dataset) == 0:
        errors.append("Dataset is empty")

    if flow.metadata.dataset_requirements:
        # Get column names
        columns = dataset.columns.tolist()

        errors.extend(
            flow.metadata.dataset_requirements.validate_dataset(columns, len(dataset))
        )

    return errors


def prepare_block_kwargs(
    block: BaseBlock, runtime_params: Optional[dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    """Prepare execution parameters for a block.

    Parameters
    ----------
    block : BaseBlock
        The block to prepare kwargs for.
    runtime_params : Optional[dict[str, dict[str, Any]]]
        Runtime parameters organized by block name.

    Returns
    -------
    dict[str, Any]
        Prepared kwargs for the block.
    """
    if runtime_params is None:
        return {}
    return runtime_params.get(block.block_name, {})


def execute_blocks_on_dataset(
    flow: "Flow",
    dataset: pd.DataFrame,
    runtime_params: dict[str, dict[str, Any]],
    flow_logger=None,
    max_concurrency: Optional[int] = None,
    column_tracker: Optional[ColumnDependencyTracker] = None,
) -> pd.DataFrame:
    """Execute all blocks in sequence on the given dataset.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    dataset : pd.DataFrame
        Dataset to process through all blocks.
    runtime_params : dict[str, dict[str, Any]]
        Runtime parameters for block execution.
    flow_logger : logging.Logger, optional
        Logger to use for this execution. Falls back to global logger if None.
    max_concurrency : Optional[int], optional
        Maximum concurrency for LLM requests across blocks.
    column_tracker : Optional[ColumnDependencyTracker], optional
        Tracker for early dropping of unused columns. If None, no early dropping.

    Returns
    -------
    pd.DataFrame
        Dataset after processing through all blocks.
    """
    # Use provided logger or fall back to global logger
    exec_logger = flow_logger if flow_logger is not None else logger
    current_dataset = dataset

    # Execute blocks in sequence
    for i, block in enumerate(flow.blocks):
        exec_logger.info(
            f"Executing block {i + 1}/{len(flow.blocks)}: "
            f"{block.block_name} ({block.__class__.__name__})"
        )

        # Prepare block execution parameters
        block_kwargs = prepare_block_kwargs(block, runtime_params)

        # Add max_concurrency to block kwargs if provided
        if max_concurrency is not None:
            block_kwargs["_flow_max_concurrency"] = max_concurrency

        # Capture metrics before execution
        start_time = time.perf_counter()
        input_rows = len(current_dataset)
        input_cols = set(current_dataset.columns)

        try:
            # Execute block with validation and logging
            current_dataset = block(current_dataset, **block_kwargs)

            # Validate output
            if len(current_dataset) == 0:
                raise EmptyDatasetError(block.block_name)

            # Capture metrics after successful execution
            execution_time = time.perf_counter() - start_time
            output_rows = len(current_dataset)
            output_cols = set(current_dataset.columns)
            added_cols = output_cols - input_cols
            removed_cols = input_cols - output_cols

            # Store block metrics
            flow._block_metrics.append(
                {
                    "block_name": block.block_name,
                    "block_class": block.__class__.__name__,
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
                f"{len(current_dataset.columns)} columns"
            )

        except EmptyDatasetError:
            # Re-raise EmptyDatasetError directly without wrapping
            raise
        except Exception as exc:
            # Capture metrics for failed execution
            execution_time = time.perf_counter() - start_time
            flow._block_metrics.append(
                {
                    "block_name": block.block_name,
                    "block_class": block.__class__.__name__,
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

        # Drop columns no longer needed by downstream blocks (outside try/except
        # so failures here are not misattributed as block execution errors)
        if column_tracker is not None:
            cols_to_drop = column_tracker.get_droppable_columns(
                i, set(current_dataset.columns)
            )
            if cols_to_drop:
                exec_logger.info(
                    f"Dropping {len(cols_to_drop)} unused columns: "
                    f"{sorted(cols_to_drop)}"
                )
                current_dataset = current_dataset.drop(columns=cols_to_drop)

    return current_dataset


def _cleanup_final_columns(
    dataset: pd.DataFrame,
    output_columns: set[str],
    original_columns: set[str],
    flow_logger,
) -> pd.DataFrame:
    """Drop all columns except output_columns and original input columns.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset to clean up.
    output_columns : set[str]
        Columns that must be preserved in output.
    original_columns : set[str]
        Original input columns (auto-preserved).
    flow_logger
        Logger for this execution.

    Returns
    -------
    pd.DataFrame
        Dataset with only output_columns + original columns retained.

    Raises
    ------
    FlowValidationError
        If any output_columns are missing from the final dataset.
    """
    current_cols = set(dataset.columns)
    cols_to_keep = output_columns | original_columns
    cols_to_drop = current_cols - cols_to_keep
    missing_cols = output_columns - current_cols

    if missing_cols:
        raise FlowValidationError(
            f"output_columns not found in final dataset: {sorted(missing_cols)}. "
            f"Check that your flow's blocks produce these columns."
        )

    if cols_to_drop:
        flow_logger.info(
            f"Dropping {len(cols_to_drop)} intermediate columns: {sorted(cols_to_drop)}"
        )
        dataset = dataset.drop(columns=list(cols_to_drop))

    return dataset


def execute_flow(
    flow: "Flow",
    dataset: Union[pd.DataFrame, datasets.Dataset],
    runtime_params: Optional[dict[str, dict[str, Any]]] = None,
    checkpoint_dir: Optional[str] = None,
    save_freq: Optional[int] = None,
    log_dir: Optional[str] = None,
    max_concurrency: Optional[int] = None,
) -> Union[pd.DataFrame, datasets.Dataset]:
    """Execute the flow blocks in sequence to generate data.

    Note: For flows with LLM blocks, set_model_config() must be called first
    to configure model settings before calling generate().

    Parameters
    ----------
    flow : Flow
        The flow instance to execute.
    dataset : Union[pd.DataFrame, datasets.Dataset]
        Input dataset to process. Can be either pandas DataFrame or HuggingFace Dataset
        (will be automatically converted to DataFrame for backwards compatibility).
    runtime_params : Optional[dict[str, dict[str, Any]]], optional
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
    Union[pd.DataFrame, datasets.Dataset]
        Processed dataset after all blocks have been executed.
        Return type matches the input type (DataFrame in -> DataFrame out, Dataset in -> Dataset out).

    Raises
    ------
    EmptyDatasetError
        If any block produces an empty dataset.
    FlowValidationError
        If flow validation fails, input dataset is empty, or model configuration
        is required but not set.
    """
    # Import here to avoid circular imports
    from .agent_config import detect_agent_blocks
    from .model_config import detect_llm_blocks

    # Convert to DataFrame if needed (backwards compatibility)
    dataset, was_dataset = convert_to_dataframe(dataset)

    # Capture original columns for preservation during cleanup
    original_columns = set(dataset.columns)

    # Determine effective output columns from metadata
    effective_output_columns = (
        set(flow.metadata.output_columns) if flow.metadata.output_columns else None
    )

    # Normalize runtime_params early
    runtime_params = runtime_params or {}

    # Validate save_freq parameter early to prevent range() errors
    if save_freq is not None and save_freq <= 0:
        raise FlowValidationError(f"save_freq must be greater than 0, got {save_freq}")

    # Validate max_concurrency parameter
    _validate_max_concurrency(max_concurrency)

    # Set up file logging if log_dir is provided
    flow_logger = logger  # Use global logger by default
    timestamp = None
    flow_name = None
    if log_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        flow_name = flow.metadata.name.replace(" ", "_").lower()
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

    # Validate preconditions
    if not flow.blocks:
        raise FlowValidationError("Cannot generate with empty flow")

    if len(dataset) == 0:
        raise FlowValidationError("Input dataset is empty")

    validate_no_duplicates(dataset)

    # Check if model configuration has been set for flows with LLM blocks
    llm_blocks = detect_llm_blocks(flow)
    if llm_blocks and not flow._model_config_set:
        raise FlowValidationError(
            f"Model configuration required before generate(). "
            f"Found {len(llm_blocks)} LLM blocks: {sorted(llm_blocks)}. "
            f"Call flow.set_model_config() first."
        )

    # Check if agent configuration has been set for flows with agent blocks
    agent_blocks = detect_agent_blocks(flow)
    if agent_blocks and not flow._agent_config_set:
        raise FlowValidationError(
            f"Agent configuration required before generate(). "
            f"Found {len(agent_blocks)} agent blocks: {sorted(agent_blocks)}. "
            f"Call flow.set_agent_config() first."
        )

    # Validate dataset requirements
    dataset_errors = validate_flow_dataset(flow, dataset)
    if dataset_errors:
        raise FlowValidationError(
            "Dataset validation failed:\n" + "\n".join(dataset_errors)
        )

    # Log concurrency control if specified
    if max_concurrency is not None:
        flow_logger.info(f"Using max_concurrency={max_concurrency} for LLM requests")

    # Initialize checkpointer if enabled
    checkpointer = None
    completed_dataset = None
    if checkpoint_dir:
        checkpointer = FlowCheckpointer(
            checkpoint_dir=checkpoint_dir,
            save_freq=save_freq,
            flow_id=flow.metadata.id,
        )

        # Load existing progress
        remaining_dataset, completed_dataset = checkpointer.load_existing_progress(
            dataset
        )

        if len(remaining_dataset) == 0:
            flow_logger.info(
                "All samples already completed, returning existing results"
            )

            # Apply column cleanup so checkpointed results match normal output
            if effective_output_columns is not None:
                completed_dataset = _cleanup_final_columns(
                    completed_dataset,
                    effective_output_columns,
                    original_columns,
                    flow_logger,
                )

            if log_dir is not None:
                _close_flow_logger(flow_logger, logger)

            return convert_from_dataframe(completed_dataset, was_dataset)

        dataset = remaining_dataset
        flow_logger.info(f"Resuming with {len(dataset)} remaining samples")

    flow_logger.info(
        f"Starting flow '{flow.metadata.name}' v{flow.metadata.version} "
        f"with {len(dataset)} samples across {len(flow.blocks)} blocks"
        + (f" (max_concurrency={max_concurrency})" if max_concurrency else "")
    )

    # Reset metrics for this execution
    flow._block_metrics = []
    run_start = time.perf_counter()

    # Initialize column tracker for early dropping of unused columns
    column_tracker: Optional[ColumnDependencyTracker] = None
    if effective_output_columns is not None:
        column_tracker = ColumnDependencyTracker(
            flow.blocks, effective_output_columns, original_columns
        )
        flow_logger.info(
            "Column cleanup enabled - will drop unused intermediate columns"
        )

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
                chunk_dataset = dataset.iloc[i:chunk_end]

                flow_logger.info(
                    f"Processing chunk {i // save_freq + 1}: samples {i} to {chunk_end - 1}"
                )

                # Execute all blocks on this chunk
                processed_chunk = execute_blocks_on_dataset(
                    flow,
                    chunk_dataset,
                    runtime_params,
                    flow_logger,
                    max_concurrency,
                    column_tracker,
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
            if (
                checkpointer
                and completed_dataset is not None
                and not completed_dataset.empty
            ):
                final_dataset = safe_concatenate_with_validation(
                    [completed_dataset, final_dataset],
                    "completed checkpoint data with newly processed data",
                )

        else:
            # Process entire dataset at once
            final_dataset = execute_blocks_on_dataset(
                flow,
                dataset,
                runtime_params,
                flow_logger,
                max_concurrency,
                column_tracker,
            )

            # Save final checkpoint if checkpointing enabled
            if checkpointer:
                checkpointer.add_completed_samples(final_dataset)
                checkpointer.save_final_checkpoint()

                # Combine with previously completed samples if any
                if completed_dataset is not None and not completed_dataset.empty:
                    final_dataset = safe_concatenate_with_validation(
                        [completed_dataset, final_dataset],
                        "completed checkpoint data with newly processed data",
                    )

        execution_successful = True

        # Drop intermediate columns if output_columns is specified
        if effective_output_columns is not None and final_dataset is not None:
            final_dataset = _cleanup_final_columns(
                final_dataset,
                effective_output_columns,
                original_columns,
                flow_logger,
            )

    finally:
        # Always display metrics and save JSON, even if execution failed
        display_metrics_summary(flow._block_metrics, flow.metadata.name, final_dataset)

        # Save metrics to JSON if log_dir is provided
        if log_dir is not None:
            save_metrics_to_json(
                flow._block_metrics,
                flow.metadata.name,
                flow.metadata.version,
                execution_successful,
                run_start,
                log_dir,
                timestamp,
                flow_name,
                flow_logger,
            )

        # Close file handlers if we opened a flow-specific logger
        if log_dir is not None:
            _close_flow_logger(flow_logger, logger)

    # Keep a basic log entry (only if execution was successful)
    if execution_successful and final_dataset is not None:
        logger.info(
            f"Flow '{flow.metadata.name}' completed successfully: "
            f"{len(final_dataset)} final samples, "
            f"{len(final_dataset.columns)} final columns"
        )

    return convert_from_dataframe(final_dataset, was_dataset)


def run_dry_run(
    flow: "Flow",
    dataset: Union[pd.DataFrame, datasets.Dataset],
    sample_size: int = 2,
    runtime_params: Optional[dict[str, dict[str, Any]]] = None,
    max_concurrency: Optional[int] = None,
    enable_time_estimation: bool = False,
) -> dict[str, Any]:
    """Perform a dry run of the flow with a subset of data.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    dataset : Union[pd.DataFrame, datasets.Dataset]
        Input dataset to test with. Can be either pandas DataFrame or HuggingFace Dataset
        (will be automatically converted to DataFrame for backwards compatibility).
    sample_size : int, default=2
        Number of samples to use for dry run testing.
    runtime_params : Optional[dict[str, dict[str, Any]]], optional
        Runtime parameters organized by block name.
    max_concurrency : Optional[int], optional
        Maximum concurrent requests for LLM blocks. If None, no limit is applied.
    enable_time_estimation : bool, default=False
        If True, estimates execution time for the full dataset and displays it
        in a Rich table. Automatically runs a second dry run if needed for
        accurate scaling analysis.

    Returns
    -------
    dict[str, Any]
        Dry run results with execution info and sample outputs.
        Time estimation is displayed in a table but not included in return value.

    Raises
    ------
    FlowValidationError
        If input dataset is empty or any block fails during dry run execution.
    """
    # Convert to DataFrame if needed (backwards compatibility)
    dataset, _ = convert_to_dataframe(dataset)

    # Validate preconditions
    if not flow.blocks:
        raise FlowValidationError("Cannot dry run empty flow")

    if len(dataset) == 0:
        raise FlowValidationError("Input dataset is empty")

    validate_no_duplicates(dataset)

    # Validate max_concurrency parameter
    _validate_max_concurrency(max_concurrency)

    # Use smaller sample size if dataset is smaller
    actual_sample_size = min(sample_size, len(dataset))

    logger.info(
        f"Starting dry run for flow '{flow.metadata.name}' "
        f"with {actual_sample_size} samples"
    )

    # Create subset dataset
    sample_dataset = dataset.iloc[:actual_sample_size]

    # Initialize dry run results
    dry_run_results = {
        "flow_name": flow.metadata.name,
        "flow_version": flow.metadata.version,
        "sample_size": actual_sample_size,
        "original_dataset_size": len(dataset),
        "max_concurrency": max_concurrency,
        "input_columns": dataset.columns.tolist(),
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

        for i, block in enumerate(flow.blocks):
            block_start_time = time.perf_counter()
            input_rows = len(current_dataset)

            logger.info(
                f"Dry run executing block {i + 1}/{len(flow.blocks)}: "
                f"{block.block_name} ({block.__class__.__name__})"
            )

            # Prepare block execution parameters
            block_kwargs = prepare_block_kwargs(block, runtime_params)

            # Add max_concurrency to block kwargs if provided
            if max_concurrency is not None:
                block_kwargs["_flow_max_concurrency"] = max_concurrency

            # Execute block with validation and logging
            current_dataset = block(current_dataset, **block_kwargs)

            block_execution_time = (
                time.perf_counter() - block_start_time
            )  # Fixed: use perf_counter consistently

            # Record block execution info
            block_info = {
                "block_name": block.block_name,
                "block_class": block.__class__.__name__,
                "execution_time_seconds": block_execution_time,
                "input_rows": input_rows,
                "output_rows": len(current_dataset),
                "output_columns": current_dataset.columns.tolist(),
                "parameters_used": block_kwargs,
            }

            dry_run_results["blocks_executed"].append(block_info)

            logger.info(
                f"Dry run block '{block.block_name}' completed: "
                f"{len(current_dataset)} samples, "
                f"{len(current_dataset.columns)} columns, "
                f"{block_execution_time:.2f}s"
            )

        # Store final results
        dry_run_results["final_dataset"] = {
            "rows": len(current_dataset),
            "columns": current_dataset.columns.tolist(),
            "sample_data": current_dataset.to_dict()
            if len(current_dataset) > 0
            else {},
        }

        execution_time = time.perf_counter() - start_time
        dry_run_results["execution_time_seconds"] = execution_time

        logger.info(
            f"Dry run completed successfully for flow '{flow.metadata.name}' "
            f"in {execution_time:.2f}s"
        )

        # Perform time estimation if requested (displays table but doesn't store in results)
        if enable_time_estimation:
            estimate_total_time(
                flow, dry_run_results, dataset, runtime_params, max_concurrency
            )

        return dry_run_results

    except (EmptyDatasetError, FlowValidationError):
        # Re-raise these errors directly without wrapping
        raise
    except Exception as exc:
        execution_time = time.perf_counter() - start_time
        dry_run_results["execution_successful"] = False
        dry_run_results["execution_time_seconds"] = execution_time
        dry_run_results["error"] = str(exc)

        logger.error(f"Dry run failed for flow '{flow.metadata.name}': {exc}")

        raise FlowValidationError(f"Dry run failed: {exc}") from exc


def estimate_total_time(
    flow: "Flow",
    first_run_results: dict[str, Any],
    dataset: pd.DataFrame,
    runtime_params: Optional[dict[str, dict[str, Any]]],
    max_concurrency: Optional[int],
) -> dict[str, Any]:
    """Estimate execution time using 2 dry runs.

    This function contains all the estimation logic. It determines if a second
    dry run is needed, executes it, and calls estimate_execution_time.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    first_run_results : dict
        Results from the first dry run.
    dataset : pd.DataFrame
        Full dataset for estimation.
    runtime_params : Optional[dict]
        Runtime parameters.
    max_concurrency : Optional[int]
        Maximum concurrency.

    Returns
    -------
    dict
        Estimation results with estimated_time_seconds, total_estimated_requests, etc.
    """
    first_sample_size = first_run_results["sample_size"]

    # Check if we need a second dry run
    has_async_blocks = any(getattr(block, "async_mode", False) for block in flow.blocks)

    # For sequential or no async blocks, single run is sufficient
    if max_concurrency == 1 or not has_async_blocks:
        estimation = estimate_execution_time(
            dry_run_1=first_run_results,
            dry_run_2=None,
            total_dataset_size=len(dataset),
            max_concurrency=max_concurrency,
        )
    else:
        # Need second measurement - always use canonical (1, 5) pair
        if first_sample_size == 1:
            # Already have 1, need 5
            logger.info("Running second dry run with 5 samples for time estimation")
            second_run = run_dry_run(
                flow,
                dataset,
                5,
                runtime_params,
                max_concurrency,
                enable_time_estimation=False,
            )
            dry_run_1, dry_run_2 = first_run_results, second_run
        elif first_sample_size == 5:
            # Already have 5, need 1
            logger.info("Running second dry run with 1 sample for time estimation")
            second_run = run_dry_run(
                flow,
                dataset,
                1,
                runtime_params,
                max_concurrency,
                enable_time_estimation=False,
            )
            dry_run_1, dry_run_2 = second_run, first_run_results
        else:
            # For other sizes: run both 1 and 5 for canonical pair
            logger.info("Running dry runs with 1 and 5 samples for time estimation")
            dry_run_1 = run_dry_run(
                flow,
                dataset,
                1,
                runtime_params,
                max_concurrency,
                enable_time_estimation=False,
            )
            dry_run_2 = run_dry_run(
                flow,
                dataset,
                5,
                runtime_params,
                max_concurrency,
                enable_time_estimation=False,
            )

        estimation = estimate_execution_time(
            dry_run_1=dry_run_1,
            dry_run_2=dry_run_2,
            total_dataset_size=len(dataset),
            max_concurrency=max_concurrency,
        )

    # Display estimation summary
    display_time_estimation_summary(estimation, len(dataset), max_concurrency)

    return estimation
