# SPDX-License-Identifier: Apache-2.0
"""Flow execution helper functions for Flow class."""

# Standard
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union
import logging
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


def _close_flow_logger(
    flow_logger: logging.Logger, module_logger: logging.Logger
) -> None:
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
            except (OSError, ValueError):
                # Ignore errors during cleanup - handler may already be closed
                # (ValueError) or the underlying stream may be in an invalid
                # state (OSError). We still want to remove it.
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


def _record_block_success_metrics(
    flow: "Flow",
    block: BaseBlock,
    execution_time: float,
    input_rows: int,
    input_cols: set[str],
    current_dataset: pd.DataFrame,
    exec_logger: logging.Logger,
) -> None:
    """Record metrics for a successfully executed block.

    Parameters
    ----------
    flow : Flow
        The flow instance (metrics are appended to flow._block_metrics).
    block : BaseBlock
        The block that was executed.
    execution_time : float
        Wall-clock time the block took to execute.
    input_rows : int
        Number of rows in the input dataset.
    input_cols : set[str]
        Column names in the input dataset.
    current_dataset : pd.DataFrame
        The dataset produced by the block.
    exec_logger : logging.Logger
        Logger for this execution.
    """
    output_cols = set(current_dataset.columns)
    flow._block_metrics.append(
        {
            "block_name": block.block_name,
            "block_class": block.__class__.__name__,
            "execution_time": execution_time,
            "input_rows": input_rows,
            "output_rows": len(current_dataset),
            "added_cols": list(output_cols - input_cols),
            "removed_cols": list(input_cols - output_cols),
            "status": "success",
        }
    )
    exec_logger.info(
        f"Block '{block.block_name}' completed successfully: "
        f"{len(current_dataset)} samples, "
        f"{len(current_dataset.columns)} columns"
    )


def _record_block_failure_metrics(
    flow: "Flow",
    block: BaseBlock,
    execution_time: float,
    input_rows: int,
    exc: Exception,
    exec_logger: logging.Logger,
) -> None:
    """Record metrics for a failed block and re-raise as FlowValidationError.

    Parameters
    ----------
    flow : Flow
        The flow instance (metrics are appended to flow._block_metrics).
    block : BaseBlock
        The block that failed.
    execution_time : float
        Wall-clock time before the failure.
    input_rows : int
        Number of rows in the input dataset.
    exc : Exception
        The exception that occurred.
    exec_logger : logging.Logger
        Logger for this execution.

    Raises
    ------
    FlowValidationError
        Always raised, wrapping the original exception.
    """
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
    exec_logger.error(f"Block '{block.block_name}' failed during execution: {exc}")
    raise FlowValidationError(
        f"Block '{block.block_name}' execution failed: {exc}"
    ) from exc


def _execute_single_block(
    flow: "Flow",
    block: BaseBlock,
    block_index: int,
    current_dataset: pd.DataFrame,
    runtime_params: dict[str, dict[str, Any]],
    exec_logger: logging.Logger,
    max_concurrency: Optional[int] = None,
    column_tracker: Optional[ColumnDependencyTracker] = None,
) -> pd.DataFrame:
    """Execute a single block on the dataset and record metrics.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    block : BaseBlock
        The block to execute.
    block_index : int
        Index of the block in the flow (0-based).
    current_dataset : pd.DataFrame
        Dataset to pass to the block.
    runtime_params : dict[str, dict[str, Any]]
        Runtime parameters for block execution.
    exec_logger : logging.Logger
        Logger for this execution.
    max_concurrency : Optional[int], optional
        Maximum concurrency for LLM requests.
    column_tracker : Optional[ColumnDependencyTracker], optional
        Tracker for early dropping of unused columns.

    Returns
    -------
    pd.DataFrame
        Dataset after processing through the block (and optional column dropping).
    """
    exec_logger.info(
        f"Executing block {block_index + 1}/{len(flow.blocks)}: "
        f"{block.block_name} ({block.__class__.__name__})"
    )

    block_kwargs = prepare_block_kwargs(block, runtime_params)
    if max_concurrency is not None:
        block_kwargs["_flow_max_concurrency"] = max_concurrency

    start_time = time.perf_counter()
    input_rows = len(current_dataset)
    input_cols = set(current_dataset.columns)

    try:
        current_dataset = block(current_dataset, **block_kwargs)

        if len(current_dataset) == 0:
            raise EmptyDatasetError(block.block_name)

        execution_time = time.perf_counter() - start_time
        _record_block_success_metrics(
            flow,
            block,
            execution_time,
            input_rows,
            input_cols,
            current_dataset,
            exec_logger,
        )
    except EmptyDatasetError:
        raise
    except Exception as exc:
        execution_time = time.perf_counter() - start_time
        _record_block_failure_metrics(
            flow,
            block,
            execution_time,
            input_rows,
            exc,
            exec_logger,
        )

    # Drop columns no longer needed by downstream blocks (outside try/except
    # so failures here are not misattributed as block execution errors)
    if column_tracker is not None:
        cols_to_drop = column_tracker.get_droppable_columns(
            block_index, set(current_dataset.columns)
        )
        if cols_to_drop:
            exec_logger.info(
                f"Dropping {len(cols_to_drop)} unused columns: {sorted(cols_to_drop)}"
            )
            current_dataset = current_dataset.drop(columns=cols_to_drop)

    return current_dataset


def execute_blocks_on_dataset(
    flow: "Flow",
    dataset: pd.DataFrame,
    runtime_params: dict[str, dict[str, Any]],
    flow_logger: Optional[logging.Logger] = None,
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
    exec_logger = flow_logger if flow_logger is not None else logger
    current_dataset = dataset

    for i, block in enumerate(flow.blocks):
        current_dataset = _execute_single_block(
            flow,
            block,
            i,
            current_dataset,
            runtime_params,
            exec_logger,
            max_concurrency,
            column_tracker,
        )

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


def _setup_flow_logger(
    log_dir: Optional[str],
    flow_metadata_name: str,
) -> tuple[logging.Logger, Optional[str], Optional[str]]:
    """Create a flow-specific logger if log_dir is provided.

    Parameters
    ----------
    log_dir : Optional[str]
        Directory to save execution logs. If None, returns the module logger.
    flow_metadata_name : str
        Name from flow metadata, used to build the log filename.

    Returns
    -------
    tuple[logging.Logger, Optional[str], Optional[str]]
        (flow_logger, timestamp, flow_name).  timestamp and flow_name are None
        when log_dir is None.
    """
    if log_dir is None:
        return logger, None, None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flow_name = flow_metadata_name.replace(" ", "_").lower()
    log_filename = f"{flow_name}_{timestamp}.log"

    unique_id = str(uuid.uuid4())[:8]
    flow_logger_name = f"{__name__}.flow_{flow_name}_{timestamp}_{unique_id}"
    flow_logger = setup_logger(
        flow_logger_name, log_dir=log_dir, log_filename=log_filename
    )
    flow_logger.propagate = False
    flow_logger.info(
        f"Flow logging enabled - logs will be saved to: {log_dir}/{log_filename}"
    )
    return flow_logger, timestamp, flow_name


def _validate_flow_preconditions(
    flow: "Flow",
    dataset: pd.DataFrame,
    save_freq: Optional[int],
    max_concurrency: Optional[int],
) -> None:
    """Validate all preconditions before flow execution.

    Checks blocks, dataset, model config, agent config, and dataset requirements.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    dataset : pd.DataFrame
        Input dataset.
    save_freq : Optional[int]
        Checkpoint save frequency.
    max_concurrency : Optional[int]
        Maximum concurrency for LLM requests.

    Raises
    ------
    FlowValidationError
        If any precondition is not met.
    """
    # Import here to avoid circular imports
    from .agent_config import detect_agent_blocks
    from .model_config import detect_llm_blocks

    if save_freq is not None and save_freq <= 0:
        raise FlowValidationError(f"save_freq must be greater than 0, got {save_freq}")

    _validate_max_concurrency(max_concurrency)

    if not flow.blocks:
        raise FlowValidationError("Cannot generate with empty flow")

    if len(dataset) == 0:
        raise FlowValidationError("Input dataset is empty")

    validate_no_duplicates(dataset)

    llm_blocks = detect_llm_blocks(flow)
    if llm_blocks and not flow._model_config_set:
        raise FlowValidationError(
            f"Model configuration required before generate(). "
            f"Found {len(llm_blocks)} LLM blocks: {sorted(llm_blocks)}. "
            f"Call flow.set_model_config() first."
        )

    agent_blocks = detect_agent_blocks(flow)
    if agent_blocks and not flow._agent_config_set:
        raise FlowValidationError(
            f"Agent configuration required before generate(). "
            f"Found {len(agent_blocks)} agent blocks: {sorted(agent_blocks)}. "
            f"Call flow.set_agent_config() first."
        )

    dataset_errors = validate_flow_dataset(flow, dataset)
    if dataset_errors:
        raise FlowValidationError(
            "Dataset validation failed:\n" + "\n".join(dataset_errors)
        )


def _initialize_checkpointer(
    flow: "Flow",
    dataset: pd.DataFrame,
    checkpoint_dir: Optional[str],
    save_freq: Optional[int],
    effective_output_columns: Optional[set[str]],
    original_columns: set[str],
    was_dataset: bool,
    flow_logger: logging.Logger,
    log_dir: Optional[str],
) -> tuple[
    Optional[FlowCheckpointer],
    Optional[pd.DataFrame],
    pd.DataFrame,
    Optional[Union[pd.DataFrame, datasets.Dataset]],
]:
    """Set up checkpointer and load existing progress if available.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    dataset : pd.DataFrame
        Input dataset.
    checkpoint_dir : Optional[str]
        Directory to save/load checkpoints.
    save_freq : Optional[int]
        Save frequency for checkpoints.
    effective_output_columns : Optional[set[str]]
        Output columns for cleanup.
    original_columns : set[str]
        Original input columns.
    was_dataset : bool
        Whether input was a datasets.Dataset.
    flow_logger : logging.Logger
        Logger for this execution.
    log_dir : Optional[str]
        Log directory (used to decide if logger needs closing).

    Returns
    -------
    tuple
        (checkpointer, completed_dataset, remaining_dataset, early_return).
        If early_return is not None, the caller should return it immediately.
    """
    if not checkpoint_dir:
        return None, None, dataset, None

    checkpointer = FlowCheckpointer(
        checkpoint_dir=checkpoint_dir,
        save_freq=save_freq,
        flow_id=flow.metadata.id,
    )

    remaining_dataset, completed_dataset = checkpointer.load_existing_progress(dataset)

    if len(remaining_dataset) == 0:
        flow_logger.info("All samples already completed, returning existing results")

        if effective_output_columns is not None:
            completed_dataset = _cleanup_final_columns(
                completed_dataset,
                effective_output_columns,
                original_columns,
                flow_logger,
            )

        if log_dir is not None:
            _close_flow_logger(flow_logger, logger)

        return (
            checkpointer,
            completed_dataset,
            remaining_dataset,
            convert_from_dataframe(completed_dataset, was_dataset),
        )

    flow_logger.info(f"Resuming with {len(remaining_dataset)} remaining samples")
    return checkpointer, completed_dataset, remaining_dataset, None


def _process_with_checkpointing(
    flow: "Flow",
    dataset: pd.DataFrame,
    checkpointer: FlowCheckpointer,
    save_freq: int,
    completed_dataset: Optional[pd.DataFrame],
    runtime_params: dict[str, dict[str, Any]],
    flow_logger: logging.Logger,
    max_concurrency: Optional[int],
    column_tracker: Optional[ColumnDependencyTracker],
) -> pd.DataFrame:
    """Process dataset in chunks with checkpoint saves.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    dataset : pd.DataFrame
        Dataset to process.
    checkpointer : FlowCheckpointer
        Active checkpointer.
    save_freq : int
        Number of samples per checkpoint chunk.
    completed_dataset : Optional[pd.DataFrame]
        Previously completed samples (may be None or empty).
    runtime_params : dict[str, dict[str, Any]]
        Runtime parameters for block execution.
    flow_logger : logging.Logger
        Logger for this execution.
    max_concurrency : Optional[int]
        Maximum concurrency for LLM requests.
    column_tracker : Optional[ColumnDependencyTracker]
        Tracker for early dropping of unused columns.

    Returns
    -------
    pd.DataFrame
        Combined dataset of all processed chunks and any prior completed data.
    """
    all_processed = []

    for i in range(0, len(dataset), save_freq):
        chunk_end = min(i + save_freq, len(dataset))
        chunk_dataset = dataset.iloc[i:chunk_end]

        flow_logger.info(
            f"Processing chunk {i // save_freq + 1}: samples {i} to {chunk_end - 1}"
        )

        processed_chunk = execute_blocks_on_dataset(
            flow,
            chunk_dataset,
            runtime_params,
            flow_logger,
            max_concurrency,
            column_tracker,
        )
        all_processed.append(processed_chunk)
        checkpointer.add_completed_samples(processed_chunk)

    checkpointer.save_final_checkpoint()

    final_dataset = safe_concatenate_with_validation(
        all_processed, "processed chunks from flow execution"
    )

    if completed_dataset is not None and not completed_dataset.empty:
        final_dataset = safe_concatenate_with_validation(
            [completed_dataset, final_dataset],
            "completed checkpoint data with newly processed data",
        )

    return final_dataset


def _process_without_checkpointing(
    flow: "Flow",
    dataset: pd.DataFrame,
    checkpointer: Optional[FlowCheckpointer],
    completed_dataset: Optional[pd.DataFrame],
    runtime_params: dict[str, dict[str, Any]],
    flow_logger: logging.Logger,
    max_concurrency: Optional[int],
    column_tracker: Optional[ColumnDependencyTracker],
) -> pd.DataFrame:
    """Process entire dataset at once, optionally saving a final checkpoint.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    dataset : pd.DataFrame
        Dataset to process.
    checkpointer : Optional[FlowCheckpointer]
        Checkpointer (may be None if checkpointing is disabled).
    completed_dataset : Optional[pd.DataFrame]
        Previously completed samples.
    runtime_params : dict[str, dict[str, Any]]
        Runtime parameters for block execution.
    flow_logger : logging.Logger
        Logger for this execution.
    max_concurrency : Optional[int]
        Maximum concurrency for LLM requests.
    column_tracker : Optional[ColumnDependencyTracker]
        Tracker for early dropping of unused columns.

    Returns
    -------
    pd.DataFrame
        Processed dataset, combined with any prior completed data.
    """
    final_dataset = execute_blocks_on_dataset(
        flow,
        dataset,
        runtime_params,
        flow_logger,
        max_concurrency,
        column_tracker,
    )

    if checkpointer:
        checkpointer.add_completed_samples(final_dataset)
        checkpointer.save_final_checkpoint()

        if completed_dataset is not None and not completed_dataset.empty:
            final_dataset = safe_concatenate_with_validation(
                [completed_dataset, final_dataset],
                "completed checkpoint data with newly processed data",
            )

    return final_dataset


def _finalize_flow_metrics(
    flow: "Flow",
    final_dataset: Optional[pd.DataFrame],
    execution_successful: bool,
    run_start: float,
    log_dir: Optional[str],
    timestamp: Optional[str],
    flow_name: Optional[str],
    flow_logger: logging.Logger,
) -> None:
    """Display and persist flow metrics, then close the flow logger.

    This function is intended to be called in a ``finally`` block so that
    metrics are always emitted regardless of success or failure.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    final_dataset : Optional[pd.DataFrame]
        The final dataset (may be None on failure).
    execution_successful : bool
        Whether the execution completed without error.
    run_start : float
        Start time from ``time.perf_counter()``.
    log_dir : Optional[str]
        Log directory.
    timestamp : Optional[str]
        Timestamp string for the log file.
    flow_name : Optional[str]
        Sanitised flow name for the log file.
    flow_logger : logging.Logger
        Logger for this execution.
    """
    display_metrics_summary(flow._block_metrics, flow.metadata.name, final_dataset)

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
        _close_flow_logger(flow_logger, logger)


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
    dataset, was_dataset = convert_to_dataframe(dataset)

    original_columns = set(dataset.columns)
    effective_output_columns = (
        set(flow.metadata.output_columns) if flow.metadata.output_columns else None
    )
    runtime_params = runtime_params or {}

    _validate_flow_preconditions(flow, dataset, save_freq, max_concurrency)

    flow_logger, timestamp, flow_name = _setup_flow_logger(
        log_dir,
        flow.metadata.name,
    )

    checkpointer, completed_dataset, dataset, early_return = _initialize_checkpointer(
        flow,
        dataset,
        checkpoint_dir,
        save_freq,
        effective_output_columns,
        original_columns,
        was_dataset,
        flow_logger,
        log_dir,
    )
    if early_return is not None:
        return early_return

    if max_concurrency is not None:
        flow_logger.info(f"Using max_concurrency={max_concurrency} for LLM requests")

    flow_logger.info(
        f"Starting flow '{flow.metadata.name}' v{flow.metadata.version} "
        f"with {len(dataset)} samples across {len(flow.blocks)} blocks"
        + (f" (max_concurrency={max_concurrency})" if max_concurrency else "")
    )

    flow._block_metrics = []
    run_start = time.perf_counter()

    column_tracker: Optional[ColumnDependencyTracker] = None
    if effective_output_columns is not None:
        column_tracker = ColumnDependencyTracker(
            flow.blocks, effective_output_columns, original_columns
        )
        flow_logger.info(
            "Column cleanup enabled - will drop unused intermediate columns"
        )

    final_dataset = None
    execution_successful = False

    try:
        if checkpointer and save_freq:
            final_dataset = _process_with_checkpointing(
                flow,
                dataset,
                checkpointer,
                save_freq,
                completed_dataset,
                runtime_params,
                flow_logger,
                max_concurrency,
                column_tracker,
            )
        else:
            final_dataset = _process_without_checkpointing(
                flow,
                dataset,
                checkpointer,
                completed_dataset,
                runtime_params,
                flow_logger,
                max_concurrency,
                column_tracker,
            )

        execution_successful = True

        if effective_output_columns is not None and final_dataset is not None:
            final_dataset = _cleanup_final_columns(
                final_dataset,
                effective_output_columns,
                original_columns,
                flow_logger,
            )

    finally:
        _finalize_flow_metrics(
            flow,
            final_dataset,
            execution_successful,
            run_start,
            log_dir,
            timestamp,
            flow_name,
            flow_logger,
        )

    if execution_successful and final_dataset is not None:
        logger.info(
            f"Flow '{flow.metadata.name}' completed successfully: "
            f"{len(final_dataset)} final samples, "
            f"{len(final_dataset.columns)} final columns"
        )

    return convert_from_dataframe(final_dataset, was_dataset)


def _validate_dry_run_preconditions(
    flow: "Flow",
    dataset: pd.DataFrame,
    max_concurrency: Optional[int],
) -> None:
    """Validate preconditions for a dry run.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    dataset : pd.DataFrame
        Input dataset.
    max_concurrency : Optional[int]
        Maximum concurrency value.

    Raises
    ------
    FlowValidationError
        If preconditions are not met.
    """
    if not flow.blocks:
        raise FlowValidationError("Cannot dry run empty flow")

    if len(dataset) == 0:
        raise FlowValidationError("Input dataset is empty")

    validate_no_duplicates(dataset)
    _validate_max_concurrency(max_concurrency)


def _execute_dry_run_blocks(
    flow: "Flow",
    sample_dataset: pd.DataFrame,
    runtime_params: dict[str, dict[str, Any]],
    max_concurrency: Optional[int],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Execute all blocks on the sample dataset and collect per-block info.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    sample_dataset : pd.DataFrame
        Subset of the dataset to run through the blocks.
    runtime_params : dict[str, dict[str, Any]]
        Runtime parameters for block execution.
    max_concurrency : Optional[int]
        Maximum concurrency for LLM requests.

    Returns
    -------
    tuple[pd.DataFrame, list[dict[str, Any]]]
        (final_dataset, blocks_executed_info).
    """
    current_dataset = sample_dataset
    blocks_executed: list[dict[str, Any]] = []

    for i, block in enumerate(flow.blocks):
        block_start_time = time.perf_counter()
        input_rows = len(current_dataset)

        logger.info(
            f"Dry run executing block {i + 1}/{len(flow.blocks)}: "
            f"{block.block_name} ({block.__class__.__name__})"
        )

        block_kwargs = prepare_block_kwargs(block, runtime_params)
        if max_concurrency is not None:
            block_kwargs["_flow_max_concurrency"] = max_concurrency

        current_dataset = block(current_dataset, **block_kwargs)

        block_execution_time = time.perf_counter() - block_start_time

        blocks_executed.append(
            {
                "block_name": block.block_name,
                "block_class": block.__class__.__name__,
                "execution_time_seconds": block_execution_time,
                "input_rows": input_rows,
                "output_rows": len(current_dataset),
                "output_columns": current_dataset.columns.tolist(),
                "parameters_used": block_kwargs,
            }
        )

        logger.info(
            f"Dry run block '{block.block_name}' completed: "
            f"{len(current_dataset)} samples, "
            f"{len(current_dataset.columns)} columns, "
            f"{block_execution_time:.2f}s"
        )

    return current_dataset, blocks_executed


def _build_dry_run_results(
    flow: "Flow",
    actual_sample_size: int,
    original_dataset_size: int,
    max_concurrency: Optional[int],
    input_columns: list[str],
) -> dict[str, Any]:
    """Create the initial dry-run results dict.

    Parameters
    ----------
    flow : Flow
        The flow instance.
    actual_sample_size : int
        Number of samples actually used.
    original_dataset_size : int
        Size of the full input dataset.
    max_concurrency : Optional[int]
        Maximum concurrency value.
    input_columns : list[str]
        Column names from the input dataset.

    Returns
    -------
    dict[str, Any]
        Initialised dry-run results dictionary.
    """
    return {
        "flow_name": flow.metadata.name,
        "flow_version": flow.metadata.version,
        "sample_size": actual_sample_size,
        "original_dataset_size": original_dataset_size,
        "max_concurrency": max_concurrency,
        "input_columns": input_columns,
        "blocks_executed": [],
        "final_dataset": None,
        "execution_successful": True,
        "execution_time_seconds": 0,
    }


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
    dataset, _ = convert_to_dataframe(dataset)
    _validate_dry_run_preconditions(flow, dataset, max_concurrency)

    actual_sample_size = min(sample_size, len(dataset))
    runtime_params = runtime_params or {}

    logger.info(
        f"Starting dry run for flow '{flow.metadata.name}' "
        f"with {actual_sample_size} samples"
    )

    sample_dataset = dataset.iloc[:actual_sample_size]
    dry_run_results = _build_dry_run_results(
        flow,
        actual_sample_size,
        len(dataset),
        max_concurrency,
        dataset.columns.tolist(),
    )

    start_time = time.perf_counter()

    try:
        current_dataset, blocks_executed = _execute_dry_run_blocks(
            flow,
            sample_dataset,
            runtime_params,
            max_concurrency,
        )

        dry_run_results["blocks_executed"] = blocks_executed
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

        if enable_time_estimation:
            estimate_total_time(
                flow, dry_run_results, dataset, runtime_params, max_concurrency
            )

        return dry_run_results

    except (EmptyDatasetError, FlowValidationError):
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
