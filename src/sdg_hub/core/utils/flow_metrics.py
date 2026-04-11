# SPDX-License-Identifier: Apache-2.0
"""Flow execution metrics utilities for display and export."""

# Standard
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Third Party
import pandas as pd


def aggregate_block_metrics(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-block metrics, coalescing chunked runs.

    Parameters
    ----------
    entries : list[dict[str, Any]]
        Raw block metrics entries from flow execution.

    Returns
    -------
    list[dict[str, Any]]
        Aggregated metrics with combined execution times and data changes.
    """
    agg: dict[tuple[Optional[str], Optional[str]], dict[str, Any]] = {}
    for m in entries:
        key: tuple[Optional[str], Optional[str]] = (
            m.get("block_name"),
            m.get("block_class"),
        )
        a = agg.setdefault(
            key,
            {
                "block_name": key[0],
                "block_class": key[1],
                "execution_time": 0.0,
                "input_rows": 0,
                "output_rows": 0,
                "added_cols": set(),
                "removed_cols": set(),
                "status": "success",
                "error_type": None,
                "error": None,
            },
        )
        a["execution_time"] += float(m.get("execution_time", 0.0))
        a["input_rows"] += int(m.get("input_rows", 0))
        a["output_rows"] += int(m.get("output_rows", 0))
        a["added_cols"].update(m.get("added_cols", []))
        a["removed_cols"].update(m.get("removed_cols", []))
        if m.get("status") == "failed":
            a["status"] = "failed"
            a["error_type"] = m.get("error_type") or a["error_type"]
            a["error"] = m.get("error") or a["error"]
    # normalize
    result = []
    for a in agg.values():
        a["added_cols"] = sorted(a["added_cols"])
        a["removed_cols"] = sorted(a["removed_cols"])
        # drop empty error fields
        if a["status"] == "success":
            a.pop("error_type", None)
            a.pop("error", None)
        result.append(a)
    return result


def _format_block_row(metrics: dict[str, Any]) -> tuple[str, str, str, str]:
    """Format a single block's metrics into table cell values.

    Parameters
    ----------
    metrics : dict[str, Any]
        Block metrics dict with execution_time, status, input_rows, etc.

    Returns
    -------
    tuple[str, str, str, str]
        (duration, row_change, col_change, status) formatted for the table.
    """
    duration = f"{metrics['execution_time']:.2f}s"

    if metrics["status"] == "success":
        row_change = f"{metrics['input_rows']:,} → {metrics['output_rows']:,}"
    else:
        row_change = f"{metrics['input_rows']:,} → ❌"

    added = len(metrics["added_cols"])
    removed = len(metrics["removed_cols"])
    if added > 0 and removed > 0:
        col_change = f"+{added}/-{removed}"
    elif added > 0:
        col_change = f"+{added}"
    elif removed > 0:
        col_change = f"-{removed}"
    else:
        col_change = "—"

    status = "[green]✓[/green]" if metrics["status"] == "success" else "[red]✗[/red]"

    return duration, row_change, col_change, status


def _resolve_panel_style(
    flow_name: str,
    failed_blocks: int,
    final_dataset: Optional[pd.DataFrame],
) -> tuple[str, str]:
    """Determine the panel title and border style based on execution outcome.

    Parameters
    ----------
    flow_name : str
        Name of the flow.
    failed_blocks : int
        Number of blocks that failed.
    final_dataset : Optional[pd.DataFrame]
        Final dataset, or None if the flow failed entirely.

    Returns
    -------
    tuple[str, str]
        (title, border_style) for the Rich Panel.
    """
    if final_dataset is None:
        label, color, border = "Failed", "red", "bright_red"
    elif failed_blocks == 0:
        label, color, border = "Complete", "green", "bright_green"
    else:
        label, color, border = "Partial", "yellow", "bright_yellow"

    title = f"[bold bright_white]{flow_name}[/bold bright_white] - [{color}]{label}[/{color}]"
    return title, border


def display_metrics_summary(
    block_metrics: list[dict[str, Any]],
    flow_name: str,
    final_dataset: Optional[pd.DataFrame] = None,
) -> None:
    """Display a rich table summarizing block execution metrics.

    Parameters
    ----------
    block_metrics : list[dict[str, Any]]
        Raw block metrics from flow execution.
    flow_name : str
        Name of the flow for display title.
    final_dataset : Optional[pd.DataFrame], optional
        Final dataset from flow execution. None if flow failed.
    """
    if not block_metrics:
        return

    console = Console()

    # Create the metrics table
    table = Table(
        show_header=True,
        header_style="bold bright_white",
        title="Flow Execution Summary",
    )
    table.add_column("Block Name", style="bright_cyan", width=20)
    table.add_column("Type", style="bright_green", width=15)
    table.add_column("Duration", justify="right", style="bright_yellow", width=10)
    table.add_column("Rows", justify="center", style="bright_blue", width=12)
    table.add_column("Columns", justify="center", style="bright_magenta", width=15)
    table.add_column("Status", justify="center", width=10)

    total_time = 0.0
    successful_blocks = 0

    for metrics in block_metrics:
        total_time += metrics["execution_time"]
        if metrics["status"] == "success":
            successful_blocks += 1

        duration, row_change, col_change, status = _format_block_row(metrics)
        table.add_row(
            metrics["block_name"],
            metrics["block_class"],
            duration,
            row_change,
            col_change,
            status,
        )

    # Add summary row
    table.add_section()
    final_row_count = len(final_dataset) if final_dataset is not None else 0
    final_col_count = (
        len(final_dataset.columns.tolist()) if final_dataset is not None else 0
    )

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{len(block_metrics)} blocks[/bold]",
        f"[bold]{total_time:.2f}s[/bold]",
        f"[bold]{final_row_count:,} final[/bold]",
        f"[bold]{final_col_count} final[/bold]",
        f"[bold][green]{successful_blocks}/{len(block_metrics)}[/green][/bold]",
    )

    # Display the table with panel
    console.print()

    failed_blocks = len(block_metrics) - successful_blocks
    title, border_style = _resolve_panel_style(flow_name, failed_blocks, final_dataset)

    console.print(
        Panel(
            table,
            title=title,
            border_style=border_style,
        )
    )
    console.print()


def _format_time_str(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted string like "45.5 seconds", "30.0 minutes (0.50 hours)", etc.
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.1f} minutes ({seconds / 3600:.2f} hours)"
    return f"{seconds / 3600:.2f} hours ({seconds / 60:.0f} minutes)"


def _build_summary_table(
    time_estimation: dict[str, Any],
    dataset_size: int,
    max_concurrency: Optional[int],
) -> Table:
    """Build the top-level summary table for time estimation.

    Parameters
    ----------
    time_estimation : dict[str, Any]
        Time estimation results.
    dataset_size : int
        Total number of samples.
    max_concurrency : Optional[int]
        Max concurrency value, or None.

    Returns
    -------
    Table
        A Rich Table with summary rows.
    """
    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_column("Metric", style="bright_cyan")
    summary_table.add_column("Value", style="bright_white")

    summary_table.add_row(
        "Estimated Time:",
        _format_time_str(time_estimation["estimated_time_seconds"]),
    )
    summary_table.add_row(
        "Total LLM Requests:",
        f"{time_estimation.get('total_estimated_requests', 0):,}",
    )

    if time_estimation.get("total_estimated_requests", 0) > 0:
        requests_per_sample = time_estimation["total_estimated_requests"] / dataset_size
        summary_table.add_row("Requests per Sample:", f"{requests_per_sample:.1f}")

    if max_concurrency is not None:
        summary_table.add_row("Max Concurrency:", str(max_concurrency))

    return summary_table


def _build_block_breakdown_table(block_estimates: list[dict[str, Any]]) -> Table:
    """Build the per-block breakdown table for time estimation.

    Parameters
    ----------
    block_estimates : list[dict[str, Any]]
        Per-block estimation dicts.

    Returns
    -------
    Table
        A Rich Table with one row per block.
    """
    block_table = Table(show_header=True, header_style="bold bright_white")
    block_table.add_column("Block Name", style="bright_cyan", width=20)
    block_table.add_column("Time", justify="right", style="bright_yellow", width=10)
    block_table.add_column("Requests", justify="right", style="bright_green", width=10)
    block_table.add_column("Throughput", justify="right", style="bright_blue", width=12)
    block_table.add_column("Amplif.", justify="right", style="bright_magenta", width=10)

    for block in block_estimates:
        block_seconds = block["estimated_time"]
        time_str = (
            f"{block_seconds:.1f}s"
            if block_seconds < 60
            else f"{block_seconds / 60:.1f}min"
        )
        block_table.add_row(
            block["block"],
            time_str,
            f"{block['estimated_requests']:,.0f}",
            f"{block['throughput']:.2f}/s",
            f"{block['amplification']:.1f}x",
        )

    return block_table


def display_time_estimation_summary(
    time_estimation: dict[str, Any],
    dataset_size: int,
    max_concurrency: Optional[int] = None,
) -> None:
    """Display a rich table summarizing time estimation results.

    Parameters
    ----------
    time_estimation : dict[str, Any]
        Time estimation results from estimate_total_time().
    dataset_size : int
        Total number of samples in the dataset.
    max_concurrency : Optional[int], optional
        Maximum concurrency used for estimation.
    """
    console = Console()

    summary_table = _build_summary_table(time_estimation, dataset_size, max_concurrency)

    console.print()
    console.print(
        Panel(
            summary_table,
            title=f"[bold bright_white]Time Estimation for {dataset_size:,} Samples[/bold bright_white]",
            border_style="bright_blue",
        )
    )

    block_estimates = time_estimation.get("block_estimates", [])
    if block_estimates:
        console.print()
        block_table = _build_block_breakdown_table(block_estimates)
        console.print(
            Panel(
                block_table,
                title="[bold bright_white]Per-Block Breakdown[/bold bright_white]",
                border_style="bright_blue",
            )
        )

    console.print()


def save_metrics_to_json(
    block_metrics: list[dict[str, Any]],
    flow_name: str,
    flow_version: str,
    execution_successful: bool,
    run_start_time: float,
    log_dir: str,
    timestamp: Optional[str] = None,
    flow_name_normalized: Optional[str] = None,
    logger=None,
) -> None:
    """Save flow execution metrics to JSON file.

    Parameters
    ----------
    block_metrics : list[dict[str, Any]]
        Raw block metrics from flow execution.
    flow_name : str
        Human-readable flow name.
    flow_version : str
        Flow version string.
    execution_successful : bool
        Whether the flow execution completed successfully.
    run_start_time : float
        Start time from time.perf_counter() for wall time calculation.
    log_dir : str
        Directory to save metrics JSON file.
    timestamp : Optional[str], optional
        Timestamp string for filename. Generated if not provided.
    flow_name_normalized : Optional[str], optional
        Normalized flow name for filename. Generated if not provided.
    logger : Optional[logging.Logger], optional
        Logger instance for status messages.
    """
    try:
        # Generate timestamp and normalized name if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if flow_name_normalized is None:
            flow_name_normalized = flow_name.replace(" ", "_").lower()

        # Aggregate metrics per block (coalesce chunked runs)
        aggregated = aggregate_block_metrics(block_metrics)

        metrics_data = {
            "flow_name": flow_name,
            "flow_version": flow_version,
            "execution_timestamp": timestamp,
            "execution_successful": execution_successful,
            "total_execution_time": sum(m["execution_time"] for m in aggregated),
            "total_wall_time": time.perf_counter() - run_start_time,  # end-to-end
            "total_blocks": len(aggregated),
            "successful_blocks": sum(1 for m in aggregated if m["status"] == "success"),
            "failed_blocks": sum(1 for m in aggregated if m["status"] == "failed"),
            "block_metrics": aggregated,
        }

        metrics_filename = f"{flow_name_normalized}_{timestamp}_metrics.json"
        metrics_path = Path(log_dir) / metrics_filename
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, sort_keys=True)

        if logger:
            logger.info(f"Metrics saved to: {metrics_path}")

    except Exception as e:
        # Metrics saving failed, warn but do not break flow
        if logger:
            logger.warning(f"Failed to save metrics: {e}")
