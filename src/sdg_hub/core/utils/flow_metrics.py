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
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for m in entries:
        key = (m.get("block_name"), m.get("block_type"))
        a = agg.setdefault(
            key,
            {
                "block_name": key[0],
                "block_type": key[1],
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
        # Format duration
        duration = f"{metrics['execution_time']:.2f}s"
        total_time += metrics["execution_time"]

        # Format row changes
        if metrics["status"] == "success":
            row_change = f"{metrics['input_rows']:,} → {metrics['output_rows']:,}"
            successful_blocks += 1
        else:
            row_change = f"{metrics['input_rows']:,} → ❌"

        # Format column changes
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

        # Format status with color
        if metrics["status"] == "success":
            status = "[green]✓[/green]"
        else:
            status = "[red]✗[/red]"

        table.add_row(
            metrics["block_name"],
            metrics["block_type"],
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

    # Determine panel title and border color based on execution status
    failed_blocks = len(block_metrics) - successful_blocks
    if final_dataset is None:
        # Flow failed completely
        title = (
            f"[bold bright_white]{flow_name}[/bold bright_white] - [red]Failed[/red]"
        )
        border_style = "bright_red"
    elif failed_blocks == 0:
        # All blocks succeeded
        title = f"[bold bright_white]{flow_name}[/bold bright_white] - [green]Complete[/green]"
        border_style = "bright_green"
    else:
        # Some blocks failed but flow completed
        title = f"[bold bright_white]{flow_name}[/bold bright_white] - [yellow]Partial[/yellow]"
        border_style = "bright_yellow"

    console.print(
        Panel(
            table,
            title=title,
            border_style=border_style,
        )
    )
    console.print()


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

    # Create main summary table
    summary_table = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
    )
    summary_table.add_column("Metric", style="bright_cyan")
    summary_table.add_column("Value", style="bright_white")

    # Format time
    est_seconds = time_estimation["estimated_time_seconds"]
    if est_seconds < 60:
        time_str = f"{est_seconds:.1f} seconds"
    elif est_seconds < 3600:
        time_str = f"{est_seconds / 60:.1f} minutes ({est_seconds / 3600:.2f} hours)"
    else:
        time_str = f"{est_seconds / 3600:.2f} hours ({est_seconds / 60:.0f} minutes)"

    summary_table.add_row("Estimated Time:", time_str)
    summary_table.add_row(
        "Total LLM Requests:", f"{time_estimation.get('total_estimated_requests', 0):,}"
    )

    if time_estimation.get("total_estimated_requests", 0) > 0:
        requests_per_sample = time_estimation["total_estimated_requests"] / dataset_size
        summary_table.add_row("Requests per Sample:", f"{requests_per_sample:.1f}")

    if max_concurrency is not None:
        summary_table.add_row("Max Concurrency:", str(max_concurrency))

    # Display summary panel
    console.print()
    console.print(
        Panel(
            summary_table,
            title=f"[bold bright_white]Time Estimation for {dataset_size:,} Samples[/bold bright_white]",
            border_style="bright_blue",
        )
    )

    # Display per-block breakdown if available
    block_estimates = time_estimation.get("block_estimates", [])
    if block_estimates:
        console.print()

        # Create per-block table
        block_table = Table(
            show_header=True,
            header_style="bold bright_white",
        )
        block_table.add_column("Block Name", style="bright_cyan", width=20)
        block_table.add_column("Time", justify="right", style="bright_yellow", width=10)
        block_table.add_column(
            "Requests", justify="right", style="bright_green", width=10
        )
        block_table.add_column(
            "Throughput", justify="right", style="bright_blue", width=12
        )
        block_table.add_column(
            "Amplif.", justify="right", style="bright_magenta", width=10
        )

        for block in block_estimates:
            # Format time
            block_seconds = block["estimated_time"]
            if block_seconds < 60:
                time_str = f"{block_seconds:.1f}s"
            else:
                time_str = f"{block_seconds / 60:.1f}min"

            # Format requests
            requests_str = f"{block['estimated_requests']:,.0f}"

            # Format throughput
            throughput_str = f"{block['throughput']:.2f}/s"

            # Format amplification
            amplif_str = f"{block['amplification']:.1f}x"

            block_table.add_row(
                block["block"],
                time_str,
                requests_str,
                throughput_str,
                amplif_str,
            )

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
