# SPDX-License-Identifier: Apache-2.0
"""Flow execution metrics utilities for display and export."""

# Standard
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json
import time

# Third Party
from datasets import Dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


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
    final_dataset: Optional[Dataset] = None,
) -> None:
    """Display a rich table summarizing block execution metrics.

    Parameters
    ----------
    block_metrics : list[dict[str, Any]]
        Raw block metrics from flow execution.
    flow_name : str
        Name of the flow for display title.
    final_dataset : Optional[Dataset], optional
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
    final_row_count = len(final_dataset) if final_dataset else 0
    final_col_count = len(final_dataset.column_names) if final_dataset else 0

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
