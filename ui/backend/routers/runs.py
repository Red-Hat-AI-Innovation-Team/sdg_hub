# SPDX-License-Identifier: Apache-2.0
"""Flow runs history router – all /api/runs/* endpoints."""

from pathlib import Path
from typing import Any, Dict
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import ALLOWED_DATASET_DIRS, UPLOADS_DIR, DATA_DIR
from models.runs import FlowRunRecord, LogAnalysisRequest
from utils.config_utils import parse_llm_statistics_from_logs
from utils.dataset_utils import load_runs_history, save_runs_history
from utils.security import is_path_within_allowed_dirs

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/runs/list")
async def list_runs():
    """Get list of all flow runs."""
    try:
        runs = load_runs_history()
        return {"runs": runs}
    except Exception as e:
        logger.error(f"Error loading runs history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/runs/create")
async def create_run(run: FlowRunRecord):
    """Create a new run record."""
    try:
        runs = load_runs_history()
        runs.append(run.model_dump())
        save_runs_history(runs)
        logger.info(f"Created run record: {run.run_id}")
        return {"status": "success", "run": run.model_dump()}
    except Exception as e:
        logger.error(f"Error creating run record: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/runs/{run_id}/update")
async def update_run(run_id: str, updates: Dict[str, Any]):
    """Update an existing run record."""
    try:
        runs = load_runs_history()
        for run in runs:
            if run["run_id"] == run_id:
                run.update(updates)
                save_runs_history(runs)
                logger.info(f"Updated run record: {run_id}")
                return {"status": "success", "run": run}
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating run record: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete a run record."""
    try:
        runs = load_runs_history()
        runs = [r for r in runs if r["run_id"] != run_id]
        save_runs_history(runs)
        logger.info(f"Deleted run record: {run_id}")
        return {"status": "success", "message": f"Run {run_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting run record: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get a specific run record."""
    try:
        runs = load_runs_history()
        for run in runs:
            if run["run_id"] == run_id:
                return run
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting run record: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/runs/config/{config_id}")
async def get_runs_by_config(config_id: str):
    """Get all runs for a specific configuration."""
    try:
        runs = load_runs_history()
        config_runs = [r for r in runs if r.get("config_id") == config_id]
        config_runs.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        return {"runs": config_runs, "total": len(config_runs)}
    except Exception as e:
        logger.error(f"Error getting runs for config {config_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/runs/{run_id}/analyze-logs")
async def analyze_run_logs(run_id: str, request: LogAnalysisRequest):
    """Analyze raw generation logs and extract LLM statistics."""
    try:
        stats = parse_llm_statistics_from_logs(request.raw_logs)
        
        runs = load_runs_history()
        for run in runs:
            if run["run_id"] == run_id:
                run["llm_statistics"] = stats
                run["llm_requests"] = stats["total_llm_requests"]
                save_runs_history(runs)
                logger.info(f"Analyzed logs for run {run_id}: {stats['total_llm_requests']} LLM requests from {len(stats.get('llm_blocks', []))} LLM blocks")
                return {
                    "status": "success",
                    "run_id": run_id,
                    "statistics": stats
                }
        
        logger.warning(f"Run {run_id} not found, but returning parsed stats anyway")
        return {
            "status": "warning",
            "message": f"Run {run_id} not found in history, stats not saved",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error analyzing logs for run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/runs/{run_id}/preview")
async def preview_run_output(run_id: str, max_rows: int = 5):
    """Preview the generated dataset for a completed run."""
    try:
        runs = load_runs_history()
        run = None
        for r in runs:
            if r["run_id"] == run_id:
                run = r
                break

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        if run["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Run is not completed (status: {run['status']})",
            )

        output_file = run.get("output_file")
        if not output_file:
            raise HTTPException(
                status_code=404, detail="No output file found for this run"
            )

        file_path = Path(output_file).resolve()
        if not is_path_within_allowed_dirs(file_path, ALLOWED_DATASET_DIRS + [UPLOADS_DIR, DATA_DIR]):
            raise HTTPException(status_code=403, detail="Output file is outside allowed directories")
        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Output file not found: {output_file}"
            )

        rows = []
        columns = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if i >= max_rows:
                    break
                try:
                    row = json.loads(line)
                    rows.append(row)
                    for key in row.keys():
                        if key not in columns:
                            columns.append(key)
                except json.JSONDecodeError:
                    continue

        total_rows = 0
        with open(file_path, "r", encoding="utf-8") as f:
            total_rows = sum(1 for line in f if line.strip())

        return {
            "run_id": run_id,
            "columns": columns,
            "rows": rows,
            "total_rows": total_rows,
            "preview_rows": len(rows),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error previewing run output: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/runs/{run_id}/download")
async def download_run_output(run_id: str):
    """Download the generated dataset for a completed run."""
    try:
        runs = load_runs_history()
        run = None
        for r in runs:
            if r["run_id"] == run_id:
                run = r
                break

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        if run["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Run is not completed (status: {run['status']})",
            )

        output_file = run.get("output_file")
        if not output_file:
            raise HTTPException(
                status_code=404, detail="No output file found for this run"
            )

        file_path = Path(output_file).resolve()
        if not is_path_within_allowed_dirs(file_path, ALLOWED_DATASET_DIRS + [UPLOADS_DIR, DATA_DIR]):
            raise HTTPException(status_code=403, detail="Output file is outside allowed directories")
        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Output file not found: {output_file}"
            )

        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type="application/x-jsonlines",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading run output: {e}")
        raise HTTPException(status_code=500, detail=str(e))
