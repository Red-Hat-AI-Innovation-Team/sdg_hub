# SPDX-License-Identifier: Apache-2.0
"""Flow execution router – dry-run, generation, checkpoints, download, and uploads."""

from pathlib import Path
from typing import Any, Dict, Optional
import asyncio
import concurrent.futures
import io
import json
import logging
import multiprocessing
import os
import queue
import signal
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
import nest_asyncio
import pandas as pd

from sdg_hub import BlockRegistry, Flow, FlowRegistry

from config import (
    UPLOADS_DIR,
    CUSTOM_FLOWS_DIR,
    OUTPUTS_DIR,
    ALLOWED_FLOW_READ_DIRS,
)
from models.execution import DryRunRequest, TestStepByStepRequest
from state import (
    current_config,
    active_dry_run,
    active_generations,
    active_generation_process,
    saved_configurations,
)
from utils.api_key_utils import get_safe_api_key, sanitize_model_config
from utils.checkpoint_utils import get_checkpoint_dir_for_config, get_checkpoint_info, clear_checkpoints
from utils.file_handling import sanitize_filename
from utils.security import (
    ensure_within_directory,
    safe_join,
    resolve_dataset_file,
    build_trusted_flow_source_dirs,
    is_path_within_allowed_dirs,
    validate_workspace_id,
)
from workers.dry_run_worker import dry_run_worker
from workers.generation_worker import generation_worker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/uploads/{filename}")
async def download_uploaded_file(filename: str):
    """Download a file from the uploads directory."""
    filename = os.path.basename(filename)
    try:
        safe_filename = sanitize_filename(filename)
        if not safe_filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = UPLOADS_DIR / safe_filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not file_path.resolve().is_relative_to(UPLOADS_DIR.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")
        
        logger.info(f"📥 Downloading file: {file_path}")
        
        return FileResponse(
            path=str(file_path.resolve()),
            filename=safe_filename,
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/flow/dry-run-stream")
async def dry_run_stream(
    sample_size: int = 2,
    enable_time_estimation: bool = False,
    max_concurrency: int = None,
    config_id: Optional[str] = None,
):
    """Stream dry run execution logs in real-time using Server-Sent Events."""

    async def generate_logs():
        """Generator that yields log events as they occur."""
        try:
            if current_config["flow"] is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No flow selected'})}\n\n"
                return
            if current_config["dataset"] is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No dataset loaded'})}\n\n"
                return

            # Check if another dry run is already running
            if active_dry_run.get("pid"):
                yield f"data: {json.dumps({'type': 'error', 'message': 'Another dry run is already in progress'})}\n\n"
                return

            flow_path = current_config.get("flow_path")
            if not flow_path:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No flow path available'})}\n\n"
                return

            model_config = current_config.get("model_config", {})
            dataset_params = current_config.get("dataset_load_params")
            
            if not dataset_params:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No dataset parameters available'})}\n\n"
                return

            logger.info(f"🧪 Starting dry run process (config_id={config_id}, sample_size={sample_size})")

            # Use spawn context for clean process isolation
            ctx = multiprocessing.get_context("spawn")
            log_queue = ctx.Queue()

            # Start worker process
            process = ctx.Process(
                target=dry_run_worker,
                args=(
                    log_queue,
                    flow_path,
                    model_config,
                    dataset_params,
                    sample_size,
                    enable_time_estimation,
                    max_concurrency,
                ),
            )

            process.start()
            
            # Track the active dry run
            active_dry_run["pid"] = process.pid
            active_dry_run["config_id"] = config_id
            active_dry_run["start_time"] = time.time()
            active_dry_run["process"] = process
            active_dry_run["queue"] = log_queue

            logger.info(f"🧪 Dry run process started (PID={process.pid})")

            # Send start message
            yield f"data: {json.dumps({'type': 'start', 'message': f'Starting dry run with {sample_size} samples (PID={process.pid})'})}\n\n"

            # Stream logs while process runs
            while process.is_alive():
                try:
                    log_entry = log_queue.get(timeout=0.1)
                    
                    if log_entry.get("type") == "complete":
                        logger.info(f"✅ Dry run completed in {log_entry.get('result', {}).get('execution_time_seconds', 0):.2f}s")
                        yield f"data: {json.dumps(log_entry)}\n\n"
                        break
                    elif log_entry.get("type") == "error":
                        logger.error(f"❌ Dry run error: {log_entry.get('message')}")
                        yield f"data: {json.dumps(log_entry)}\n\n"
                        break
                    else:
                        yield f"data: {json.dumps(log_entry)}\n\n"
                except Exception:
                    await asyncio.sleep(0.1)

            # Drain any remaining messages from the queue
            while True:
                try:
                    log_entry = log_queue.get_nowait()
                    if log_entry.get("type") == "complete":
                        logger.info(f"✅ Dry run completed in {log_entry.get('result', {}).get('execution_time_seconds', 0):.2f}s")
                    yield f"data: {json.dumps(log_entry)}\n\n"
                except Exception:
                    break

            # Wait for process to finish
            process.join(timeout=2)

            # Check exit code
            if process.exitcode is None:
                logger.warning(f"⚠️ Dry run process didn't finish in time, killing (PID={process.pid})")
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            elif process.exitcode != 0 and process.exitcode != -9:
                logger.error(f"❌ Dry run process exited with code {process.exitcode}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Process exited with code {process.exitcode}'})}\n\n"

            logger.info(f"🛑 Dry run process finished (PID={process.pid}, exit_code={process.exitcode})")

        except Exception as e:
            logger.error(f"❌ Dry run stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        finally:
            active_dry_run["pid"] = None
            active_dry_run["config_id"] = None
            active_dry_run["start_time"] = None
            active_dry_run["process"] = None
            active_dry_run["queue"] = None

    return StreamingResponse(
        generate_logs(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/api/flow/dry-run")
async def dry_run_flow(request: DryRunRequest):
    """Perform a dry run of the configured flow (non-streaming)."""
    try:
        if current_config["flow"] is None:
            raise HTTPException(status_code=400, detail="No flow selected")
        if current_config["dataset"] is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")

        flow = current_config["flow"]
        dataset = current_config["dataset"]

        def run_dry_run_sync():
            """Run dry run in a separate thread with its own event loop."""
            return flow.dry_run(
                dataset,
                sample_size=request.sample_size,
                enable_time_estimation=request.enable_time_estimation,
                max_concurrency=request.max_concurrency,
            )

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            dry_result = await loop.run_in_executor(executor, run_dry_run_sync)

        logger.info(f"Dry run completed: {dry_result['execution_time_seconds']:.2f}s")

        return {"status": "success", "result": dry_result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during dry run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/flow/cancel-dry-run")
async def cancel_dry_run():
    """Cancel an active dry run by killing the worker process."""
    pid = active_dry_run.get("pid")
    config_id = active_dry_run.get("config_id")
    
    if not pid:
        return {
            "status": "success",
            "message": "No dry run is currently running",
        }
    
    logger.info(f"🛑 Cancelling dry run process (PID={pid}, config_id={config_id})")
    
    try:
        os.kill(pid, signal.SIGKILL)
        
        active_dry_run["pid"] = None
        active_dry_run["config_id"] = None
        active_dry_run["start_time"] = None
        active_dry_run["process"] = None
        active_dry_run["queue"] = None
        
        logger.info(f"✅ Dry run process killed (PID={pid})")
        
        return {
            "status": "success",
            "message": f"Dry run process {pid} stopped",
            "config_id": config_id,
        }
    except ProcessLookupError:
        logger.warning(f"⚠️ Dry run process {pid} not found (already finished)")
        active_dry_run["pid"] = None
        active_dry_run["config_id"] = None
        active_dry_run["start_time"] = None
        active_dry_run["process"] = None
        active_dry_run["queue"] = None
        
        return {
            "status": "success",
            "message": "Process already finished or not found",
        }
    except Exception as e:
        logger.error(f"❌ Failed to cancel dry run: {e}")
        return {
            "status": "error",
            "message": f"Failed to cancel: {str(e)}",
        }


@router.get("/api/flow/dry-run-status")
async def get_dry_run_status():
    """Get the current status of any active dry run."""
    pid = active_dry_run.get("pid")
    
    if not pid:
        return {
            "is_running": False,
            "pid": None,
            "config_id": None,
            "elapsed_seconds": None,
        }
    
    elapsed = time.time() - active_dry_run["start_time"] if active_dry_run["start_time"] else 0
    
    return {
        "is_running": True,
        "pid": pid,
        "config_id": active_dry_run["config_id"],
        "elapsed_seconds": round(elapsed, 2),
    }


@router.post("/api/flow/test-step-by-step")
async def test_flow_step_by_step(request: TestStepByStepRequest):
    """Execute a flow step-by-step for testing in the visual flow editor."""
    def execute_block_in_thread(block_instance, dataset):
        """Execute a block in a separate thread to avoid event loop conflicts."""
        return block_instance(dataset)
    
    async def generate_events():
        try:
            import yaml
            
            model_config = request.model_config_data
            sample_data = request.sample_data
            workspace_id = os.path.basename(request.workspace_id) if request.workspace_id else None
            
            # Load blocks from workspace if workspace_id is provided
            if workspace_id:
                try:
                    workspace_dir = validate_workspace_id(workspace_id)
                except HTTPException:
                    yield f"data: {json.dumps({'type': 'test_error', 'message': 'Invalid workspace ID'})}\n\n"
                    return
                
                if not workspace_dir.exists():
                    yield f"data: {json.dumps({'type': 'test_error', 'message': f'Workspace not found: {workspace_id}'})}\n\n"
                    return
                
                flow_yaml_path = workspace_dir / "flow.yaml"
                if not flow_yaml_path.exists():
                    yield f"data: {json.dumps({'type': 'test_error', 'message': 'No flow.yaml in workspace'})}\n\n"
                    return
                
                with open(str(flow_yaml_path.resolve()), "r") as f:
                    flow_data = yaml.safe_load(f)
                
                blocks_config = flow_data.get("blocks", [])
                
                import shutil
                from pathlib import PurePosixPath
                missing_files = []
                copied_files = []
                
                for block in blocks_config:
                    block_config = block.get("block_config", {})
                    if "prompt_config_path" in block_config and "full_prompt_path" not in block_config:
                        prompt_path = block_config["prompt_config_path"]
                        
                        prompt_filename = os.path.basename(PurePosixPath(prompt_path).name)
                        if not prompt_filename:
                            continue
                        full_path = safe_join(workspace_dir, prompt_filename)
                        
                        if not full_path.exists():
                            found_file = None
                            trusted_dirs = build_trusted_flow_source_dirs()
                            for flow_name, source_dir in trusted_dirs.items():
                                candidate = source_dir / prompt_filename
                                if candidate.exists() and candidate.is_file():
                                    found_file = candidate
                                    break
                            
                            if not found_file and CUSTOM_FLOWS_DIR.exists():
                                for flow_dir in CUSTOM_FLOWS_DIR.iterdir():
                                    if flow_dir.is_dir():
                                        candidate = flow_dir / prompt_filename
                                        if candidate.exists() and candidate.is_file():
                                            found_file = candidate
                                            break
                            
                            if found_file:
                                found_resolved = found_file.resolve()
                                if is_path_within_allowed_dirs(found_resolved, ALLOWED_FLOW_READ_DIRS + [CUSTOM_FLOWS_DIR]):
                                    shutil.copy2(str(found_resolved), str(full_path.resolve()))
                                    copied_files.append(f"{prompt_filename} (from {found_file.parent.name})")
                                    logger.info(f"🧪 Auto-copied missing prompt file: {found_resolved} -> {full_path}")
                                else:
                                    logger.warning(f"🧪 Found file outside trusted dirs, skipping: {found_file}")
                            else:
                                block_name = block_config.get("block_name", "unknown")
                                missing_files.append(f"{prompt_filename} (block: {block_name})")
                        
                        block_config["full_prompt_path"] = str(full_path)
                
                if copied_files:
                    logger.info(f"🧪 Auto-copied {len(copied_files)} missing prompt files: {copied_files}")
                
                if missing_files:
                    error_msg = f"Missing prompt files in workspace: {', '.join(missing_files)}. Could not find these files in any trusted flow directory."
                    logger.error(f"🧪 {error_msg}")
                    yield f"data: {json.dumps({'type': 'test_error', 'message': error_msg})}\n\n"
                    return
                
                logger.info(f"🧪 Loaded {len(blocks_config)} blocks from workspace: {workspace_id}")
            else:
                blocks_config = request.blocks
            
            if not blocks_config:
                yield f"data: {json.dumps({'type': 'test_error', 'message': 'No blocks provided'})}\n\n"
                return
            
            df = pd.DataFrame([sample_data])
            
            logger.info(f"🧪 Starting step-by-step test with {len(blocks_config)} blocks")
            logger.info(f"📊 Initial dataset columns: {df.columns.tolist()}")
            
            block_results = []
            current_dataset = df
            
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            loop = asyncio.get_event_loop()
            
            for i, block_config in enumerate(blocks_config):
                block_type = block_config.get("block_type")
                block_name = block_config.get("block_config", {}).get("block_name", f"block_{i}")
                
                yield f"data: {json.dumps({'type': 'block_start', 'block_index': i, 'block_name': block_name, 'block_type': block_type})}\n\n"
                
                try:
                    input_data = current_dataset.to_dict(orient='records')
                    input_columns = current_dataset.columns.tolist()
                    
                    block_class = BlockRegistry._get(block_type)
                    
                    config = block_config.get("block_config", {}).copy()
                    
                    if block_type == 'PromptBuilderBlock' and 'full_prompt_path' in config:
                        config['prompt_config_path'] = config.pop('full_prompt_path')
                    
                    block_instance = block_class(**config)
                    
                    is_llm_block = hasattr(block_instance, 'block_type') and block_instance.block_type == 'llm'
                    if is_llm_block:
                        if model_config.get("model"):
                            block_instance.model = model_config["model"]
                        if model_config.get("api_base"):
                            block_instance.api_base = model_config["api_base"]
                        if model_config.get("api_key"):
                            from pydantic import SecretStr
                            block_instance.api_key = SecretStr(model_config["api_key"])
                    
                    start_time_block = time.time()
                    if is_llm_block:
                        output_dataset = await loop.run_in_executor(
                            executor, 
                            execute_block_in_thread, 
                            block_instance, 
                            current_dataset
                        )
                    else:
                        output_dataset = block_instance(current_dataset)
                    execution_time = time.time() - start_time_block
                    
                    output_data = output_dataset.to_dict(orient='records')
                    output_columns = output_dataset.columns.tolist()
                    
                    new_columns = [col for col in output_columns if col not in input_columns]
                    
                    block_result = {
                        'block_index': i,
                        'block_name': block_name,
                        'block_type': block_type,
                        'status': 'success',
                        'execution_time_ms': int(execution_time * 1000),
                        'input_data': input_data,
                        'input_columns': input_columns,
                        'output_data': output_data,
                        'output_columns': output_columns,
                        'new_columns': new_columns,
                        'input_rows': len(current_dataset),
                        'output_rows': len(output_dataset),
                    }
                    block_results.append(block_result)
                    
                    current_dataset = output_dataset
                    
                    yield f"data: {json.dumps({'type': 'block_complete', **block_result})}\n\n"
                    
                    logger.info(f"✅ Block {i} ({block_name}) completed in {execution_time:.2f}s")
                    
                    if len(current_dataset) == 0:
                        remaining = len(blocks_config) - i - 1
                        note = (
                            f"Filter block '{block_name}' removed all samples. "
                            f"Skipping {remaining} remaining block(s). "
                            "This is normal in single-sample test mode when the "
                            "evaluation judges the sample below the threshold."
                        )
                        logger.warning(f"⚠️ {note}")
                        yield f"data: {json.dumps({'type': 'filter_empty', 'block_index': i, 'block_name': block_name, 'message': note})}\n\n"
                        
                        for skip_idx in range(i + 1, len(blocks_config)):
                            skip_block = blocks_config[skip_idx]
                            skip_name = skip_block.get("block_config", {}).get("block_name", f"block_{skip_idx}")
                            skip_type = skip_block.get("block_type", "unknown")
                            yield f"data: {json.dumps({'type': 'block_complete', 'block_index': skip_idx, 'block_name': skip_name, 'block_type': skip_type, 'status': 'skipped', 'input_data': [], 'output_data': [], 'input_columns': output_columns, 'output_columns': output_columns, 'new_columns': [], 'input_rows': 0, 'output_rows': 0, 'execution_time_ms': 0, 'skipped': True, 'skip_reason': 'Empty dataset from filter block'})}\n\n"
                        
                        yield f"data: {json.dumps({'type': 'test_complete', 'message': f'Test completed. {note}', 'total_blocks': len(blocks_config), 'final_output': [], 'final_columns': output_columns, 'filter_emptied': True})}\n\n"
                        logger.info(f"🎉 Step-by-step test completed (filter emptied dataset at block {i})")
                        return
                    
                except Exception as block_error:
                    error_msg = str(block_error)
                    logger.error(f"❌ Block {i} ({block_name}) failed: {error_msg}")
                    
                    yield f"data: {json.dumps({'type': 'block_error', 'block_index': i, 'block_name': block_name, 'block_type': block_type, 'error': error_msg})}\n\n"
                    
                    yield f"data: {json.dumps({'type': 'test_error', 'message': f'Block {block_name} failed: {error_msg}', 'failed_block_index': i})}\n\n"
                    return
            
            final_output = current_dataset.to_dict(orient='records')
            
            yield f"data: {json.dumps({'type': 'test_complete', 'message': 'Test completed successfully', 'total_blocks': len(blocks_config), 'final_output': final_output, 'final_columns': current_dataset.columns.tolist()})}\n\n"
            
            logger.info(f"🎉 Step-by-step test completed successfully with {len(blocks_config)} blocks")
            
        except Exception as e:
            logger.error(f"❌ Test execution error: {e}")
            yield f"data: {json.dumps({'type': 'test_error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/flow/generate-stream")
async def generate_stream(
    max_concurrency: int = None,
    log_dir: str = None,
    config_id: Optional[str] = None,
    enable_checkpoints: bool = True,
    save_freq: Optional[int] = None,
    resume_from_checkpoint: bool = False,
):
    """Stream flow generation logs in real-time using multiprocessing."""

    async def generate_logs():
        """Generator that yields log events as they occur."""
        try:
            flow_path = None
            dataset_params = None
            model_config = {}
            flow_obj = None
            flow_name = "unknown"

            if config_id and config_id in saved_configurations:
                config = saved_configurations[config_id]
                logger.info(
                    f"🔧 Loading isolated config for generation: {config_id} (flow: {config.flow_name})"
                )

                if (
                    config.flow_path
                    and config.flow_path != "."
                    and config.flow_path != ""
                ):
                    flow_path_obj = Path(config.flow_path)
                    if flow_path_obj.exists():
                        flow_path = config.flow_path
                        flow_obj = Flow.from_yaml(flow_path)

                if flow_path is None:
                    try:
                        if config.flow_id:
                            flow_path = FlowRegistry.get_flow_path(config.flow_id)
                            if flow_path:
                                flow_obj = Flow.from_yaml(flow_path)
                    except Exception as e:
                        logger.warning(
                            f"Could not find flow by ID, trying by name: {e}"
                        )

                    if flow_path is None and config.flow_name:
                        flow_path = FlowRegistry.get_flow_path(config.flow_name)
                        if flow_path:
                            flow_obj = Flow.from_yaml(flow_path)

                if flow_path is None:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Could not load flow for config {config_id}'})}\n\n"
                    return

                model_config = (config.model_configuration or {}).copy()

                dataset_config = config.dataset_configuration or {}
                if (
                    dataset_config.get("data_files")
                    and dataset_config.get("data_files") != "."
                ):
                    safe_dataset_path = resolve_dataset_file(
                        dataset_config["data_files"]
                    )
                    dataset_params = {
                        "data_files": str(safe_dataset_path),
                        "file_format": dataset_config.get("file_format", "jsonl"),
                        "csv_delimiter": dataset_config.get("csv_delimiter", ","),
                        "csv_encoding": dataset_config.get("csv_encoding", "utf-8"),
                        "shuffle": dataset_config.get("shuffle", False),
                        "seed": dataset_config.get("seed", 42),
                        "num_samples": dataset_config.get("num_samples"),
                        "added_columns": dataset_config.get("added_columns"),
                    }
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Dataset not configured for this flow'})}\n\n"
                    return

                flow_name = config.flow_name or (
                    flow_obj.metadata.name if flow_obj else "unknown"
                )

            else:
                if not current_config["flow"] or not current_config["flow_path"]:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'No flow selected'})}\n\n"
                    return

                if not current_config.get("dataset_load_params"):
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Dataset source info missing. Please reload the dataset.'})}\n\n"
                    return

                flow_path = current_config["flow_path"]
                dataset_params = current_config["dataset_load_params"]
                model_config = current_config.get("model_config", {}).copy()
                flow_obj = current_config["flow"]
                flow_name = flow_obj.metadata.name if flow_obj else "unknown"

            if model_config.get("api_key"):
                model_config["api_key"] = get_safe_api_key(model_config)

            ctx = multiprocessing.get_context("spawn")
            log_queue = ctx.Queue()

            if (
                active_generation_process.get("pid")
                and active_generation_process.get("config_id") == config_id
            ):
                old_pid = active_generation_process["pid"]
                logger.warning(
                    f"⚠️ Previous generation for config {config_id} still active (PID={old_pid}). Killing it."
                )
                try:
                    os.kill(old_pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                active_generation_process["pid"] = None
                active_generation_process["config_id"] = None
                if config_id in active_generations:
                    del active_generations[config_id]

            checkpoint_dir = None
            if enable_checkpoints and config_id:
                checkpoint_dir = str(get_checkpoint_dir_for_config(config_id))

            process = ctx.Process(
                target=generation_worker,
                args=(
                    log_queue,
                    flow_path,
                    model_config,
                    dataset_params,
                    max_concurrency,
                    log_dir,
                    checkpoint_dir,
                    save_freq if enable_checkpoints else None,
                    resume_from_checkpoint,
                ),
            )

            process.start()
            active_generation_process["pid"] = process.pid
            active_generation_process["config_id"] = config_id

            active_generations[config_id] = {
                "queue": log_queue,
                "process": process,
                "start_time": time.time(),
                "flow_name": flow_name,
                "flow_path": flow_path,
                "checkpoint_dir": checkpoint_dir,
                "resume_from_checkpoint": resume_from_checkpoint,
            }

            logger.info(
                f"🚀 Generation worker started (PID={process.pid}) for flow: {flow_path} (config_id={config_id})"
            )

            yield f"data: {json.dumps({'type': 'start', 'message': f'Starting generation process (PID: {process.pid})'})}\n\n"

            while process.is_alive():
                while not log_queue.empty():
                    try:
                        item = log_queue.get_nowait()

                        if item["type"] == "result":
                            dataset_list = item["dataset_list"]
                            column_names = item["column_names"]
                            num_samples = len(dataset_list)
                            num_columns = len(column_names)

                            current_config["generated_dataset"] = dataset_list
                            current_config["generated_dataset_info"] = {
                                "num_samples": num_samples,
                                "num_columns": num_columns,
                                "columns": column_names,
                            }

                            from datetime import datetime

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_flow_name = (
                                sanitize_filename(flow_name.replace(" ", "_").lower())
                                if flow_name
                                else "unknown"
                            )
                            output_filename = f"{output_flow_name}_{timestamp}.jsonl"
                            output_path = ensure_within_directory(
                                OUTPUTS_DIR, OUTPUTS_DIR / output_filename
                            )

                            with open(output_path, "w") as f:
                                for record in dataset_list:
                                    f.write(json.dumps(record) + "\n")

                            logger.info(f"Saved generated dataset to: {output_path}")
                            current_config["last_generated_file"] = str(output_path)

                            completion_data = {
                                "type": "complete",
                                "num_samples": num_samples,
                                "num_columns": num_columns,
                                "output_file": output_filename,
                                "llm_requests": item.get("llm_requests", 0),
                            }
                            yield f"data: {json.dumps(completion_data)}\n\n"

                        elif item["type"] == "error":
                            yield f"data: {json.dumps(item)}\n\n"
                        else:
                            yield f"data: {json.dumps(item)}\n\n"

                    except queue.Empty:
                        break

                await asyncio.sleep(0.1)

            while not log_queue.empty():
                try:
                    item = log_queue.get_nowait()
                    if item["type"] == "error":
                        yield f"data: {json.dumps(item)}\n\n"
                except Exception:
                    break

            process.join()
            logger.info(
                f"🛑 Generation worker finished (PID={process.pid}, exit_code={process.exitcode}, config_id={config_id})"
            )
            active_generation_process["pid"] = None
            active_generation_process["config_id"] = None

            if config_id and config_id in active_generations:
                del active_generations[config_id]

            if process.exitcode != 0:
                msg = (
                    "Generation cancelled."
                    if process.exitcode in [-15, -9]
                    else f"Process exited with code {process.exitcode}"
                )
                yield f"data: {json.dumps({'type': 'error', 'message': msg})}\n\n"

        except Exception as e:
            logger.error(f"Error in generation stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_logs(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/api/flow/generation-status")
async def get_generation_status(config_id: Optional[str] = None):
    """Check if there are any running generations."""
    try:
        running_generations = []

        dead_configs = []
        for cfg_id, gen_info in active_generations.items():
            process = gen_info.get("process")
            if process and not process.is_alive():
                dead_configs.append(cfg_id)

        for cfg_id in dead_configs:
            del active_generations[cfg_id]
            if active_generation_process.get("config_id") == cfg_id:
                active_generation_process["pid"] = None
                active_generation_process["config_id"] = None

        if config_id:
            if config_id in active_generations:
                gen_info = active_generations[config_id]
                process = gen_info.get("process")
                if process and process.is_alive():
                    return {
                        "status": "running",
                        "config_id": config_id,
                        "pid": process.pid,
                        "start_time": gen_info.get("start_time"),
                        "can_reconnect": True,
                    }
            return {
                "status": "not_running",
                "config_id": config_id,
                "can_reconnect": False,
            }

        for cfg_id, gen_info in active_generations.items():
            process = gen_info.get("process")
            if process and process.is_alive():
                running_generations.append(
                    {
                        "config_id": cfg_id,
                        "pid": process.pid,
                        "start_time": gen_info.get("start_time"),
                    }
                )

        return {
            "status": "success",
            "running_generations": running_generations,
            "count": len(running_generations),
        }

    except Exception as e:
        logger.error(f"Error checking generation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/flow/reconnect-stream")
async def reconnect_stream(config_id: str):
    """Reconnect to an existing running generation's log stream."""

    async def generate_logs():
        """Generator that yields log events from an existing process."""
        try:
            if config_id not in active_generations:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No active generation found for this configuration'})}\n\n"
                return

            gen_info = active_generations[config_id]
            process = gen_info.get("process")
            log_queue = gen_info.get("queue")

            if not process or not process.is_alive():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Generation process is no longer running'})}\n\n"
                if config_id in active_generations:
                    del active_generations[config_id]
                return

            yield f"data: {json.dumps({'type': 'reconnected', 'message': f'Reconnected to generation process (PID: {process.pid})'})}\n\n"

            while process.is_alive():
                while not log_queue.empty():
                    try:
                        item = log_queue.get_nowait()

                        if item["type"] == "result":
                            dataset_list = item["dataset_list"]
                            column_names = item["column_names"]
                            num_samples = len(dataset_list)
                            num_columns = len(column_names)

                            current_config["generated_dataset"] = dataset_list
                            current_config["generated_dataset_info"] = {
                                "num_samples": num_samples,
                                "num_columns": num_columns,
                                "columns": column_names,
                            }

                            from datetime import datetime

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            raw_flow_name = (
                                gen_info.get("flow_name", "flow")
                                .replace(" ", "_")
                                .lower()
                            )
                            safe_flow_name = sanitize_filename(raw_flow_name) or "flow"
                            output_filename = f"{safe_flow_name}_{timestamp}.jsonl"
                            output_path = ensure_within_directory(
                                OUTPUTS_DIR, OUTPUTS_DIR / output_filename
                            )

                            with open(output_path, "w") as f:
                                for record in dataset_list:
                                    f.write(json.dumps(record) + "\n")

                            logger.info(f"Saved generated dataset to: {output_path}")
                            current_config["last_generated_file"] = str(output_path)

                            completion_data = {
                                "type": "complete",
                                "num_samples": num_samples,
                                "num_columns": num_columns,
                                "output_file": output_filename,
                                "llm_requests": item.get("llm_requests", 0),
                            }
                            yield f"data: {json.dumps(completion_data)}\n\n"

                            if config_id in active_generations:
                                del active_generations[config_id]

                        elif item["type"] == "error":
                            yield f"data: {json.dumps(item)}\n\n"
                            if config_id in active_generations:
                                del active_generations[config_id]
                        else:
                            yield f"data: {json.dumps(item)}\n\n"

                    except queue.Empty:
                        break

                await asyncio.sleep(0.1)

            while not log_queue.empty():
                try:
                    item = log_queue.get_nowait()
                    if item["type"] == "result":
                        dataset_list = item["dataset_list"]
                        column_names = item["column_names"]
                        completion_data = {
                            "type": "complete",
                            "num_samples": len(dataset_list),
                            "num_columns": len(column_names),
                            "output_file": None,
                            "llm_requests": item.get("llm_requests", 0),
                        }
                        yield f"data: {json.dumps(completion_data)}\n\n"
                    elif item["type"] == "error":
                        yield f"data: {json.dumps(item)}\n\n"
                except Exception:
                    break

            process.join()
            logger.info(
                f"🛑 Reconnected generation finished (PID={process.pid}, config_id={config_id})"
            )

            if config_id in active_generations:
                del active_generations[config_id]
            if active_generation_process.get("config_id") == config_id:
                active_generation_process["pid"] = None
                active_generation_process["config_id"] = None

            if process.exitcode != 0:
                msg = (
                    "Generation cancelled."
                    if process.exitcode in [-15, -9]
                    else f"Process exited with code {process.exitcode}"
                )
                yield f"data: {json.dumps({'type': 'error', 'message': msg})}\n\n"

        except Exception as e:
            logger.error(f"Error in reconnect stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_logs(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/api/flow/cancel-generation")
async def cancel_generation(config_id: Optional[str] = None):
    """Cancel generation by killing the worker process."""
    try:
        pid = active_generation_process.get("pid")
        active_config_id = active_generation_process.get("config_id")
        logger.info(
            f"🧵 Cancel request received. Active PID: {pid}, active_config_id: {active_config_id}, requested_config_id: {config_id}"
        )

        if config_id and active_config_id and config_id != active_config_id:
            logger.warning(
                f"⚠️ Cancel request for config {config_id} ignored; active config is {active_config_id}."
            )
            return {
                "status": "ignored",
                "message": f"Active generation belongs to a different configuration ({active_config_id}).",
            }

        if pid:
            logger.warning(f"⚠️ Stopping generation process: {pid}")
            try:
                os.kill(pid, signal.SIGKILL)

                active_generation_process["pid"] = None
                active_generation_process["config_id"] = None

                if config_id and config_id in active_generations:
                    del active_generations[config_id]
                elif active_config_id and active_config_id in active_generations:
                    del active_generations[active_config_id]

                logger.info(f"✅ Successfully killed process {pid}")

                return {
                    "status": "success",
                    "message": f"Generation process {pid} stopped.",
                }
            except ProcessLookupError:
                logger.warning(f"⚠️ Process {pid} not found when attempting to cancel.")
                active_generation_process["pid"] = None
                active_generation_process["config_id"] = None
                if config_id and config_id in active_generations:
                    del active_generations[config_id]
                return {
                    "status": "success",
                    "message": "Process already finished or not found.",
                }
            except Exception as kill_error:
                logger.error(f"❌ Failed to cancel process {pid}: {kill_error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to cancel process {pid}: {kill_error}",
                )
        else:
            logger.info(
                "ℹ️ Cancel requested but no active generation process was tracked."
            )
            active_generation_process["config_id"] = None
            if config_id and config_id in active_generations:
                del active_generations[config_id]
            return {
                "status": "success",
                "message": "No active generation process found.",
            }

    except Exception as e:
        logger.error(f"Error cancelling generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/flow/checkpoints/{config_id}")
async def get_checkpoints(config_id: str):
    """Get checkpoint information for a configuration."""
    try:
        info = get_checkpoint_info(config_id)
        return {"status": "success", "config_id": config_id, **info}
    except Exception as e:
        logger.error(f"Error getting checkpoint info for {config_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/flow/checkpoints/{config_id}")
async def delete_checkpoints(config_id: str):
    """Clear all checkpoints for a configuration."""
    try:
        success = clear_checkpoints(config_id)
        if success:
            return {
                "status": "success",
                "message": f"Checkpoints cleared for configuration {config_id}",
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to clear checkpoints for {config_id}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing checkpoints for {config_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/flow/download-generated")
async def download_generated():
    """Download the generated dataset as JSONL."""
    try:
        if not current_config.get("generated_dataset"):
            raise HTTPException(
                status_code=404, detail="No generated dataset available"
            )

        dataset_list = current_config["generated_dataset"]

        output = io.StringIO()
        for item in dataset_list:
            output.write(json.dumps(item) + "\n")

        info = current_config.get("generated_dataset_info", {})
        num_samples = info.get("num_samples", len(dataset_list))

        return Response(
            content=output.getvalue(),
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f"attachment; filename=generated_data_{num_samples}_samples_{int(time.time())}.jsonl"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading generated dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))
