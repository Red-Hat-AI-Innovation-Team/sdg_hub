# SPDX-License-Identifier: Apache-2.0
"""Workspace management router – all /api/workspace/* endpoints."""

import logging
import os
import shutil
import time

from fastapi import APIRouter, HTTPException

from config import (
    CUSTOM_FLOWS_DIR,
    ALLOWED_FLOW_READ_DIRS,
)
from models.workspace import (
    CreateWorkspaceRequest,
    UpdateWorkspaceFlowRequest,
    UpdateWorkspacePromptRequest,
    FinalizeWorkspaceRequest,
)
from utils.file_handling import sanitize_filename, slugify_name
from utils.security import (
    ensure_within_directory,
    safe_join,
    get_trusted_flow_source_dir,
    build_trusted_flow_source_dirs,
    is_path_within_allowed_dirs,
    validate_workspace_id,
)
from utils.safe_io import copy_validated_file

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/workspace/create")
async def create_workspace(request: CreateWorkspaceRequest):
    """Create a new workspace for flow editing."""
    import yaml
    import secrets
    from pathlib import PurePosixPath
    
    try:
        timestamp = int(time.time())
        random_suffix = secrets.token_hex(4)
        workspace_id = f"temp_ws_{timestamp}_{random_suffix}"
        
        CUSTOM_FLOWS_DIR.mkdir(parents=True, exist_ok=True)
        workspace_dir = ensure_within_directory(
            CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / workspace_id
        )
        workspace_dir.mkdir(exist_ok=True)
        
        flow_data = None
        blocks = []
        
        logger.info(f"Creating workspace: source_flow_name='{request.source_flow_name}'")
        
        if request.source_flow_name:
            source_dir = get_trusted_flow_source_dir(request.source_flow_name)
            logger.info(f"Registry lookup result: {source_dir}")
            
            if not source_dir or not source_dir.exists():
                flow_name_clean = request.source_flow_name.replace(" (Custom)", "").replace(" (Copy)", "")
                
                custom_flow_dir_name = slugify_name(flow_name_clean, prefix="flow")
                custom_flow_dir = CUSTOM_FLOWS_DIR / custom_flow_dir_name
                
                if custom_flow_dir.exists() and (custom_flow_dir / "flow.yaml").exists():
                    source_dir = custom_flow_dir
                    logger.info(f"Found custom flow source: {source_dir}")
                else:
                    if CUSTOM_FLOWS_DIR.exists():
                        for flow_dir in CUSTOM_FLOWS_DIR.iterdir():
                            if flow_dir.is_dir() and not flow_dir.name.startswith("temp_"):
                                flow_yaml = flow_dir / "flow.yaml"
                                if flow_yaml.exists():
                                    try:
                                        with open(flow_yaml, "r") as f:
                                            flow_content = yaml.safe_load(f)
                                            metadata_name = flow_content.get("metadata", {}).get("name", "")
                                            if (metadata_name == flow_name_clean or 
                                                metadata_name == request.source_flow_name or
                                                f"{metadata_name} (Custom)" == request.source_flow_name):
                                                source_dir = flow_dir
                                                logger.info(f"Found custom flow by metadata name: {source_dir}")
                                                break
                                    except Exception as e:
                                        logger.warning(f"Could not read flow.yaml in {flow_dir}: {e}")
            
            if not source_dir or not source_dir.exists():
                raise HTTPException(status_code=404, detail=f"Source flow '{request.source_flow_name}' not found")
            
            source_resolved = source_dir.resolve()
            if not is_path_within_allowed_dirs(source_resolved, ALLOWED_FLOW_READ_DIRS + [CUSTOM_FLOWS_DIR]):
                raise HTTPException(status_code=400, detail="Source flow is not in a trusted directory")
            
            logger.info(f"Copying files from source: {source_resolved}")
            copied_files = []
            for src_file in source_resolved.iterdir():
                if src_file.is_file():
                    safe_name = os.path.basename(src_file.name)
                    dst_file = safe_join(workspace_dir, safe_name)
                    copy_validated_file(src_file, dst_file)
                    copied_files.append(safe_name)
            logger.info(f"Copied {len(copied_files)} files: {copied_files}")
            
            flow_yaml_path = workspace_dir / "flow.yaml"
            if flow_yaml_path.exists():
                with open(flow_yaml_path, "r") as f:
                    flow_data = yaml.safe_load(f)
                
                blocks = flow_data.get("blocks", [])
                flow_modified = False
                for block in blocks:
                    block_config = block.get("block_config", {})
                    if "prompt_config_path" in block_config:
                        prompt_path = block_config["prompt_config_path"]
                        
                        if ".." in prompt_path or "/" in prompt_path:
                            resolved_path = (source_dir / prompt_path).resolve()
                            
                            if (
                                resolved_path.exists()
                                and resolved_path.is_file()
                                and is_path_within_allowed_dirs(resolved_path, ALLOWED_FLOW_READ_DIRS + [CUSTOM_FLOWS_DIR])
                            ):
                                dst_filename = os.path.basename(resolved_path.name)
                                dst_file = safe_join(workspace_dir, dst_filename)
                                
                                if not dst_file.exists():
                                    copy_validated_file(resolved_path, dst_file)
                                    copied_files.append(f"{dst_filename} (from {prompt_path})")
                                    logger.info(f"Copied referenced file: {prompt_path} -> {dst_filename}")
                                
                                block_config["prompt_config_path"] = dst_filename
                                flow_modified = True
                            else:
                                logger.warning(f"Referenced prompt file not found: {prompt_path} (resolved to {resolved_path})")
                
                if flow_modified:
                    with open(flow_yaml_path, "w") as f:
                        yaml.dump(flow_data, f, default_flow_style=False, sort_keys=False)
                    logger.info(f"Updated flow.yaml with resolved prompt paths")
                
                for block in blocks:
                    block_config = block.get("block_config", {})
                    if "prompt_config_path" in block_config:
                        prompt_filename = block_config["prompt_config_path"]
                        full_path = str(workspace_dir / prompt_filename)
                        block_config["full_prompt_path"] = full_path
        else:
            logger.info(f"Creating EMPTY workspace (no source_flow_name provided)")
            flow_data = {
                "metadata": {
                    "name": "New Flow",
                    "description": "A custom flow",
                    "version": "1.0.0",
                    "author": "SDG Hub User",
                    "tags": [],
                },
                "blocks": []
            }
            
            flow_yaml_path = workspace_dir / "flow.yaml"
            with open(flow_yaml_path, "w") as f:
                yaml.dump(flow_data, f, default_flow_style=False, sort_keys=False)
            
            blocks = []
        
        logger.info(f"Created workspace: {workspace_id} (source: {request.source_flow_name or 'empty'})")
        
        return {
            "workspace_id": workspace_id,
            "workspace_path": str(workspace_dir),
            "flow_data": flow_data,
            "blocks": blocks,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/workspace/{workspace_id}/update-flow")
async def update_workspace_flow(workspace_id: str, request: UpdateWorkspaceFlowRequest):
    """Update the flow.yaml in a workspace."""
    workspace_id = os.path.basename(workspace_id)
    import yaml
    from pathlib import PurePosixPath
    
    try:
        workspace_dir = validate_workspace_id(workspace_id)
        
        if not workspace_dir.exists():
            raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")
        
        blocks_list = []
        if request.blocks:
            for block in request.blocks:
                if isinstance(block, dict):
                    blocks_list.append(dict(block))
                else:
                    blocks_list.append(block.copy() if hasattr(block, 'copy') else dict(block))
        
        flow_data = {
            "metadata": request.metadata,
            "blocks": blocks_list,
        }
        
        for block in flow_data.get("blocks", []):
            block_config = block.get("block_config", {})
            if "prompt_config_path" in block_config:
                prompt_path = block_config["prompt_config_path"]
                
                filename = os.path.basename(PurePosixPath(prompt_path).name)
                if not filename:
                    continue
                
                workspace_file_by_name = safe_join(workspace_dir, filename)
                
                if workspace_file_by_name.exists():
                    if prompt_path != filename:
                        logger.info(f"Normalizing prompt path: {prompt_path} -> {filename}")
                    block_config["prompt_config_path"] = filename
                else:
                    found_file = None
                    
                    trusted_dirs = build_trusted_flow_source_dirs()
                    for flow_name, source_dir in trusted_dirs.items():
                        candidate = source_dir / filename
                        if candidate.exists() and candidate.is_file():
                            found_file = candidate.resolve()
                            break
                    
                    if not found_file and CUSTOM_FLOWS_DIR.exists():
                        for flow_dir in CUSTOM_FLOWS_DIR.iterdir():
                            if flow_dir.is_dir() and flow_dir != workspace_dir:
                                candidate = flow_dir / filename
                                if candidate.exists() and candidate.is_file():
                                    found_file = candidate.resolve()
                                    break
                    
                    if found_file:
                        if is_path_within_allowed_dirs(found_file, ALLOWED_FLOW_READ_DIRS + [CUSTOM_FLOWS_DIR]):
                            safe_dst_name = os.path.basename(filename)
                            dst_file = safe_join(workspace_dir, safe_dst_name)
                            if not dst_file.exists():
                                shutil.copy2(str(found_file.resolve()), str(dst_file.resolve()))
                                logger.info(f"Auto-copied missing prompt file during update: {found_file} -> {dst_file}")
                            block_config["prompt_config_path"] = filename
                        else:
                            logger.warning(f"Found file outside trusted dirs, skipping: {found_file}")
                    else:
                        logger.warning(f"Could not find prompt file: {filename} (original path: {prompt_path})")
        
        flow_yaml_path = workspace_dir / "flow.yaml"
        with open(flow_yaml_path, "w") as f:
            yaml.dump(flow_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated flow.yaml in workspace: {workspace_id}")
        
        return {"status": "success", "message": "Flow updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating workspace flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/workspace/{workspace_id}/update-prompt")
async def update_workspace_prompt(workspace_id: str, request: UpdateWorkspacePromptRequest):
    """Create or update a prompt file in the workspace."""
    workspace_id = os.path.basename(workspace_id)
    import yaml
    
    try:
        workspace_dir = validate_workspace_id(workspace_id)
        
        if not workspace_dir.exists():
            raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")
        
        safe_filename = sanitize_filename(request.prompt_filename)
        if not safe_filename.endswith(".yaml"):
            safe_filename += ".yaml"
        
        prompt_content = request.prompt_config
        if isinstance(prompt_content, dict) and 'messages' in prompt_content:
            prompt_content = prompt_content['messages']
        
        prompt_path = ensure_within_directory(workspace_dir, workspace_dir / safe_filename)
        with open(prompt_path, "w") as f:
            yaml.dump(prompt_content, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated prompt '{safe_filename}' in workspace: {workspace_id}")
        
        return {
            "status": "success",
            "prompt_filename": safe_filename,
            "full_prompt_path": str(prompt_path),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating workspace prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/workspace/{workspace_id}/finalize")
async def finalize_workspace(workspace_id: str, request: FinalizeWorkspaceRequest):
    """Finalize a workspace by renaming it to a permanent flow directory."""
    workspace_id = os.path.basename(workspace_id)
    import yaml
    
    try:
        workspace_dir = validate_workspace_id(workspace_id)
        
        if not workspace_dir.exists():
            raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")
        
        clean_name = os.path.basename(request.flow_name)
        for suffix in [" (Custom)", " (Copy)"]:
            clean_name = clean_name.replace(suffix, "")
        
        final_dir_name = slugify_name(clean_name, prefix="flow")
        final_dir = ensure_within_directory(
            CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / final_dir_name
        )
        
        if final_dir.exists():
            counter = 1
            while True:
                final_dir_name = f"{slugify_name(clean_name, prefix='flow')}_{counter}"
                final_dir = ensure_within_directory(
                    CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / final_dir_name
                )
                if not final_dir.exists():
                    break
                counter += 1
        
        flow_yaml_path = ensure_within_directory(workspace_dir, workspace_dir / "flow.yaml")
        if flow_yaml_path.exists():
            with open(str(flow_yaml_path.resolve()), "r") as f:
                flow_data = yaml.safe_load(f)
            
            if flow_data:
                flow_data["metadata"] = flow_data.get("metadata", {})
                flow_data["metadata"]["name"] = request.flow_name
                
                for block in flow_data.get("blocks", []):
                    block_config = block.get("block_config", {})
                    if "full_prompt_path" in block_config:
                        del block_config["full_prompt_path"]
                
                with open(str(flow_yaml_path.resolve()), "w") as f:
                    yaml.dump(flow_data, f, default_flow_style=False, sort_keys=False)
        
        shutil.move(str(workspace_dir.resolve()), str(final_dir.resolve()))
        
        logger.info(f"Finalized workspace {workspace_id} -> {final_dir_name}")
        
        return {
            "status": "success",
            "flow_name": request.flow_name,
            "flow_dir": final_dir_name,
            "flow_path": str(final_dir / "flow.yaml"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finalizing workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/workspace/{workspace_id}")
async def delete_workspace(workspace_id: str):
    """Delete a workspace (cleanup on cancel)."""
    workspace_id = os.path.basename(workspace_id)
    try:
        workspace_dir = validate_workspace_id(workspace_id)
        
        if not workspace_dir.exists():
            return {"status": "success", "message": "Workspace already deleted"}
        
        ensure_within_directory(CUSTOM_FLOWS_DIR, workspace_dir)
        shutil.rmtree(str(workspace_dir.resolve()))
        
        logger.info(f"Deleted workspace: {workspace_id}")
        
        return {"status": "success", "message": f"Workspace '{workspace_id}' deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/workspace/{workspace_id}/blocks")
async def get_workspace_blocks(workspace_id: str):
    """Get the blocks from a workspace with full prompt paths resolved."""
    workspace_id = os.path.basename(workspace_id)
    import yaml
    
    try:
        workspace_dir = validate_workspace_id(workspace_id)
        
        if not workspace_dir.exists():
            raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")
        
        flow_yaml_path = ensure_within_directory(workspace_dir, workspace_dir / "flow.yaml")
        if not flow_yaml_path.exists():
            return {"blocks": [], "metadata": {}}
        
        with open(str(flow_yaml_path.resolve()), "r") as f:
            flow_data = yaml.safe_load(f)
        
        blocks = flow_data.get("blocks", [])
        for block in blocks:
            block_config = block.get("block_config", {})
            if "prompt_config_path" in block_config:
                prompt_filename = block_config["prompt_config_path"]
                full_path = str(workspace_dir / prompt_filename)
                block_config["full_prompt_path"] = full_path
        
        return {
            "blocks": blocks,
            "metadata": flow_data.get("metadata", {}),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workspace blocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))
