# SPDX-License-Identifier: Apache-2.0
"""Custom flows router – all /api/custom-flows/* endpoints."""

import io
import logging
import os
import shutil
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from config import CUSTOM_FLOWS_DIR
from utils.security import ensure_within_directory

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/custom-flows")
async def list_custom_flows():
    """List all custom flows with their files for download."""
    import yaml
    
    try:
        custom_flows = []
        
        if not CUSTOM_FLOWS_DIR.exists():
            return {"flows": []}
        
        for flow_dir in sorted(CUSTOM_FLOWS_DIR.iterdir()):
            if not flow_dir.is_dir() or flow_dir.name.startswith("temp_"):
                continue
            
            flow_yaml_path = flow_dir / "flow.yaml"
            if not flow_yaml_path.exists():
                continue
            
            try:
                with open(flow_yaml_path, "r") as f:
                    flow_data = yaml.safe_load(f)
                
                metadata = flow_data.get("metadata", {})
                
                yaml_files = []
                for yaml_file in sorted(flow_dir.glob("*.yaml")):
                    file_stat = yaml_file.stat()
                    yaml_files.append({
                        "filename": yaml_file.name,
                        "size_bytes": file_stat.st_size,
                        "modified": file_stat.st_mtime,
                    })
                
                custom_flows.append({
                    "directory_name": flow_dir.name,
                    "name": metadata.get("name", flow_dir.name),
                    "description": metadata.get("description", ""),
                    "version": metadata.get("version", "1.0.0"),
                    "author": metadata.get("author", "Unknown"),
                    "tags": metadata.get("tags", []),
                    "files": yaml_files,
                    "file_count": len(yaml_files),
                })
            except Exception as e:
                logger.warning(f"Could not read flow {flow_dir.name}: {e}")
                continue
        
        return {"flows": custom_flows}
        
    except Exception as e:
        logger.error(f"Error listing custom flows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/custom-flows/{flow_name}/download/{filename}")
async def download_custom_flow_file(flow_name: str, filename: str):
    """Download a specific YAML file from a custom flow."""
    try:
        safe_flow_name = os.path.basename(flow_name)
        safe_filename = os.path.basename(filename)
        if not safe_flow_name or not safe_filename:
            raise HTTPException(status_code=400, detail="Invalid flow name or filename")
        
        flow_dir = ensure_within_directory(
            CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / safe_flow_name
        )
        
        if not flow_dir.exists():
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")
        
        if not safe_filename.endswith(".yaml"):
            raise HTTPException(status_code=400, detail="Only YAML files can be downloaded")
        
        file_path = ensure_within_directory(flow_dir, flow_dir / safe_filename)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
        
        return FileResponse(
            path=str(file_path.resolve()),
            filename=safe_filename,
            media_type="application/x-yaml",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/custom-flows/{flow_name}/download-all")
async def download_custom_flow_all(flow_name: str):
    """Download all files from a custom flow as a ZIP archive."""
    try:
        safe_flow_name = os.path.basename(flow_name)
        if not safe_flow_name:
            raise HTTPException(status_code=400, detail="Invalid flow name")
        
        flow_dir = ensure_within_directory(
            CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / safe_flow_name
        )
        
        if not flow_dir.exists():
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for yaml_file in flow_dir.glob("*.yaml"):
                zip_file.write(yaml_file, yaml_file.name)
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={flow_name}.zip"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating ZIP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/custom-flows")
async def delete_all_custom_flows():
    """Delete all custom flows (excluding temp workspaces)."""
    try:
        if not CUSTOM_FLOWS_DIR.exists():
            return {"status": "success", "deleted_count": 0, "message": "No custom flows directory exists"}
        
        deleted_count = 0
        errors = []
        
        for flow_dir in CUSTOM_FLOWS_DIR.iterdir():
            if flow_dir.is_dir() and not flow_dir.name.startswith("temp_"):
                try:
                    shutil.rmtree(flow_dir)
                    deleted_count += 1
                    logger.info(f"Deleted custom flow: {flow_dir.name}")
                except Exception as e:
                    errors.append(f"{flow_dir.name}: {str(e)}")
                    logger.error(f"Error deleting flow {flow_dir.name}: {e}")
        
        if errors:
            return {
                "status": "partial",
                "deleted_count": deleted_count,
                "errors": errors,
                "message": f"Deleted {deleted_count} flows with {len(errors)} errors"
            }
        
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} custom flows"
        }
        
    except Exception as e:
        logger.error(f"Error deleting all custom flows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/custom-flows/{flow_name}")
async def delete_custom_flow(flow_name: str):
    """Delete a custom flow and all its files."""
    try:
        if flow_name.startswith("temp_"):
            raise HTTPException(status_code=400, detail="Cannot delete temp workspaces through this endpoint")
        
        safe_name = os.path.basename(flow_name)
        if not safe_name or safe_name != flow_name:
            raise HTTPException(status_code=400, detail="Invalid flow name")
        
        flow_dir = ensure_within_directory(
            CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / safe_name
        )
        
        if not flow_dir.exists():
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")
        
        shutil.rmtree(str(flow_dir.resolve()))
        
        logger.info(f"Deleted custom flow: {flow_name}")
        
        return {"status": "success", "message": f"Deleted flow '{flow_name}'"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))
