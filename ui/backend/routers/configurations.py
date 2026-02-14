# SPDX-License-Identifier: Apache-2.0
"""Saved configurations router – all /api/configurations/* endpoints."""

from pathlib import Path
import logging

from fastapi import APIRouter, HTTPException

from sdg_hub import Flow, FlowRegistry

from models.configurations import SavedConfiguration, SaveConfigurationRequest
from state import current_config, saved_configurations
from utils.api_key_utils import sanitize_model_config
from utils.config_utils import (
    load_saved_configurations_from_disk,
    persist_saved_configurations,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _load_saved_configs_from_disk():
    """Load saved configurations from disk into module-level dict."""
    load_saved_configurations_from_disk(saved_configurations, SavedConfiguration)


def _persist_saved_configs():
    """Persist module-level saved configurations to disk."""
    persist_saved_configurations(saved_configurations)


@router.post("/api/configurations/save")
async def save_configuration(request: SaveConfigurationRequest):
    """Save a flow configuration (API keys stored locally for convenience)."""
    try:
        from datetime import datetime
        import uuid

        config_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        original_api_key = request.model_configuration.get("api_key", "")
        is_direct_api_key = bool(original_api_key) and not (
            original_api_key in ["EMPTY", "NONE", ""]
            or original_api_key.startswith("env:")
        )

        stored_model_config = request.model_configuration.copy()

        config = SavedConfiguration(
            id=config_id,
            flow_name=request.flow_name,
            flow_id=request.flow_id,
            flow_path=request.flow_path,
            model_configuration=stored_model_config,
            dataset_configuration=request.dataset_configuration,
            dry_run_configuration=request.dry_run_configuration,
            tags=request.tags,
            status=request.status or "configured",
            created_at=now,
            updated_at=now,
        )

        saved_configurations[config_id] = config
        _persist_saved_configs()
        logger.info(f"✅ Saved configuration: {config_id} for flow {request.flow_name}")

        response_config = config.dict()
        response_config["model_configuration"] = sanitize_model_config(
            response_config["model_configuration"], mask_key=True
        )

        response = {
            "status": "success",
            "config_id": config_id,
            "configuration": response_config,
        }

        if is_direct_api_key:
            response["warning"] = (
                "⚠️ API key stored locally in plaintext for convenience. "
                "Remove this configuration if you share this machine."
            )

        return response

    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/configurations/list")
async def list_configurations():
    """List all saved configurations."""
    try:
        configs = []
        for config in saved_configurations.values():
            config_dict = config.dict()
            config_dict["model_configuration"] = sanitize_model_config(
                config_dict["model_configuration"], mask_key=True
            )
            configs.append(config_dict)

        logger.info(f"Listed {len(configs)} configurations")
        return {"status": "success", "configurations": configs}
    except Exception as e:
        logger.error(f"Error listing configurations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/configurations/{config_id}")
async def get_configuration(config_id: str):
    """Get a specific configuration."""
    try:
        if config_id not in saved_configurations:
            raise HTTPException(
                status_code=404, detail=f"Configuration {config_id} not found"
            )

        config = saved_configurations[config_id]
        config_dict = config.dict()

        config_dict["model_configuration"] = sanitize_model_config(
            config_dict["model_configuration"], mask_key=True
        )

        return {"status": "success", "configuration": config_dict}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/configurations/{config_id}")
async def delete_configuration(config_id: str):
    """Delete a configuration."""
    try:
        if config_id not in saved_configurations:
            raise HTTPException(
                status_code=404, detail=f"Configuration {config_id} not found"
            )

        del saved_configurations[config_id]
        _persist_saved_configs()
        logger.info(f"Deleted configuration: {config_id}")
        return {"status": "success", "message": f"Configuration {config_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/configurations/{config_id}/load")
async def load_configuration(config_id: str):
    """Load a configuration into current state."""
    try:
        if config_id not in saved_configurations:
            raise HTTPException(
                status_code=404, detail=f"Configuration {config_id} not found"
            )

        config = saved_configurations[config_id]

        flow = None
        flow_path_str = None

        if config.flow_path and config.flow_path != "." and config.flow_path != "":
            flow_path = Path(config.flow_path)
            if flow_path.exists():
                flow = Flow.from_yaml(str(flow_path))
                flow_path_str = config.flow_path

        if flow is None:
            try:
                if config.flow_id:
                    flow_path_str = FlowRegistry.get_flow_path(config.flow_id)
                    if flow_path_str:
                        flow = Flow.from_yaml(flow_path_str)
            except Exception as e:
                logger.warning(f"Could not find flow by ID, trying by name: {e}")

            if flow is None and config.flow_name:
                flow_path_str = FlowRegistry.get_flow_path(config.flow_name)
                if flow_path_str:
                    flow = Flow.from_yaml(flow_path_str)

        if flow is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not load flow. Path: {config.flow_path}, ID: {config.flow_id}, Name: {config.flow_name}",
            )

        current_config["flow"] = flow
        current_config["flow_path"] = flow_path_str
        current_config["model_config"] = config.model_configuration

        if config.model_configuration:
            flow.set_model_config(**config.model_configuration)

        logger.info(f"Loaded configuration: {config_id}")
        return {
            "status": "success",
            "message": f"Configuration {config_id} loaded",
            "configuration": config.dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))
