# SPDX-License-Identifier: Apache-2.0
"""Config, blocks, and prompts router – /api/config/*, /api/blocks/*, /api/prompts/*."""

from pathlib import Path
from typing import Any, Dict
import json
import logging
import os
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from sdg_hub import BlockRegistry, Flow, FlowRegistry

from config import CUSTOM_FLOWS_DIR
from models.datasets import DatasetFormat
from state import current_config
from utils.api_key_utils import sanitize_model_config
from utils.file_handling import (
    load_dataset_as_pandas,
    sanitize_filename,
    slugify_name,
)
from utils.security import (
    ensure_within_directory,
    resolve_flow_file,
    resolve_prompt_file,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/config/import")
async def import_config(file: UploadFile = File(...)):
    """Import a previously exported configuration file."""
    try:
        content = await file.read()
        config_data = json.loads(content)

        flow_info = config_data.get("flow")
        model_cfg = config_data.get("model_config")
        dataset_cfg = config_data.get("dataset_config")

        if not flow_info or not flow_info.get("name"):
            raise HTTPException(
                status_code=400,
                detail="Invalid configuration file: missing flow information",
            )

        flow_name = flow_info["name"]
        flow_path = FlowRegistry.get_flow_path(flow_name)
        if not flow_path:
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")

        flow = Flow.from_yaml(flow_path)
        current_config["flow"] = flow
        current_config["flow_path"] = flow_path

        logger.info(f"Imported flow: {flow_name}")

        if model_cfg:
            kwargs = {}
            if model_cfg.get("model"):
                kwargs["model"] = model_cfg["model"]
            if model_cfg.get("api_base"):
                kwargs["api_base"] = model_cfg["api_base"]
            if model_cfg.get("api_key"):
                kwargs["api_key"] = model_cfg["api_key"]
            if model_cfg.get("additional_params"):
                kwargs.update(model_cfg["additional_params"])

            if kwargs:
                flow.set_model_config(**kwargs)
                current_config["model_config"] = model_cfg
                logger.info(f"Applied model configuration: {model_cfg.get('model')}")

        dataset_loaded = False
        if dataset_cfg and dataset_cfg.get("data_files"):
            try:
                data_files = dataset_cfg["data_files"]
                file_format = dataset_cfg.get("file_format", "auto")

                df = load_dataset_as_pandas(
                    Path(data_files),
                    DatasetFormat(file_format)
                    if file_format != "auto"
                    else DatasetFormat.AUTO,
                    dataset_cfg.get("csv_delimiter", ","),
                    dataset_cfg.get("csv_encoding", "utf-8"),
                )

                if dataset_cfg.get("shuffle"):
                    df = df.sample(
                        frac=1, random_state=dataset_cfg.get("seed", 42)
                    ).reset_index(drop=True)

                if dataset_cfg.get("num_samples"):
                    df = df.head(min(dataset_cfg["num_samples"], len(df)))

                current_config["dataset"] = df
                current_config["dataset_info"] = {
                    "num_samples": len(df),
                    "columns": df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                }
                dataset_loaded = True
                logger.info(f"📊 Loaded dataset: {len(df)} samples (pandas)")
            except Exception as e:
                logger.warning(f"Could not load dataset from config: {e}")

        return {
            "status": "success",
            "message": "Configuration imported successfully",
            "flow": {
                "name": flow.metadata.name,
                "id": flow.metadata.id,
                "version": flow.metadata.version,
            },
            "model_configured": bool(model_cfg),
            "dataset_loaded": dataset_loaded,
            "imported_config": {
                "flow": flow_info,
                "model_config": model_cfg,
                "dataset_config": dataset_cfg,
            },
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}")
    except Exception as e:
        logger.error(f"Error importing config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/config/current")
async def get_current_config():
    """Get the current configuration state."""
    flow_info = None
    if current_config["flow"]:
        flow = current_config["flow"]
        flow_info = {
            "name": flow.metadata.name,
            "id": flow.metadata.id,
            "version": flow.metadata.version,
            "blocks_count": len(flow.blocks),
        }

    safe_model_config = sanitize_model_config(
        current_config["model_config"], mask_key=True
    )

    return {
        "flow": flow_info,
        "model_config": safe_model_config,
        "dataset_info": current_config["dataset_info"],
    }


@router.post("/api/config/reset")
async def reset_config():
    """Reset the current configuration."""
    current_config["flow"] = None
    current_config["flow_path"] = None
    current_config["model_config"] = {}
    current_config["dataset"] = None
    current_config["dataset_info"] = {}

    logger.info("Configuration reset")

    return {"status": "success", "message": "Configuration reset"}


@router.get("/api/blocks/list")
async def list_blocks():
    """List all available block types."""
    try:
        blocks = BlockRegistry.list_blocks()
        return {"blocks": blocks}
    except Exception as e:
        logger.error(f"Error listing blocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/blocks/templates")
async def get_block_templates():
    """Get pre-configured block templates from existing flows."""
    try:
        import yaml

        block_templates = []

        flows = FlowRegistry.list_flows()

        for flow_info in flows:
            try:
                flow_name = flow_info["name"]
                flow_path = FlowRegistry.get_flow_path(flow_name)

                if not flow_path:
                    continue

                try:
                    validated_path = resolve_flow_file(flow_path)
                except HTTPException:
                    logger.warning(f"Skipping flow {flow_name}: path validation failed")
                    continue

                with open(validated_path, "r") as f:
                    flow_data = yaml.safe_load(f)

                blocks = flow_data.get("blocks", [])
                for block in blocks:
                    block_config = block.get("block_config", {})
                    block_name = block_config.get("block_name", "unknown")

                    template = {
                        "id": f"{flow_name}::{block_name}",
                        "name": block_name,
                        "type": block.get("block_type"),
                        "source_flow": flow_name,
                        "config": block_config,
                        "category": "template",
                    }

                    block_templates.append(template)

            except Exception as e:
                logger.warning(f"Could not extract blocks from flow {flow_info}: {e}")
                continue

        logger.info(f"Found {len(block_templates)} block templates")
        return {"templates": block_templates}

    except Exception as e:
        logger.error(f"Error getting block templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/prompts/load")
async def load_prompt_template(prompt_path: str):
    """Load an existing prompt template YAML file."""
    try:
        import yaml

        prompt_file = resolve_prompt_file(prompt_path)

        from utils.safe_io import read_validated_file
        with read_validated_file(prompt_file) as f:
            messages = yaml.safe_load(f)

        logger.info(f"Loaded prompt template from: {prompt_file}")

        return {
            "status": "success",
            "messages": messages,
            "file_path": str(prompt_file),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading prompt template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/prompts/save")
async def save_prompt_template(prompt_data: Dict[str, Any]):
    """Save a prompt template YAML file."""
    try:
        import yaml

        prompt_name = prompt_data.get("prompt_name", "custom_prompt")
        prompt_content = prompt_data.get("prompt_content", [])
        flow_name = prompt_data.get("flow_name", "custom_flow")

        CUSTOM_FLOWS_DIR.mkdir(parents=True, exist_ok=True)

        safe_flow_name = slugify_name(flow_name, prefix="flow")
        flow_dir = ensure_within_directory(
            CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / safe_flow_name
        )
        flow_dir.mkdir(parents=True, exist_ok=True)

        prompt_filename = (
            sanitize_filename(f"{prompt_name}.yaml")
            or f"prompt_{int(time.time())}.yaml"
        )
        prompt_path = ensure_within_directory(flow_dir, flow_dir / prompt_filename)

        with open(prompt_path, "w") as f:
            yaml.dump(prompt_content, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Saved prompt template to: {prompt_path}")

        return {
            "status": "success",
            "prompt_path": str(prompt_path),
            "prompt_filename": prompt_filename,
            "message": f"Prompt '{prompt_name}' saved successfully",
        }

    except Exception as e:
        logger.error(f"Error saving prompt template: {e}")
        raise HTTPException(status_code=500, detail=str(e))
