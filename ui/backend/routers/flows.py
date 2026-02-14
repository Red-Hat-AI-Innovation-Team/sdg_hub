# SPDX-License-Identifier: Apache-2.0
"""Flow discovery, selection, creation, and template endpoints."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Form, HTTPException

from sdg_hub import Flow, FlowRegistry

from config import (
    CUSTOM_FLOWS_DIR,
    ALLOWED_FLOW_READ_DIRS,
)
from models.flows import FlowSearchRequest, FlowInfo
from state import current_config
from utils.file_handling import sanitize_filename, slugify_name
from utils.security import (
    ensure_within_directory,
    get_trusted_flow_source_dir,
    is_path_within_allowed_dirs,
    resolve_flow_file,
    _get_trusted_flow_paths,
)
from utils.safe_io import read_validated_file, copy_validated_file

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/flows/list", response_model=List[str])
async def list_flows():
    """List all available flows including custom flows."""
    try:
        flows = FlowRegistry.list_flows()
        # Extract just the flow names from the list of dicts
        flow_names = [flow["name"] for flow in flows]

        # Also check for custom flows (using validated CUSTOM_FLOWS_DIR constant)
        if CUSTOM_FLOWS_DIR.exists():
            for flow_dir in CUSTOM_FLOWS_DIR.iterdir():
                if flow_dir.is_dir():
                    flow_yaml = flow_dir / "flow.yaml"
                    if flow_yaml.exists():
                        try:
                            import yaml

                            # Validate path is within allowed directory
                            validated_path = ensure_within_directory(
                                CUSTOM_FLOWS_DIR, flow_yaml
                            )
                            with open(validated_path, "r") as f:
                                flow_data = yaml.safe_load(f)
                                custom_flow_name = flow_data.get("metadata", {}).get(
                                    "name"
                                )
                                if (
                                    custom_flow_name
                                    and custom_flow_name not in flow_names
                                ):
                                    flow_names.append(f"{custom_flow_name} (Custom)")
                        except Exception as e:
                            logger.warning(
                                f"Could not load custom flow from {flow_dir}: {e}"
                            )

        logger.info(f"Listed {len(flow_names)} flows")
        return flow_names
    except Exception as e:
        logger.error(f"Error listing flows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/flows/list-with-details", response_model=List[FlowInfo])
async def list_flows_with_details():
    """List all available flows with full details in a single request.

    This endpoint returns all flow information at once, avoiding the need
    for multiple individual /api/flows/{name}/info calls.
    """
    try:
        import yaml

        flows_with_details = []

        # Get SDG Hub flows from registry
        for entry in FlowRegistry._entries.values():
            metadata = entry.metadata
            flow_path = entry.path

            # Build recommended_models dict
            recommended_models = None
            if metadata.recommended_models:
                recommended_models = {
                    "default": metadata.recommended_models.default,
                    "compatible": metadata.recommended_models.compatible or [],
                    "experimental": metadata.recommended_models.experimental or [],
                }

            # Build dataset_requirements dict
            dataset_requirements = None
            if metadata.dataset_requirements:
                dataset_requirements = metadata.dataset_requirements.model_dump()

            flows_with_details.append(
                FlowInfo(
                    name=metadata.name,
                    id=metadata.id,
                    path=str(flow_path) if flow_path else None,
                    description=metadata.description,
                    version=metadata.version,
                    author=metadata.author,
                    tags=metadata.tags or [],
                    recommended_models=recommended_models,
                    dataset_requirements=dataset_requirements,
                )
            )

        # Also check for custom flows
        if CUSTOM_FLOWS_DIR.exists():
            existing_names = {f.name for f in flows_with_details}
            for flow_dir in CUSTOM_FLOWS_DIR.iterdir():
                if flow_dir.is_dir():
                    flow_yaml = flow_dir / "flow.yaml"
                    if flow_yaml.exists():
                        try:
                            validated_path = ensure_within_directory(
                                CUSTOM_FLOWS_DIR, flow_yaml
                            )
                            with open(validated_path, "r") as f:
                                flow_data = yaml.safe_load(f)

                            metadata = flow_data.get("metadata", {})
                            custom_flow_name = metadata.get("name")
                            if custom_flow_name and custom_flow_name not in existing_names:
                                display_name = f"{custom_flow_name} (Custom)"

                                # Extract recommended_models
                                rec_models = metadata.get("recommended_models")
                                recommended_models = None
                                if rec_models:
                                    recommended_models = {
                                        "default": rec_models.get("default"),
                                        "compatible": rec_models.get("compatible", []),
                                        "experimental": rec_models.get("experimental", []),
                                    }

                                flows_with_details.append(
                                    FlowInfo(
                                        name=display_name,
                                        id=metadata.get("id", ""),
                                        path=str(validated_path),
                                        description=metadata.get("description", ""),
                                        version=metadata.get("version", "1.0.0"),
                                        author=metadata.get("author", ""),
                                        tags=metadata.get("tags", []),
                                        recommended_models=recommended_models,
                                        dataset_requirements=metadata.get(
                                            "dataset_requirements"
                                        ),
                                    )
                                )
                                existing_names.add(custom_flow_name)
                        except Exception as e:
                            logger.warning(
                                f"Could not load custom flow from {flow_dir}: {e}"
                            )

        logger.info(f"Listed {len(flows_with_details)} flows with details")
        return flows_with_details
    except Exception as e:
        logger.error(f"Error listing flows with details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/flows/search", response_model=List[str])
async def search_flows(request: FlowSearchRequest):
    """Search flows by tag or name."""
    try:
        if request.tag:
            flows = FlowRegistry.search_flows(tag=request.tag)
            # Extract flow names from list of dicts
            flow_names = [flow["name"] for flow in flows]
            logger.info(f"Found {len(flow_names)} flows with tag '{request.tag}'")
        elif request.name_filter:
            all_flows = FlowRegistry.list_flows()
            # Extract names and filter
            all_flow_names = [flow["name"] for flow in all_flows]
            flow_names = [
                f for f in all_flow_names if request.name_filter.lower() in f.lower()
            ]
            logger.info(
                f"Found {len(flow_names)} flows matching '{request.name_filter}'"
            )
        else:
            flows = FlowRegistry.list_flows()
            flow_names = [flow["name"] for flow in flows]

        return flow_names
    except Exception as e:
        logger.error(f"Error searching flows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/flows/{flow_name:path}/info", response_model=FlowInfo)
async def get_flow_info(flow_name: str):
    """Get detailed information about a specific flow."""
    try:
        # Check if this is a custom flow (has "(Custom)" suffix)
        is_custom = flow_name.endswith(" (Custom)")
        actual_flow_name = (
            flow_name.replace(" (Custom)", "") if is_custom else flow_name
        )

        # Try to get flow path from registry
        flow_path = FlowRegistry.get_flow_path(actual_flow_name)

        # If not found in registry and is custom, check custom_flows directory
        if not flow_path and is_custom:
            # Normalize the flow name to match directory name using slugify
            flow_dir_name = slugify_name(actual_flow_name, prefix="flow")
            custom_flow_path = ensure_within_directory(
                CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / flow_dir_name / "flow.yaml"
            )

            logger.info(f"Looking for custom flow at: {custom_flow_path}")

            if custom_flow_path.exists():
                flow_path = str(custom_flow_path)
            else:
                # Try to find by scanning the directory
                if CUSTOM_FLOWS_DIR.exists():
                    for flow_dir in CUSTOM_FLOWS_DIR.iterdir():
                        if flow_dir.is_dir():
                            potential_path = flow_dir / "flow.yaml"
                            if potential_path.exists():
                                import yaml

                                # Validate path before reading
                                validated_path = ensure_within_directory(
                                    CUSTOM_FLOWS_DIR, potential_path
                                )
                                with open(validated_path, "r") as f:
                                    flow_data = yaml.safe_load(f)
                                    if (
                                        flow_data.get("metadata", {}).get("name")
                                        == actual_flow_name
                                    ):
                                        flow_path = str(validated_path)
                                        break

        if not flow_path:
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")

        # Validate flow_path is within allowed directories
        validated_flow_path = resolve_flow_file(flow_path)

        # Load flow
        flow = Flow.from_yaml(str(validated_flow_path))

        # Extract flow information
        flow_info = FlowInfo(
            name=flow.metadata.name,
            id=flow.metadata.id,
            path=str(validated_flow_path),
            description=flow.metadata.description,
            version=flow.metadata.version,
            author=flow.metadata.author,
            tags=flow.metadata.tags or [],
            recommended_models=flow.get_model_recommendations(),
            dataset_requirements=(
                flow.get_dataset_requirements().model_dump()
                if flow.get_dataset_requirements()
                else None
            ),
        )

        logger.info(f"Retrieved info for flow '{flow_name}'")
        return flow_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting flow info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/flows/select-by-path")
async def select_flow_by_path(request: Dict[str, Any]):
    """Select a flow by its file path."""
    try:
        flow_path = request.get("flow_path")
        if not flow_path:
            raise HTTPException(status_code=400, detail="flow_path is required")

        # Validate and resolve the flow path within allowed directories
        validated_flow_path = resolve_flow_file(flow_path)

        # Load the flow
        flow = Flow.from_yaml(str(validated_flow_path))

        # Update current config
        current_config["flow"] = flow
        current_config["flow_path"] = str(validated_flow_path)
        current_config["model_config"] = {}
        current_config["dataset"] = None
        current_config["dataset_info"] = {}

        logger.info(f"Selected flow from path: {validated_flow_path}")

        return {
            "status": "success",
            "message": f"Flow loaded from {flow_path}",
            "flow_name": flow.metadata.name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting flow by path: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/flows/{flow_name:path}/yaml")
async def get_flow_yaml(flow_name: str):
    """Get the raw YAML content of a flow for cloning."""
    try:
        import yaml

        # Check if this is a custom flow - can have "(Custom)" or "(Copy)" suffix
        is_custom = flow_name.endswith(" (Custom)") or flow_name.endswith(" (Copy)")
        actual_flow_name = flow_name
        if flow_name.endswith(" (Custom)"):
            actual_flow_name = flow_name.replace(" (Custom)", "")
        elif flow_name.endswith(" (Copy)"):
            actual_flow_name = flow_name.replace(" (Copy)", "")

        # Get flow path from registry first
        flow_path = FlowRegistry.get_flow_path(actual_flow_name)

        # If not found in registry, check custom_flows directory
        # (regardless of suffix - any flow in custom_flows should be loadable)
        if not flow_path:
            # First try slugified name
            flow_dir_name = slugify_name(actual_flow_name, prefix="flow")
            custom_flow_path = ensure_within_directory(
                CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / flow_dir_name / "flow.yaml"
            )

            if custom_flow_path.exists():
                flow_path = str(custom_flow_path)
            else:
                # Try to find by scanning within CUSTOM_FLOWS_DIR and matching metadata name
                if CUSTOM_FLOWS_DIR.exists():
                    for flow_dir in CUSTOM_FLOWS_DIR.iterdir():
                        if flow_dir.is_dir():
                            potential_path = flow_dir / "flow.yaml"
                            if potential_path.exists():
                                # Validate path is within allowed directory
                                validated_path = ensure_within_directory(
                                    CUSTOM_FLOWS_DIR, potential_path
                                )
                                with open(validated_path, "r") as f:
                                    flow_data = yaml.safe_load(f)
                                    metadata_name = flow_data.get("metadata", {}).get("name", "")
                                    # Match against both the actual name and the full name with suffix
                                    if metadata_name == actual_flow_name or metadata_name == flow_name:
                                        flow_path = str(validated_path)
                                        break

        if not flow_path:
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")

        # Validate flow_path is within allowed directories before reading
        validated_flow_path = resolve_flow_file(flow_path)

        # Read and parse the YAML file
        with read_validated_file(validated_flow_path) as f:
            flow_data = yaml.safe_load(f)

        logger.info(f"Retrieved YAML for flow: {flow_name}")
        return flow_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting flow YAML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/flows/{flow_name:path}/select")
async def select_flow(flow_name: str):
    """Select a flow for configuration."""
    try:
        # Check if this is a custom flow
        is_custom = flow_name.endswith(" (Custom)")
        actual_flow_name = (
            flow_name.replace(" (Custom)", "") if is_custom else flow_name
        )

        # Try to get flow path from registry
        flow_path = FlowRegistry.get_flow_path(actual_flow_name)

        # If not found and is custom, check custom_flows directory
        if not flow_path and is_custom:
            # Normalize the flow name using slugify
            flow_dir_name = slugify_name(actual_flow_name, prefix="flow")
            custom_flow_path = ensure_within_directory(
                CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / flow_dir_name / "flow.yaml"
            )

            if custom_flow_path.exists():
                flow_path = str(custom_flow_path)
            else:
                # Scan directory to find by metadata name
                if CUSTOM_FLOWS_DIR.exists():
                    import yaml

                    for flow_dir in CUSTOM_FLOWS_DIR.iterdir():
                        if flow_dir.is_dir():
                            potential_path = flow_dir / "flow.yaml"
                            if potential_path.exists():
                                # Validate path before reading
                                validated_path = ensure_within_directory(
                                    CUSTOM_FLOWS_DIR, potential_path
                                )
                                with open(validated_path, "r") as f:
                                    flow_data = yaml.safe_load(f)
                                    if (
                                        flow_data.get("metadata", {}).get("name")
                                        == actual_flow_name
                                    ):
                                        flow_path = str(validated_path)
                                        break

        if not flow_path:
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")

        # Validate flow_path is within allowed directories
        validated_flow_path = resolve_flow_file(flow_path)

        # Load flow
        flow = Flow.from_yaml(str(validated_flow_path))

        # Store in current config
        current_config["flow"] = flow
        current_config["flow_path"] = str(validated_flow_path)
        current_config["model_config"] = {}

        logger.info(f"Selected flow: {flow_name}")

        return {
            "status": "success",
            "message": f"Flow '{flow_name}' selected",
            "flow_info": {
                "name": flow.metadata.name,
                "id": flow.metadata.id,
                "version": flow.metadata.version,
                "blocks_count": len(flow.blocks),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/flows/create")
async def create_flow(
    flow_yaml: str = Form(...),
    flow_name: str = Form(...),
    prompt_templates: str = Form(None),
    source_flow_name: str = Form(None),
):
    """Save a new custom flow to the flows directory."""
    flow_name = os.path.basename(flow_name)
    source_flow_name = os.path.basename(source_flow_name) if source_flow_name else source_flow_name
    try:
        import shutil

        import yaml

        # Parse the YAML to validate it
        yaml.safe_load(flow_yaml)  # Validates YAML syntax

        # Create custom flows directory
        CUSTOM_FLOWS_DIR.mkdir(parents=True, exist_ok=True)

        # Create flow-specific directory with sanitized name
        safe_flow_dir_name = slugify_name(flow_name, prefix="flow")
        flow_dir = ensure_within_directory(
            CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / safe_flow_dir_name
        )
        flow_dir.mkdir(exist_ok=True)

        # If this is a cloned flow, copy prompt template files from source
        # Use trusted whitelist to get source directory
        if source_flow_name:
            # Get source directory from trusted registry whitelist
            trusted_source_dir = get_trusted_flow_source_dir(source_flow_name)

            if trusted_source_dir and trusted_source_dir.exists():
                # Build list of safe files to copy from trusted directory
                # Files are enumerated from trusted path, not derived from user input
                safe_yaml_files = [
                    f
                    for f in trusted_source_dir.iterdir()
                    if f.is_file() and f.suffix == ".yaml" and f.name != "flow.yaml"
                ]

                for yaml_file in safe_yaml_files:
                    # Sanitize destination filename and validate path
                    safe_filename = sanitize_filename(os.path.basename(yaml_file.name)) or "prompt.yaml"
                    dest_file = ensure_within_directory(
                        flow_dir, flow_dir / safe_filename
                    )
                    # Copy from trusted source
                    copy_validated_file(yaml_file, dest_file)
                    logger.info(f"Copied prompt template: {yaml_file.name}")

        # Save flow.yaml
        flow_yaml_path = ensure_within_directory(flow_dir, flow_dir / "flow.yaml")
        with open(str(flow_yaml_path.resolve()), "w") as f:
            f.write(flow_yaml)

        logger.info(f"Saved flow to: {flow_yaml_path}")

        # Save prompt templates if provided
        if prompt_templates:
            templates_data = json.loads(prompt_templates)
            for block_name, messages in templates_data.items():
                # Sanitize block_name for use in filename
                safe_block_name = (
                    sanitize_filename(f"{block_name}.yaml")
                    or f"prompt_{int(time.time())}.yaml"
                )
                template_yaml_path = ensure_within_directory(
                    flow_dir, flow_dir / safe_block_name
                )

                # Convert messages to YAML format
                template_yaml = yaml.dump(
                    messages, default_flow_style=False, allow_unicode=True
                )

                with open(str(template_yaml_path.resolve()), "w") as f:
                    f.write(template_yaml)

                logger.info(f"Saved prompt template: {template_yaml_path}")

        # Re-discover flows to include the new one
        FlowRegistry.discover_flows()

        return {
            "status": "success",
            "message": f"Flow '{flow_name}' created successfully",
            "flow_path": str(flow_yaml_path),
            "flow_dir": str(flow_dir),
        }

    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/flows/templates")
async def get_flow_templates():
    """Get all flows as templates that can be cloned and modified."""
    try:
        import yaml

        flow_templates = []

        # Scan all discovered flows
        flows = FlowRegistry.list_flows()

        for flow_info in flows:
            try:
                flow_name = flow_info["name"]
                flow_path = FlowRegistry.get_flow_path(flow_name)

                if not flow_path:
                    continue

                # Validate flow_path is within allowed directories
                try:
                    validated_path = resolve_flow_file(flow_path)
                except HTTPException:
                    logger.warning(
                        f"Skipping flow template {flow_name}: path validation failed"
                    )
                    continue

                # Read the flow YAML
                with open(validated_path, "r") as f:
                    flow_data = yaml.safe_load(f)

                # Get flow directory for resolving prompt paths
                flow_dir = validated_path.parent
                
                # Update blocks to include full prompt paths
                blocks = flow_data.get("blocks", [])
                for block in blocks:
                    block_config = block.get("block_config", {})
                    if "prompt_config_path" in block_config:
                        prompt_filename = block_config["prompt_config_path"]
                        # Construct full path relative to flow directory
                        full_prompt_path = str(flow_dir / prompt_filename)
                        block_config["full_prompt_path"] = full_prompt_path

                # Create template entry with full flow configuration
                template = {
                    "id": flow_info["id"],
                    "name": flow_name,
                    "flow_dir": str(flow_dir),
                    "metadata": flow_data.get("metadata", {}),
                    "blocks": blocks,
                    "num_blocks": len(blocks),
                    "tags": flow_data.get("metadata", {}).get("tags", []),
                    "description": flow_data.get("metadata", {}).get("description", ""),
                }

                flow_templates.append(template)

            except Exception as e:
                logger.warning(f"Could not load flow template {flow_info}: {e}")
                continue

        logger.info(f"Found {len(flow_templates)} flow templates")
        return {"templates": flow_templates}

    except Exception as e:
        logger.error(f"Error getting flow templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/flows/save-custom")
async def save_custom_flow(flow_data: Dict[str, Any]):
    """Save a custom flow to the custom_flows directory."""
    try:
        import shutil

        import yaml

        CUSTOM_FLOWS_DIR.mkdir(parents=True, exist_ok=True)

        # Create flow directory
        flow_name = os.path.basename(flow_data.get("metadata", {}).get("name", "unnamed_flow"))
        # Remove common suffixes like (Custom) and (Copy) before sanitizing
        base_flow_name = flow_name.replace(" (Custom)", "").replace(" (Copy)", "")
        safe_name = slugify_name(base_flow_name, prefix="flow")
        flow_dir = ensure_within_directory(
            CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / safe_name
        )
        logger.info(
            f"Flow directory: {flow_dir} (original name: {flow_name}, base name: {base_flow_name})"
        )
        flow_dir.mkdir(parents=True, exist_ok=True)

        # Get prompts from wizard (if user created/edited them)
        wizard_prompts = flow_data.get("prompts", {})

        # Get temp flow directory if prompts were saved there
        temp_flow_name = flow_data.get("temp_flow_name")
        temp_flow_dir = None
        if temp_flow_name:
            temp_slug = slugify_name(temp_flow_name, prefix="temp_flow")
            temp_flow_dir = ensure_within_directory(
                CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / temp_slug
            )
            logger.info(f"Temp flow directory for prompt copying: {temp_flow_dir}")
        else:
            logger.info("No temp flow directory provided")

        # Get source flow directory if this is a cloned flow
        source_flow_dir = None
        source_flow_id = flow_data.get("source_flow_id")
        source_flow_name = flow_data.get("source_flow_name")

        # Try source_flow_name first (for cloning predefined flows)
        # Use trusted whitelist to get source directory
        if source_flow_name:
            # Get from trusted registry whitelist - path comes from pre-built list, not user input
            source_flow_dir = get_trusted_flow_source_dir(source_flow_name)
            if source_flow_dir:
                logger.info(
                    f"Source flow directory from trusted registry (name '{source_flow_name}'): {source_flow_dir}"
                )

        # Fall back to source_flow_id if name didn't work
        if not source_flow_dir and source_flow_id:
            # Get from trusted registry whitelist - path comes from pre-built list, not user input
            source_flow_dir = get_trusted_flow_source_dir(source_flow_id)
            if source_flow_dir:
                logger.info(
                    f"Source flow directory from trusted registry (ID): {source_flow_dir}"
                )

        # Update block prompt_config_paths and save prompts
        blocks = flow_data.get("blocks", [])
        for block in blocks:
            if (
                "block_config" in block
                and "prompt_config_path" in block["block_config"]
            ):
                old_path = block["block_config"]["prompt_config_path"]

                # Extract just the filename from the old path
                old_path_obj = Path(old_path)
                prompt_filename = (
                    sanitize_filename(os.path.basename(old_path_obj.name))
                    or f"{block.get('block_config', {}).get('block_name', 'prompt')}.yaml"
                )

                # New path is just the filename (relative to flow directory)
                new_prompt_path = prompt_filename
                block["block_config"]["prompt_config_path"] = new_prompt_path

                new_prompt_file = ensure_within_directory(
                    flow_dir, flow_dir / prompt_filename
                )
                block_name = block.get("block_config", {}).get("block_name", "")

                # Check if user created/edited this prompt in the wizard
                if block_name and block_name in wizard_prompts:
                    # Save the wizard-created prompt
                    logger.info(f"Saving wizard prompt for block: {block_name}")
                    with open(new_prompt_file, "w") as f:
                        yaml.dump(
                            wizard_prompts[block_name], f, default_flow_style=False
                        )
                    logger.info(f"Saved wizard prompt: {new_prompt_file}")
                else:
                    # First check if prompt already exists in target flow directory (for editing existing flows)
                    if new_prompt_file.exists():
                        logger.info(
                            f"Prompt already exists in flow directory, skipping copy: {new_prompt_file}"
                        )
                        continue  # Skip to next block - prompt is already in place

                    # Use trusted whitelist approach to find source file
                    # Use paths from trusted registry only
                    trusted_source_path = None

                    # Check temp flow directory first (trusted - we control it)
                    if temp_flow_dir and temp_flow_dir.exists():
                        temp_prompt_file = (temp_flow_dir / prompt_filename).resolve()
                        # Verify temp file is within temp_flow_dir (no traversal)
                        if temp_prompt_file.exists() and is_path_within_allowed_dirs(
                            temp_prompt_file, [temp_flow_dir]
                        ):
                            trusted_source_path = temp_prompt_file
                            logger.info(f"Found prompt in temp flow: {trusted_source_path}")

                    # If not found in temp, look up in trusted flow registry
                    if not trusted_source_path:
                        trusted_paths = _get_trusted_flow_paths()
                        # Look for the prompt file by name in our trusted paths
                        for trusted_path in trusted_paths.values():
                            if trusted_path.name == prompt_filename and trusted_path.exists():
                                trusted_source_path = trusted_path
                                logger.info(f"Found prompt in trusted registry: {trusted_source_path}")
                                break

                    # Check source flow directory (already validated by get_trusted_flow_source_dir)
                    if not trusted_source_path and source_flow_dir:
                        # source_flow_dir comes from get_trusted_flow_source_dir() which already
                        # validates it's from the registry and within allowed directories.
                        # Search by scanning files in source_flow_dir and its parent
                        # instead of constructing paths from user-supplied old_path.
                        search_dirs = [source_flow_dir]
                        if source_flow_dir.parent and source_flow_dir.parent != source_flow_dir:
                            search_dirs.append(source_flow_dir.parent)
                        
                        for search_dir in search_dirs:
                            if not search_dir.exists():
                                continue
                            for yaml_file in search_dir.iterdir():
                                if (
                                    yaml_file.is_file()
                                    and yaml_file.name == prompt_filename
                                    and yaml_file.suffix in (".yaml", ".yml")
                                    and is_path_within_allowed_dirs(yaml_file.resolve(), ALLOWED_FLOW_READ_DIRS)
                                ):
                                    trusted_source_path = yaml_file.resolve()
                                    logger.info(f"Found prompt in source flow directory: {trusted_source_path}")
                                    break
                            if trusted_source_path:
                                break

                    if not trusted_source_path:
                        logger.warning(
                            f"Prompt source not found in trusted paths for block {block_name} ({prompt_filename})."
                        )

                    # Copy the file only if we found it in trusted paths
                    if trusted_source_path:
                        copy_validated_file(trusted_source_path, new_prompt_file)
                        logger.info(
                            f"Copied prompt file: {trusted_source_path} -> {new_prompt_file}"
                        )
                    else:
                        # Can't find source - this is an error
                        error_msg = f"CRITICAL: Could not find source prompt file: {old_path} (block: {block_name}, source_flow_id: {source_flow_id})"
                        logger.error(error_msg)
                        raise Exception(error_msg)

        # Save flow.yaml
        flow_path = ensure_within_directory(flow_dir, flow_dir / "flow.yaml")
        with open(flow_path, "w") as f:
            yaml.dump(flow_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved custom flow to: {flow_path}")

        # Auto-select the flow after saving so dataset loading works
        try:
            saved_flow = Flow.from_yaml(str(flow_path))
            current_config["flow"] = saved_flow
            current_config["flow_path"] = str(flow_path)
            current_config["model_config"] = {}
            current_config["dataset"] = None
            current_config["dataset_info"] = {}
            logger.info(f"Auto-selected saved custom flow: {flow_name}")
        except Exception as e:
            logger.warning(f"Could not auto-select saved flow: {e}")

        return {
            "status": "success",
            "flow_path": str(flow_path),
            "message": f"Custom flow '{flow_name}' saved successfully",
        }

    except Exception as e:
        logger.error(f"Error saving custom flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))
