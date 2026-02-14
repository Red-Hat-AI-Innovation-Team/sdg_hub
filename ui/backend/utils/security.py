# SPDX-License-Identifier: Apache-2.0
"""Security utilities: path validation, traversal protection, directory enforcement."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException

from config import (
    _WORKSPACE_ID_RE,
    ALLOWED_DATASET_DIRS,
    ALLOWED_FLOW_READ_DIRS,
    BASE_DIR,
    CUSTOM_FLOWS_DIR,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


def ensure_within_directory(base_dir: Path, target_path: Path) -> Path:
    """Ensure target_path resides within base_dir."""
    base_resolved = base_dir.resolve() if base_dir.exists() else base_dir.absolute()

    if target_path.exists():
        resolved = target_path.resolve()
    else:
        parent = target_path.parent
        name = target_path.name
        if parent.exists():
            resolved = parent.resolve() / name
        else:
            resolved = target_path.absolute()

    try:
        resolved.relative_to(base_resolved)
        return resolved
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Path '{target_path}' is outside allowed directory '{base_dir}'.",
        )


def validate_workspace_id(workspace_id: str) -> Path:
    """Validate a workspace ID and return a safe, resolved directory path.

    Ensures path safety by:
    1. Matching against a strict allowlist regex (no path separators)
    2. Applying os.path.basename() to strip any directory components
    3. Verifying the result is within CUSTOM_FLOWS_DIR
    """
    if not _WORKSPACE_ID_RE.match(workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace ID format")

    safe_id = os.path.basename(workspace_id)
    return ensure_within_directory(CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / safe_id)


def safe_join(base_dir: Path, filename: str) -> Path:
    """Safely join a directory with a user-supplied filename.

    Sanitizes input by applying os.path.basename() to strip directory
    components from the filename before joining.
    """
    safe_name = os.path.basename(filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return ensure_within_directory(base_dir, base_dir / safe_name)


def detect_path_traversal(path_str: str) -> bool:
    """Detect potential path traversal attempts in a path string.

    Returns True if path traversal patterns are detected, False otherwise.
    """
    if not path_str:
        return False

    normalized = os.path.normpath(path_str)

    traversal_patterns = ["..", "..\\", "../"]
    for pattern in traversal_patterns:
        if pattern in path_str or pattern in normalized:
            return True

    if normalized.startswith(".."):
        return True

    return False


def is_path_within_allowed_dirs(path: Path, allowed_dirs: List[Path]) -> bool:
    """Check if a path is within any of the allowed directories."""
    resolved = path.resolve() if path.exists() else path.absolute()
    for allowed_dir in allowed_dirs:
        try:
            resolved.relative_to(allowed_dir)
            return True
        except ValueError:
            continue
    return False


def resolve_dataset_file(path_str: str) -> Path:
    """Resolve dataset path and ensure it is under an allowed directory.

    Includes protection against path traversal attacks.
    """
    if detect_path_traversal(path_str):
        raise HTTPException(
            status_code=400, detail="Invalid path: path traversal detected."
        )

    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidates = [DATA_DIR / path_str, BASE_DIR / path_str]
    else:
        candidates = [candidate]

    for c in candidates:
        resolved = c.resolve()
        for allowed_dir in ALLOWED_DATASET_DIRS:
            if resolved == allowed_dir or allowed_dir in resolved.parents:
                if not resolved.exists():
                    raise HTTPException(
                        status_code=404, detail=f"Dataset file not found: {path_str}"
                    )
                return resolved

    allowed_text = ", ".join(str(d) for d in ALLOWED_DATASET_DIRS)
    raise HTTPException(
        status_code=400, detail=f"Datasets must reside within: {allowed_text}"
    )


def build_trusted_flow_source_dirs() -> Dict[str, Path]:
    """Build a whitelist of trusted flow source directories from the registry."""
    from sdg_hub import FlowRegistry

    trusted_dirs: Dict[str, Path] = {}

    for flow_info in FlowRegistry.list_flows():
        flow_name = flow_info["name"]
        flow_path = FlowRegistry.get_flow_path(flow_name)
        if flow_path:
            source_dir = Path(flow_path).parent.resolve()
            if is_path_within_allowed_dirs(source_dir, ALLOWED_FLOW_READ_DIRS):
                trusted_dirs[flow_name] = source_dir

    return trusted_dirs


def get_trusted_flow_source_dir(flow_name: str) -> Optional[Path]:
    """Get trusted source directory for a flow from the registry whitelist."""
    trusted_dirs = build_trusted_flow_source_dirs()
    return trusted_dirs.get(flow_name)


def _get_trusted_flow_paths() -> Dict[str, Path]:
    """Build a dictionary of trusted flow/prompt paths from FlowRegistry."""
    from sdg_hub import FlowRegistry

    trusted_paths: Dict[str, Path] = {}

    try:
        registered_flows = FlowRegistry.list_flows()
        for flow_name in registered_flows:
            flow_path = FlowRegistry.get_flow_path(flow_name)
            if flow_path:
                flow_path_obj = Path(flow_path).resolve()
                trusted_paths[flow_name] = flow_path_obj
                flow_dir = flow_path_obj.parent
                if flow_dir.exists():
                    for yaml_file in flow_dir.glob("*.yaml"):
                        resolved_yaml = yaml_file.resolve()
                        trusted_paths[str(resolved_yaml)] = resolved_yaml
                    for yml_file in flow_dir.glob("*.yml"):
                        resolved_yml = yml_file.resolve()
                        trusted_paths[str(resolved_yml)] = resolved_yml
    except Exception as e:
        logger.warning(f"Could not enumerate FlowRegistry flows: {e}")

    if CUSTOM_FLOWS_DIR.exists():
        for flow_subdir in CUSTOM_FLOWS_DIR.iterdir():
            if flow_subdir.is_dir():
                for yaml_file in flow_subdir.glob("*.yaml"):
                    resolved_yaml = yaml_file.resolve()
                    trusted_paths[str(resolved_yaml)] = resolved_yaml
                for yml_file in flow_subdir.glob("*.yml"):
                    resolved_yml = yml_file.resolve()
                    trusted_paths[str(resolved_yml)] = resolved_yml

    return trusted_paths


def resolve_flow_file(path_str: str, must_exist: bool = True) -> Path:
    """Resolve flow file path and ensure it is under an allowed directory."""
    if detect_path_traversal(path_str):
        raise HTTPException(
            status_code=400, detail="Invalid path: path traversal detected."
        )

    candidate = Path(path_str)
    if not candidate.is_absolute():
        for base_dir in ALLOWED_FLOW_READ_DIRS:
            potential = base_dir / path_str
            if potential.exists():
                candidate = potential
                break
        else:
            candidate = CUSTOM_FLOWS_DIR / path_str

    resolved = candidate.resolve() if candidate.exists() else candidate.absolute()

    if not is_path_within_allowed_dirs(resolved, ALLOWED_FLOW_READ_DIRS):
        raise HTTPException(
            status_code=400, detail="Flow file must reside within allowed directories."
        )

    if must_exist and not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Flow file not found: {path_str}")

    return resolved


def resolve_prompt_file(path_str: str, flow_dir: Optional[Path] = None) -> Path:
    """Resolve prompt file path and ensure it is under an allowed directory."""
    if detect_path_traversal(path_str):
        raise HTTPException(
            status_code=400, detail="Invalid path: path traversal detected."
        )

    candidate = Path(path_str)

    if not candidate.is_absolute() and flow_dir:
        potential = flow_dir / candidate.name
        if potential.exists():
            candidate = potential

    if not candidate.exists():
        for base_dir in ALLOWED_FLOW_READ_DIRS:
            for yaml_file in base_dir.rglob(candidate.name):
                candidate = yaml_file
                break
            if candidate.exists():
                break

    if not candidate.exists():
        raise HTTPException(
            status_code=404, detail=f"Prompt file not found: {path_str}"
        )

    resolved = candidate.resolve()

    if not is_path_within_allowed_dirs(resolved, ALLOWED_FLOW_READ_DIRS):
        raise HTTPException(
            status_code=400,
            detail="Prompt file must reside within allowed directories.",
        )

    return resolved
