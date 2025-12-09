#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SDG Hub API Server

FastAPI server that exposes sdg_hub functionality for the UI.
Provides endpoints for flow discovery, model configuration, and dataset management.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import multiprocessing
import os
import queue
import re
import time

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import nest_asyncio
import pandas as pd
import uvicorn

from sdg_hub import BlockRegistry, Flow, FlowRegistry

# Configure logging with DEBUG level for troubleshooting
logging.getLogger("uvicorn").setLevel(logging.DEBUG)
logging.getLogger("multipart").setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
    logger.info(f"🔐 Loaded environment variables from: {env_file}")
else:
    logger.info("ℹ️  No .env file found. Using system environment variables only.")

# Note: nest_asyncio.apply() is called conditionally in the dry_run endpoint
# to avoid conflicts with uvloop when reload=True


# ============================================================================
# Paths & Storage Helpers
# ============================================================================

BASE_DIR = Path(__file__).parent

# Support isolated data directories via environment variable
# This allows running multiple instances with separate data (useful for demos)
DATA_DIR_NAME = os.getenv("SDG_HUB_DATA_DIR", "")
if DATA_DIR_NAME:
    DATA_DIR = (BASE_DIR / DATA_DIR_NAME).resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Using isolated data directory: {DATA_DIR}")
else:
    DATA_DIR = BASE_DIR

UPLOADS_DIR = (DATA_DIR / "uploads").resolve()
CUSTOM_FLOWS_DIR = (DATA_DIR / "custom_flows").resolve()
SAVED_CONFIG_FILE = (DATA_DIR / "saved_configurations.json").resolve()
CHECKPOINTS_DIR = (DATA_DIR / "checkpoints").resolve()

# Ensure required directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_FLOWS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

FILENAME_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]")
MAX_UPLOAD_SIZE_MB = int(os.getenv("SDG_HUB_MAX_UPLOAD_MB", "512"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

ALLOWED_DATASET_DIRS: List[Path] = [UPLOADS_DIR]
extra_dirs = os.getenv("SDG_HUB_ALLOWED_DATA_DIRS", "")
if extra_dirs:
    for raw_dir in extra_dirs.split(os.pathsep):
        candidate = raw_dir.strip()
        if not candidate:
            continue
        resolved_dir = Path(candidate).expanduser().resolve()
        try:
            resolved_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(
                f"⚠️ Could not prepare dataset directory '{candidate}': {exc}"
            )
            continue
        ALLOWED_DATASET_DIRS.append(resolved_dir)
        logger.info(f"✅ Added allowed dataset directory: {resolved_dir}")


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename."""
    if not filename:
        return ""
    basename = os.path.basename(filename)
    sanitized = FILENAME_SANITIZER.sub("_", basename)
    return sanitized.strip("._")


def slugify_name(name: str, prefix: str = "flow") -> str:
    """Generate a directory-safe slug for flows/prompts."""
    base = name or ""
    slug = FILENAME_SANITIZER.sub("_", base.lower())
    slug = slug.strip("_") or f"{prefix}_{int(time.time())}"
    return slug


def ensure_within_directory(base_dir: Path, target_path: Path) -> Path:
    """Ensure target_path resides within base_dir."""
    # Ensure base_dir is resolved
    base_resolved = base_dir.resolve() if base_dir.exists() else base_dir.absolute()

    # Resolve target_path - for non-existent files, resolve the parent then join
    if target_path.exists():
        resolved = target_path.resolve()
    else:
        # For non-existent files, resolve what we can and then join
        parent = target_path.parent
        name = target_path.name
        if parent.exists():
            resolved = parent.resolve() / name
        else:
            # Fall back to absolute path
            resolved = target_path.absolute()

    # Check if the resolved path is within the base directory
    # Use string comparison of resolved paths to handle edge cases
    try:
        resolved.relative_to(base_resolved)
        return resolved
    except ValueError:
        # Not relative to base_dir
        raise HTTPException(
            status_code=400,
            detail=f"Path '{target_path}' is outside allowed directory '{base_dir}'.",
        )


def detect_path_traversal(path_str: str) -> bool:
    """Detect potential path traversal attempts in a path string.

    Returns True if path traversal patterns are detected, False otherwise.
    """
    if not path_str:
        return False

    # Normalize the path to detect hidden traversal attempts
    normalized = os.path.normpath(path_str)

    # Check for common path traversal patterns
    traversal_patterns = ["..", "..\\", "../"]
    for pattern in traversal_patterns:
        if pattern in path_str or pattern in normalized:
            return True

    # Check if normpath changed the path to go outside (starts with ..)
    if normalized.startswith(".."):
        return True

    return False


def resolve_dataset_file(path_str: str) -> Path:
    """Resolve dataset path and ensure it is under an allowed directory.

    Includes protection against path traversal attacks.
    """
    # First, check for path traversal attempts in the raw input
    if detect_path_traversal(path_str):
        raise HTTPException(
            status_code=400, detail="Invalid path: path traversal detected."
        )

    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = BASE_DIR / path_str
    resolved = candidate.resolve()
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


# Get SDG Hub flows directory (for reading predefined flows)
SDG_HUB_FLOWS_DIR = (
    Path(__file__).parent.parent.parent / "src" / "sdg_hub" / "flows"
).resolve()

# Allowed directories for reading flow files
ALLOWED_FLOW_READ_DIRS: List[Path] = [CUSTOM_FLOWS_DIR, SDG_HUB_FLOWS_DIR]


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


def resolve_flow_file(path_str: str, must_exist: bool = True) -> Path:
    """Resolve flow file path and ensure it is under an allowed directory.

    Includes protection against path traversal attacks.
    """
    # First, check for path traversal attempts in the raw input
    if detect_path_traversal(path_str):
        raise HTTPException(
            status_code=400, detail="Invalid path: path traversal detected."
        )

    candidate = Path(path_str)
    if not candidate.is_absolute():
        # Try relative to custom flows first, then sdg_hub flows
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
    """Resolve prompt file path and ensure it is under an allowed directory.

    Includes protection against path traversal attacks.

    Args:
        path_str: The prompt file path (can be relative or absolute)
        flow_dir: Optional flow directory to resolve relative paths against
    """
    # First, check for path traversal attempts in the raw input
    if detect_path_traversal(path_str):
        raise HTTPException(
            status_code=400, detail="Invalid path: path traversal detected."
        )

    candidate = Path(path_str)

    # If relative path and flow_dir provided, try there first
    if not candidate.is_absolute() and flow_dir:
        potential = flow_dir / candidate.name
        if potential.exists():
            candidate = potential

    # If still not found, search in allowed directories
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


# ============================================================================
# Checkpoint Utilities
# ============================================================================


def get_checkpoint_dir_for_config(config_id: str) -> Path:
    """Get the checkpoint directory path for a specific configuration.

    Validates that config_id doesn't contain path traversal sequences.
    """
    # Sanitize config_id to prevent path traversal
    safe_config_id = sanitize_filename(config_id)
    if not safe_config_id:
        raise HTTPException(status_code=400, detail="Invalid configuration ID")

    checkpoint_dir = CHECKPOINTS_DIR / safe_config_id
    # Verify the resulting path is within CHECKPOINTS_DIR
    return ensure_within_directory(CHECKPOINTS_DIR, checkpoint_dir)


def get_checkpoint_info(config_id: str) -> Dict[str, Any]:
    """Get information about existing checkpoints for a configuration.

    Returns:
        Dict with:
        - has_checkpoints: bool
        - checkpoint_count: int
        - samples_completed: int (estimated from checkpoint files)
        - last_checkpoint_time: str (ISO format) or None
        - checkpoint_dir: str
    """
    checkpoint_dir = get_checkpoint_dir_for_config(config_id)

    if not checkpoint_dir.exists():
        return {
            "has_checkpoints": False,
            "checkpoint_count": 0,
            "samples_completed": 0,
            "last_checkpoint_time": None,
            "checkpoint_dir": str(checkpoint_dir),
        }

    # Find checkpoint files (format: checkpoint_NNNN.jsonl)
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.jsonl"))

    if not checkpoint_files:
        return {
            "has_checkpoints": False,
            "checkpoint_count": 0,
            "samples_completed": 0,
            "last_checkpoint_time": None,
            "checkpoint_dir": str(checkpoint_dir),
        }

    # Count total samples from all checkpoint files
    total_samples = 0
    for cp_file in checkpoint_files:
        try:
            # Count lines in JSONL file
            with open(cp_file, "r") as f:
                total_samples += sum(1 for _ in f)
        except Exception:
            pass

    # Get last modification time
    last_checkpoint = checkpoint_files[-1]
    last_modified = last_checkpoint.stat().st_mtime
    from datetime import datetime

    last_checkpoint_time = datetime.fromtimestamp(last_modified).isoformat()

    return {
        "has_checkpoints": True,
        "checkpoint_count": len(checkpoint_files),
        "samples_completed": total_samples,
        "last_checkpoint_time": last_checkpoint_time,
        "checkpoint_dir": str(checkpoint_dir),
    }


def clear_checkpoints(config_id: str) -> bool:
    """Clear all checkpoints for a configuration."""
    checkpoint_dir = get_checkpoint_dir_for_config(config_id)

    if not checkpoint_dir.exists():
        return True

    try:
        import shutil

        shutil.rmtree(checkpoint_dir)
        return True
    except Exception as e:
        logger.error(f"Failed to clear checkpoints for {config_id}: {e}")
        return False


# ============================================================================
# Security Utilities
# ============================================================================


def mask_api_key(key: Optional[str]) -> str:
    """
    Mask an API key for safe display.

    Args:
        key: The API key to mask

    Returns:
        Masked version of the key
    """
    if not key:
        return ""
    if key == "EMPTY":
        return "EMPTY"
    # Show first 4 and last 4 characters if key is long enough
    if len(key) > 8:
        return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"
    else:
        return "*" * len(key)


def sanitize_model_config(
    config: Dict[str, Any], mask_key: bool = True
) -> Dict[str, Any]:
    """
    Remove or mask sensitive information from model configuration.

    Args:
        config: The model configuration dictionary
        mask_key: If True, mask the API key; if False, remove it entirely

    Returns:
        Sanitized configuration
    """
    if not config:
        return {}

    sanitized = config.copy()

    if "api_key" in sanitized:
        api_key = sanitized["api_key"]

        # Check if API key is a safe value that can be stored
        is_safe_value = (
            api_key in ["EMPTY", "NONE", ""]  # Special test values
            or api_key.startswith("env:")  # Environment variable references
        )

        if is_safe_value:
            # Keep safe values as-is
            pass
        elif mask_key:
            # Mask actual API keys for display
            sanitized["api_key"] = mask_api_key(api_key)
        else:
            # Remove actual API keys for storage
            del sanitized["api_key"]

    return sanitized


def resolve_env_variable(value: str) -> Optional[str]:
    """
    Resolve environment variable references.

    Args:
        value: A string that might be an env var reference like "env:OPENAI_API_KEY"

    Returns:
        The resolved value, or None if env var not found
    """
    if not value:
        return value

    if value.startswith("env:"):
        env_var_name = value[4:]  # Remove "env:" prefix
        import os

        env_value = os.getenv(env_var_name)
        if env_value:
            logger.info(f"✅ Resolved environment variable: {env_var_name}")
            return env_value
        else:
            logger.warning(f"⚠️ Environment variable not found: {env_var_name}")
            return None

    return value


def get_safe_api_key(config: Dict[str, Any]) -> Optional[str]:
    """
    Get API key from config, resolving env vars if needed.

    Args:
        config: Model configuration dictionary

    Returns:
        The actual API key to use
    """
    api_key = config.get("api_key")
    if not api_key:
        return None

    # Resolve environment variable if referenced
    return resolve_env_variable(api_key)


def validate_api_key_format(
    api_key: str, provider: Optional[str] = None
) -> tuple[bool, Optional[str]]:
    """
    Validate API key format (not functionality).

    Args:
        api_key: The API key to validate
        provider: Optional provider hint (openai, anthropic, etc.)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is required"

    # Allow special values
    if api_key in ["EMPTY", "NONE"]:
        return True, None

    # Allow environment variable references
    if api_key.startswith("env:"):
        env_var_name = api_key[4:]
        if not env_var_name:
            return False, "Environment variable name cannot be empty"
        if not env_var_name.replace("_", "").isalnum():
            return False, "Invalid environment variable name"
        return True, None

    # Basic format validation
    if len(api_key) < 8:
        return False, "API key too short (minimum 8 characters)"

    if len(api_key) > 512:
        return False, "API key too long (maximum 512 characters)"

    # Provider-specific validation
    if provider:
        provider_lower = provider.lower()

        if "openai" in provider_lower:
            if not (api_key.startswith("sk-") or api_key.startswith("sess-")):
                return False, "OpenAI keys should start with 'sk-' or 'sess-'"

        elif "anthropic" in provider_lower:
            if not api_key.startswith("sk-ant-"):
                return False, "Anthropic keys should start with 'sk-ant-'"

        elif "cohere" in provider_lower:
            if len(api_key) < 40:
                return False, "Cohere keys are typically longer"

    # Warn about suspicious patterns
    if api_key in ["your-api-key", "your-key-here", "test", "example"]:
        return False, "Please replace placeholder with actual API key"

    return True, None


# Initialize FastAPI app
app = FastAPI(
    title="SDG Hub API",
    description="API for SDG Hub synthetic data generation configuration and execution",
    version="1.0.0",
)

# Configure CORS
# Allow multiple frontend ports for running parallel demo instances
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server (default)
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://localhost:3001",  # Demo instance 1
        "http://127.0.0.1:3001",
        "http://localhost:3002",  # Demo instance 2
        "http://127.0.0.1:3002",
        "http://localhost:3003",  # Demo instance 3
        "http://127.0.0.1:3003",
        "http://localhost:3004",  # Demo instance 4
        "http://127.0.0.1:3004",
        "http://localhost:3005",  # Demo instance 5
        "http://127.0.0.1:3005",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state for current configuration
current_config = {
    "flow": None,
    "flow_path": None,
    "model_config": {},
    "dataset": None,
    "dataset_info": {},
    "dataset_load_params": None,
}

# Generation control
generation_cancel_flag = {"should_cancel": False}
active_generation_process = {"pid": None, "config_id": None}

# Global log queues per config_id for reconnection support
# Maps config_id -> {"queue": multiprocessing.Queue, "process": Process, "start_time": timestamp}
active_generations = {}


def generation_worker(
    log_queue,
    flow_path,
    model_config,
    dataset_params,
    max_concurrency,
    log_dir,
    checkpoint_dir=None,
    save_freq=None,
    resume_from_checkpoint=False,
):
    """Worker process for running flow generation.

    Args:
        log_queue: Queue for sending logs back to main process
        flow_path: Path to the flow YAML file
        model_config: Model configuration dict
        dataset_params: Dataset parameters dict
        max_concurrency: Maximum concurrent requests
        log_dir: Directory for logs
        checkpoint_dir: Directory for saving checkpoints (optional)
        save_freq: Save checkpoint every N samples (optional)
        resume_from_checkpoint: If True, resume from existing checkpoints
    """
    try:
        # Redirect stdout/stderr to queue
        import sys
        import time

        class TeeOutput:
            def __init__(self, queue):
                self.queue = queue

            def write(self, text):
                if text:
                    self.queue.put(
                        {"type": "log", "message": text, "timestamp": time.time()}
                    )

            def flush(self):
                pass

        sys.stdout = TeeOutput(log_queue)
        sys.stderr = TeeOutput(log_queue)

        # Load flow
        from sdg_hub import Flow

        flow = Flow.from_yaml(flow_path)

        # Apply model configuration
        if model_config:
            # Re-apply configuration logic
            kwargs = {}
            if model_config.get("model"):
                kwargs["model"] = model_config["model"]
            if model_config.get("api_base"):
                kwargs["api_base"] = model_config["api_base"]
            if model_config.get("api_key"):
                kwargs["api_key"] = model_config["api_key"]
            if model_config.get("additional_params"):
                kwargs.update(model_config["additional_params"])

            if kwargs:
                flow.set_model_config(**kwargs)

        # Load dataset as pandas DataFrame for optimal performance
        from pathlib import Path

        import pandas as pd

        if dataset_params:
            data_files = dataset_params["data_files"]
            file_path = Path(data_files)
            file_format = dataset_params.get("file_format", "auto")

            # Auto-detect format from extension if needed
            if file_format == "auto":
                suffix = file_path.suffix.lower()
                format_map = {
                    ".jsonl": "jsonl",
                    ".json": "json",
                    ".csv": "csv",
                    ".parquet": "parquet",
                    ".pq": "parquet",
                }
                file_format = format_map.get(suffix, "jsonl")

            # Load as pandas DataFrame
            if file_format == "jsonl":
                df = pd.read_json(data_files, lines=True)
            elif file_format == "json":
                df = pd.read_json(data_files)
            elif file_format == "csv":
                csv_delimiter = dataset_params.get("csv_delimiter", ",")
                csv_encoding = dataset_params.get("csv_encoding", "utf-8")
                df = pd.read_csv(
                    data_files, delimiter=csv_delimiter, encoding=csv_encoding
                )
            elif file_format == "parquet":
                df = pd.read_parquet(data_files)
            else:
                df = pd.read_json(data_files, lines=True)

            # Apply shuffle if requested
            if dataset_params.get("shuffle"):
                df = df.sample(
                    frac=1, random_state=dataset_params.get("seed", 42)
                ).reset_index(drop=True)

            # Limit samples if specified
            if dataset_params.get("num_samples"):
                df = df.head(min(dataset_params["num_samples"], len(df)))
        else:
            log_queue.put(
                {"type": "error", "message": "No dataset parameters provided"}
            )
            return

        # Clear checkpoints if not resuming (starting fresh)
        if checkpoint_dir and not resume_from_checkpoint:
            import shutil

            checkpoint_path = Path(checkpoint_dir)
            # Validate checkpoint_path is within CHECKPOINTS_DIR before removing
            try:
                checkpoint_path.resolve().relative_to(CHECKPOINTS_DIR.resolve())
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
            except ValueError:
                log_queue.put(
                    {"type": "error", "message": "Invalid checkpoint directory"}
                )
                log_queue.put(
                    {
                        "type": "log",
                        "message": "🗑️ Cleared existing checkpoints for fresh start\n",
                        "timestamp": time.time(),
                    }
                )

        # Prepare checkpoint parameters
        generate_kwargs = {"max_concurrency": max_concurrency, "log_dir": log_dir}

        if checkpoint_dir:
            generate_kwargs["checkpoint_dir"] = checkpoint_dir
            if save_freq:
                generate_kwargs["save_freq"] = save_freq

            if resume_from_checkpoint:
                log_queue.put(
                    {
                        "type": "log",
                        "message": "📂 Resuming from checkpoint...\n",
                        "timestamp": time.time(),
                    }
                )
            else:
                log_queue.put(
                    {
                        "type": "log",
                        "message": f"💾 Checkpointing enabled (save every {save_freq or 'completion'} samples)\n",
                        "timestamp": time.time(),
                    }
                )

        # Run generation with pandas DataFrame
        generated_df = flow.generate(df, **generate_kwargs)

        # Convert result to list for pickling back (pandas DataFrame)
        # Handle both pandas DataFrame and HuggingFace Dataset returns
        if hasattr(generated_df, "to_dict"):
            # pandas DataFrame
            dataset_list = generated_df.to_dict(orient="records")
            column_names = generated_df.columns.tolist()
        else:
            # HuggingFace Dataset (fallback for backward compatibility)
            dataset_list = generated_df.to_list()
            column_names = list(generated_df.column_names)

        log_queue.put(
            {
                "type": "result",
                "dataset_list": dataset_list,
                "column_names": column_names,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        log_queue.put({"type": "error", "message": str(e)})


# Run history storage
RUNS_HISTORY_FILE = Path("runs_history.json")


def load_runs_history():
    """Load runs history from file."""
    if RUNS_HISTORY_FILE.exists():
        try:
            with open(RUNS_HISTORY_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []
    return []


def save_runs_history(runs):
    """Save runs history to file."""
    with open(RUNS_HISTORY_FILE, "w") as f:
        json.dump(runs, f, indent=2)


# ============================================================================
# Request/Response Models
# ============================================================================


class FlowSearchRequest(BaseModel):
    """Request model for flow search."""

    tag: Optional[str] = None
    name_filter: Optional[str] = None


class FlowInfo(BaseModel):
    """Flow information model."""

    name: str
    id: str
    path: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = []
    recommended_models: Optional[Dict[str, Any]] = None
    dataset_requirements: Optional[Dict[str, Any]] = None


class ModelConfigRequest(BaseModel):
    """Model configuration request."""

    model: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    blocks: Optional[List[str]] = None
    additional_params: Optional[Dict[str, Any]] = {}


class DatasetFormat(str, Enum):
    """Supported dataset file formats."""

    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AUTO = "auto"  # Auto-detect from file extension


class DatasetLoadRequest(BaseModel):
    """Dataset loading request with pandas support."""

    data_files: str
    file_format: DatasetFormat = DatasetFormat.AUTO
    num_samples: Optional[int] = None
    shuffle: bool = False
    seed: int = 42
    # CSV-specific options
    csv_delimiter: str = ","
    csv_encoding: str = "utf-8"


class DryRunRequest(BaseModel):
    """Dry run request."""

    sample_size: int = 2
    enable_time_estimation: bool = False
    max_concurrency: Optional[int] = None


class FlowRunRecord(BaseModel):
    """Flow run record for history tracking."""

    run_id: str
    config_id: str
    flow_name: str
    flow_type: str  # 'existing' or 'custom'
    model_name: str
    status: str  # 'running', 'completed', 'failed'
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    input_samples: int
    output_samples: Optional[int] = None
    output_columns: Optional[int] = None
    dataset_file: Optional[str] = None
    output_file: Optional[str] = None  # Path to generated JSONL file
    error_message: Optional[str] = None


# ============================================================================
# Startup Event
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize registries on startup."""
    logger.info("Starting SDG Hub API Server...")
    try:
        # Ensure working directories exist
        CUSTOM_FLOWS_DIR.mkdir(parents=True, exist_ok=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

        # Add custom flows directory to Python path so FlowRegistry can discover it
        import sys

        custom_flows_path = str(CUSTOM_FLOWS_DIR)
        if custom_flows_path not in sys.path:
            sys.path.insert(0, custom_flows_path)

        # Discover flows and blocks
        FlowRegistry.discover_flows()
        BlockRegistry.discover_blocks()
        load_saved_configurations_from_disk()
        logger.info("✅ Successfully discovered flows and blocks")
        logger.info(f"📁 Custom flows directory: {CUSTOM_FLOWS_DIR}")
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "sdg_hub_api"}


# ============================================================================
# Flow Discovery Endpoints
# ============================================================================


@app.get("/api/flows/list", response_model=List[str])
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


@app.post("/api/flows/search", response_model=List[str])
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


@app.get("/api/flows/{flow_name:path}/info", response_model=FlowInfo)
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


@app.post("/api/flows/select-by-path")
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


@app.get("/api/flows/{flow_name:path}/yaml")
async def get_flow_yaml(flow_name: str):
    """Get the raw YAML content of a flow for cloning."""
    try:
        import yaml

        # Check if this is a custom flow
        is_custom = flow_name.endswith(" (Custom)")
        actual_flow_name = (
            flow_name.replace(" (Custom)", "") if is_custom else flow_name
        )

        # Get flow path
        flow_path = FlowRegistry.get_flow_path(actual_flow_name)

        # If not found in registry and is custom, check custom_flows directory
        if not flow_path and is_custom:
            flow_dir_name = slugify_name(actual_flow_name, prefix="flow")
            custom_flow_path = ensure_within_directory(
                CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / flow_dir_name / "flow.yaml"
            )

            if custom_flow_path.exists():
                flow_path = str(custom_flow_path)
            else:
                # Try to find by scanning within CUSTOM_FLOWS_DIR only
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
                                    if (
                                        flow_data.get("metadata", {}).get("name")
                                        == actual_flow_name
                                    ):
                                        flow_path = str(validated_path)
                                        break

        if not flow_path:
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")

        # Validate flow_path is within allowed directories before reading
        validated_flow_path = resolve_flow_file(flow_path)

        # Read and parse the YAML file
        with open(validated_flow_path, "r") as f:
            flow_data = yaml.safe_load(f)

        logger.info(f"Retrieved YAML for flow: {flow_name}")
        return flow_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting flow YAML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/flows/{flow_name:path}/select")
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


# ============================================================================
# Model Configuration Endpoints
# ============================================================================


@app.get("/api/model/recommendations")
async def get_model_recommendations():
    """Get model recommendations for the selected flow."""
    try:
        if not current_config["flow"]:
            raise HTTPException(status_code=400, detail="No flow selected")

        flow = current_config["flow"]
        recommendations = flow.get_model_recommendations()
        default_model = flow.get_default_model()

        return {
            "default_model": default_model,
            "recommendations": recommendations,
            "requires_config": flow.is_model_config_required(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/configure")
async def configure_model(config: ModelConfigRequest):
    """Configure model settings for the selected flow."""
    try:
        if not current_config["flow"]:
            raise HTTPException(status_code=400, detail="No flow selected")

        flow = current_config["flow"]

        # Validate API key format
        if config.api_key:
            is_valid, error_msg = validate_api_key_format(config.api_key, config.model)
            if not is_valid:
                raise HTTPException(
                    status_code=400, detail=f"Invalid API key: {error_msg}"
                )

        # Build kwargs from config
        kwargs = {}
        if config.model:
            kwargs["model"] = config.model
        if config.api_base:
            kwargs["api_base"] = config.api_base
        if config.api_key:
            # Resolve environment variable if referenced
            resolved_key = resolve_env_variable(config.api_key)
            if resolved_key is None and config.api_key.startswith("env:"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Environment variable not found: {config.api_key[4:]}",
                )
            kwargs["api_key"] = resolved_key
        if config.blocks:
            kwargs["blocks"] = config.blocks

        # Add any additional parameters
        if config.additional_params:
            kwargs.update(config.additional_params)

        # Apply configuration
        flow.set_model_config(**kwargs)

        # Store config (keep the original reference, not resolved value)
        current_config["model_config"] = config.model_dump()

        logger.info(f"🔐 Model configured: {config.model} (API key validated)")

        # Return sanitized config (mask API key)
        safe_config = sanitize_model_config(
            current_config["model_config"], mask_key=True
        )

        return {
            "status": "success",
            "message": "Model configuration applied",
            "config": safe_config,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Dataset Management Endpoints
# ============================================================================

# Supported dataset file extensions
SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".csv", ".parquet", ".pq"}


@app.post("/api/dataset/upload")
async def upload_dataset_file(request: Request):
    """Upload a dataset file and save it temporarily.

    Only accepts supported formats: JSONL, JSON, CSV, Parquet.
    Manually parses multipart form data to avoid FastAPI parsing issues.
    """
    try:
        # Log request details for debugging
        content_type = request.headers.get("content-type", "")
        logger.info("=== UPLOAD REQUEST DEBUG ===")
        logger.info(f"Content-Type: {content_type}")
        logger.info(f"Content-Length: {request.headers.get('content-length')}")

        # Read the entire body first to avoid async context issues with large files
        body = await request.body()
        logger.info(f"Body size: {len(body)} bytes")

        # Parse the multipart data manually
        # Extract boundary from content-type
        boundary = None
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[9:].strip('"')
                break

        if not boundary:
            raise HTTPException(
                status_code=400, detail="Missing boundary in content-type"
            )

        # Simple multipart parser for single file upload
        boundary_bytes = f"--{boundary}".encode()
        parts = body.split(boundary_bytes)

        file_data = None
        filename = None

        for part in parts:
            if b"Content-Disposition" not in part:
                continue

            # Split headers from content
            if b"\r\n\r\n" in part:
                headers_section, content = part.split(b"\r\n\r\n", 1)
            elif b"\n\n" in part:
                headers_section, content = part.split(b"\n\n", 1)
            else:
                continue

            headers_str = headers_section.decode("utf-8", errors="ignore")

            # Check if this is the file field
            if 'name="file"' in headers_str:
                # Extract filename
                import re as regex

                filename_match = regex.search(r'filename="([^"]+)"', headers_str)
                if filename_match:
                    filename = filename_match.group(1)

                # Remove trailing boundary markers
                if content.endswith(b"--\r\n"):
                    content = content[:-4]
                elif content.endswith(b"--\n"):
                    content = content[:-3]
                elif content.endswith(b"\r\n"):
                    content = content[:-2]
                elif content.endswith(b"\n"):
                    content = content[:-1]

                file_data = content
                break

        if file_data is None or filename is None:
            raise HTTPException(status_code=400, detail="No file found in upload")

        logger.info(f"Parsed filename: {filename}")
        logger.info(f"File data size: {len(file_data)} bytes")
        logger.info("=== END DEBUG ===")

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        safe_filename = sanitize_filename(filename)
        if not safe_filename:
            raise HTTPException(status_code=400, detail="Invalid filename provided.")

        # Validate file format
        file_extension = Path(safe_filename).suffix.lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: '{file_extension}'. Please upload a dataset in one of these formats: JSONL (.jsonl), JSON (.json), CSV (.csv), or Parquet (.parquet)",
            )

        file_path = ensure_within_directory(UPLOADS_DIR, UPLOADS_DIR / safe_filename)
        bytes_written = len(file_data)

        if bytes_written > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds max upload size of {MAX_UPLOAD_SIZE_MB} MB.",
            )

        try:
            with open(file_path, "wb") as destination:
                destination.write(file_data)
        except Exception:
            if file_path.exists():
                file_path.unlink(missing_ok=True)
            raise

        relative_path = Path("uploads") / safe_filename
        logger.info(
            f"📊 Uploaded dataset file: {file_path} ({bytes_written} bytes, format: {file_extension})"
        )

        return {
            "status": "success",
            "message": f"File '{filename}' uploaded successfully",
            "file_path": str(relative_path),
            "file_size": bytes_written,
            "format": file_extension[1:],  # Remove the dot
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def detect_file_format(file_path: Path) -> DatasetFormat:
    """Auto-detect file format from extension.

    Raises HTTPException if format is not supported.
    """
    suffix = file_path.suffix.lower()
    format_map = {
        ".jsonl": DatasetFormat.JSONL,
        ".json": DatasetFormat.JSON,
        ".csv": DatasetFormat.CSV,
        ".parquet": DatasetFormat.PARQUET,
        ".pq": DatasetFormat.PARQUET,
    }

    if suffix not in format_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: '{suffix}'. Please upload a dataset in one of these formats: JSONL (.jsonl), JSON (.json), CSV (.csv), or Parquet (.parquet)",
        )

    return format_map[suffix]


def is_supported_format(file_path: Path) -> bool:
    """Check if file has a supported format."""
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_dataset_as_pandas(
    file_path: Path,
    file_format: DatasetFormat,
    csv_delimiter: str = ",",
    csv_encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load dataset file as pandas DataFrame.

    Supports JSONL, JSON, CSV, and Parquet formats for optimal performance.
    """
    if file_format == DatasetFormat.AUTO:
        file_format = detect_file_format(file_path)

    logger.info(f"Loading dataset as pandas DataFrame (format: {file_format.value})")

    if file_format == DatasetFormat.JSONL:
        # JSONL: one JSON object per line
        df = pd.read_json(file_path, lines=True)
    elif file_format == DatasetFormat.JSON:
        # Regular JSON (array of objects or object with arrays)
        df = pd.read_json(file_path)
    elif file_format == DatasetFormat.CSV:
        df = pd.read_csv(file_path, delimiter=csv_delimiter, encoding=csv_encoding)
    elif file_format == DatasetFormat.PARQUET:
        df = pd.read_parquet(file_path)
    else:
        # Default to JSONL
        df = pd.read_json(file_path, lines=True)

    return df


@app.post("/api/dataset/load")
async def load_dataset_from_file(request: DatasetLoadRequest):
    """Load dataset from file using pandas for optimal performance.

    Supports multiple formats: JSONL, JSON, CSV, Parquet.
    """
    try:
        if not current_config["flow"]:
            raise HTTPException(status_code=400, detail="No flow selected")

        safe_dataset_path = resolve_dataset_file(request.data_files)

        # Load dataset as pandas DataFrame
        df = load_dataset_as_pandas(
            safe_dataset_path,
            request.file_format,
            request.csv_delimiter,
            request.csv_encoding,
        )

        # Apply shuffle if requested
        if request.shuffle:
            df = df.sample(frac=1, random_state=request.seed).reset_index(drop=True)

        # Limit samples if specified
        if request.num_samples:
            df = df.head(min(request.num_samples, len(df)))

        # Store dataset as pandas DataFrame
        current_config["dataset"] = df
        current_config["dataset_info"] = {
            "num_samples": len(df),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        # Store loading parameters for worker process reconstruction
        current_config["dataset_load_params"] = {
            **request.model_dump(),
            "data_files": str(safe_dataset_path),
        }

        logger.info(
            f"📊 Loaded dataset from {safe_dataset_path}: {len(df)} samples, {len(df.columns)} columns (pandas DataFrame)"
        )

        return {
            "status": "success",
            "message": f"Dataset loaded with {len(df)} samples",
            "dataset_info": current_config["dataset_info"],
            "format": request.file_format.value,
        }

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset/schema")
async def get_dataset_schema():
    """Get the required dataset schema for the selected flow.

    Note: In latest sdg_hub, get_dataset_schema() returns pd.DataFrame.
    """
    try:
        if not current_config["flow"]:
            raise HTTPException(status_code=400, detail="No flow selected")

        flow = current_config["flow"]
        schema_df = flow.get_dataset_schema()  # Returns pd.DataFrame in latest sdg_hub

        requirements = flow.get_dataset_requirements()

        # Handle pandas DataFrame return type
        return {
            "columns": schema_df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in schema_df.dtypes.items()},
            "requirements": requirements.model_dump() if requirements else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset/preview")
async def preview_dataset():
    """Get a preview of the loaded dataset (pandas DataFrame)."""
    try:
        if current_config["dataset"] is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")

        df = current_config["dataset"]

        # Get first 5 samples for preview
        preview_size = min(5, len(df))
        preview_df = df.head(preview_size)
        # Convert to dict format compatible with frontend (orient='list' for column-based)
        preview_data = preview_df.to_dict(orient="list")

        return {
            "num_samples": len(df),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": preview_data,
            "preview_size": preview_size,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Flow Execution Endpoints
# ============================================================================


@app.get("/api/flow/dry-run-stream")
async def dry_run_stream(
    sample_size: int = 2,
    enable_time_estimation: bool = False,
    max_concurrency: int = None,
):
    """Stream dry run execution logs in real-time using Server-Sent Events."""
    import asyncio
    import concurrent.futures
    import queue

    async def generate_logs():
        """Generator that yields log events as they occur."""
        try:
            if current_config["flow"] is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No flow selected'})}\n\n"
                return
            if current_config["dataset"] is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No dataset loaded'})}\n\n"
                return

            flow = current_config["flow"]
            dataset = current_config["dataset"]

            # Create a queue for capturing logs
            log_queue = queue.Queue()

            # Capture Rich console output with proper terminal emulation
            import os
            import sys

            # Force Rich to use ANSI codes by setting TERM
            os.environ["TERM"] = "xterm-256color"
            os.environ["FORCE_COLOR"] = "1"

            class TeeOutput:
                """Captures output while also sending to queue."""

                def __init__(self, original, queue):
                    self.original = original
                    self.queue = queue

                def write(self, text):
                    self.original.write(text)
                    self.original.flush()
                    # Send to queue (including ANSI codes)
                    if text:  # Send everything, including empty lines for formatting
                        self.queue.put(
                            {"type": "log", "message": text, "timestamp": time.time()}
                        )

                def flush(self):
                    self.original.flush()

                def isatty(self):
                    return True  # Pretend we're a terminal for Rich

            # Redirect stdout/stderr to capture Rich output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = TeeOutput(old_stdout, log_queue)
            sys.stderr = TeeOutput(old_stderr, log_queue)

            try:
                # Start message
                yield f"data: {json.dumps({'type': 'start', 'message': f'Starting dry run with {sample_size} samples'})}\n\n"
                await asyncio.sleep(0.1)

                def run_dry_run_sync():
                    """Run dry run and capture result."""
                    return flow.dry_run(
                        dataset,
                        sample_size=sample_size,
                        enable_time_estimation=enable_time_estimation,
                        max_concurrency=max_concurrency,
                    )

                # Run in thread and stream logs
                loop = asyncio.get_event_loop()
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = loop.run_in_executor(executor, run_dry_run_sync)

                # Stream logs while execution happens
                while not future.done():
                    try:
                        log_entry = log_queue.get(timeout=0.1)
                        yield f"data: {json.dumps(log_entry)}\n\n"
                    except queue.Empty:
                        await asyncio.sleep(0.1)

                # Get remaining logs
                while not log_queue.empty():
                    try:
                        log_entry = log_queue.get_nowait()
                        yield f"data: {json.dumps(log_entry)}\n\n"
                    except queue.Empty:
                        break

                # Get result
                dry_result = await future

                # Send completion event with full results
                yield f"data: {json.dumps({'type': 'complete', 'result': dry_result})}\n\n"

            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_logs(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/flow/dry-run")
async def dry_run_flow(request: DryRunRequest):
    """Perform a dry run of the configured flow (non-streaming)."""
    try:
        if current_config["flow"] is None:
            raise HTTPException(status_code=400, detail="No flow selected")
        if current_config["dataset"] is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")

        flow = current_config["flow"]
        dataset = current_config["dataset"]

        # Run the dry run in a thread to avoid event loop conflicts
        import asyncio
        import concurrent.futures

        def run_dry_run_sync():
            """Run dry run in a separate thread with its own event loop."""
            return flow.dry_run(
                dataset,
                sample_size=request.sample_size,
                enable_time_estimation=request.enable_time_estimation,
                max_concurrency=request.max_concurrency,
            )

        # Execute in thread pool to avoid event loop conflicts
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


# ============================================================================
# Full Generation Endpoint
# ============================================================================


@app.get("/api/flow/generate-stream")
async def generate_stream(
    max_concurrency: int = None,
    log_dir: str = None,
    config_id: Optional[str] = None,
    enable_checkpoints: bool = True,
    save_freq: Optional[int] = None,
    resume_from_checkpoint: bool = False,
):
    """Stream flow generation logs in real-time using multiprocessing.

    Args:
        max_concurrency: Maximum concurrent requests
        log_dir: Directory for logs
        config_id: Configuration ID for tracking and loading isolated config
        enable_checkpoints: If True, enable checkpointing (default True)
        save_freq: Save checkpoint every N samples (no default - configured via Model Configuration)
        resume_from_checkpoint: If True, resume from existing checkpoints
    """
    import asyncio

    async def generate_logs():
        """Generator that yields log events as they occur."""
        try:
            # CRITICAL FIX: When config_id is provided, load configuration directly
            # from saved_configurations to avoid race conditions when running multiple
            # flows in parallel. Each generation gets its own isolated config.
            flow_path = None
            dataset_params = None
            model_config = {}
            flow_obj = None
            flow_name = "unknown"

            if config_id and config_id in saved_configurations:
                # Load configuration directly from saved_configurations (isolated)
                config = saved_configurations[config_id]
                logger.info(
                    f"🔧 Loading isolated config for generation: {config_id} (flow: {config.flow_name})"
                )

                # Get flow path
                if (
                    config.flow_path
                    and config.flow_path != "."
                    and config.flow_path != ""
                ):
                    flow_path_obj = Path(config.flow_path)
                    if flow_path_obj.exists():
                        flow_path = config.flow_path
                        flow_obj = Flow.from_yaml(flow_path)

                # If flow_path doesn't work, try to find by flow_id or flow_name
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

                # Get model config from saved configuration
                model_config = (config.model_configuration or {}).copy()

                # Get dataset params from saved configuration
                dataset_config = config.dataset_configuration or {}
                if (
                    dataset_config.get("data_files")
                    and dataset_config.get("data_files") != "."
                ):
                    # Construct dataset_load_params from saved dataset_configuration
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
                    }
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Dataset not configured for this flow'})}\n\n"
                    return

                flow_name = config.flow_name or (
                    flow_obj.metadata.name if flow_obj else "unknown"
                )

            else:
                # Fall back to global current_config (for backward compatibility)
                if not current_config["flow"] or not current_config["flow_path"]:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'No flow selected'})}\n\n"
                    return

                # Check for dataset params
                if not current_config.get("dataset_load_params"):
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Dataset source info missing. Please reload the dataset.'})}\n\n"
                    return

                flow_path = current_config["flow_path"]
                dataset_params = current_config["dataset_load_params"]
                model_config = current_config.get("model_config", {}).copy()
                flow_obj = current_config["flow"]
                flow_name = flow_obj.metadata.name if flow_obj else "unknown"

            # Resolve API key safely
            if model_config.get("api_key"):
                model_config["api_key"] = get_safe_api_key(model_config)

            # Setup multiprocessing
            ctx = multiprocessing.get_context("spawn")
            log_queue = ctx.Queue()

            # Clean up any previous generation for this config
            if (
                active_generation_process.get("pid")
                and active_generation_process.get("config_id") == config_id
            ):
                old_pid = active_generation_process["pid"]
                logger.warning(
                    f"⚠️ Previous generation for config {config_id} still active (PID={old_pid}). Killing it."
                )
                try:
                    import os
                    import signal

                    os.kill(old_pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                active_generation_process["pid"] = None
                active_generation_process["config_id"] = None
                # Clean up from active_generations
                if config_id in active_generations:
                    del active_generations[config_id]

            # Setup checkpoint directory if enabled
            checkpoint_dir = None
            if enable_checkpoints and config_id:
                checkpoint_dir = str(get_checkpoint_dir_for_config(config_id))

            # Start worker process with checkpoint support
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

            # Store in active_generations for reconnection support
            # Note: flow_name was already set earlier from isolated config or current_config
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

            # Stream logs
            while process.is_alive():
                while not log_queue.empty():
                    try:
                        item = log_queue.get_nowait()

                        if item["type"] == "result":
                            # Completion!
                            dataset_list = item["dataset_list"]
                            column_names = item["column_names"]
                            num_samples = len(dataset_list)
                            num_columns = len(column_names)

                            # Update state
                            current_config["generated_dataset"] = dataset_list
                            current_config["generated_dataset_info"] = {
                                "num_samples": num_samples,
                                "num_columns": num_columns,
                                "columns": column_names,
                            }

                            # Save to JSONL
                            from datetime import datetime

                            outputs_dir = Path("outputs")
                            outputs_dir.mkdir(exist_ok=True)

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            # Use the flow_name we determined earlier (from isolated config or current_config)
                            # Sanitize flow_name to prevent path traversal
                            output_flow_name = (
                                sanitize_filename(flow_name.replace(" ", "_").lower())
                                if flow_name
                                else "unknown"
                            )
                            output_filename = f"{output_flow_name}_{timestamp}.jsonl"
                            output_path = ensure_within_directory(
                                outputs_dir, outputs_dir / output_filename
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
                            }
                            yield f"data: {json.dumps(completion_data)}\n\n"

                        elif item["type"] == "error":
                            yield f"data: {json.dumps(item)}\n\n"
                        else:
                            # Log message
                            yield f"data: {json.dumps(item)}\n\n"

                    except queue.Empty:
                        break

                await asyncio.sleep(0.1)

            # Process any remaining logs after death
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

            # Clean up from active_generations
            if config_id and config_id in active_generations:
                del active_generations[config_id]

            if process.exitcode != 0:
                # Check for SIGTERM (-15) or SIGKILL (-9)
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


@app.get("/api/flow/generation-status")
async def get_generation_status(config_id: Optional[str] = None):
    """Check if there are any running generations (optionally for a specific config)."""
    try:
        running_generations = []

        # Clean up any dead processes from active_generations
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

        # If specific config requested
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

        # Return all running generations
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


@app.get("/api/flow/reconnect-stream")
async def reconnect_stream(config_id: str):
    """Reconnect to an existing running generation's log stream."""
    import asyncio

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
                # Clean up
                if config_id in active_generations:
                    del active_generations[config_id]
                return

            yield f"data: {json.dumps({'type': 'reconnected', 'message': f'Reconnected to generation process (PID: {process.pid})'})}\n\n"

            # Stream logs from the queue
            while process.is_alive():
                while not log_queue.empty():
                    try:
                        item = log_queue.get_nowait()

                        if item["type"] == "result":
                            # Completion!
                            dataset_list = item["dataset_list"]
                            column_names = item["column_names"]
                            num_samples = len(dataset_list)
                            num_columns = len(column_names)

                            # Update state
                            current_config["generated_dataset"] = dataset_list
                            current_config["generated_dataset_info"] = {
                                "num_samples": num_samples,
                                "num_columns": num_columns,
                                "columns": column_names,
                            }

                            # Save to JSONL
                            from datetime import datetime

                            outputs_dir = Path("outputs")
                            outputs_dir.mkdir(exist_ok=True)

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            # Sanitize flow_name to prevent path traversal
                            raw_flow_name = (
                                gen_info.get("flow_name", "flow")
                                .replace(" ", "_")
                                .lower()
                            )
                            safe_flow_name = sanitize_filename(raw_flow_name) or "flow"
                            output_filename = f"{safe_flow_name}_{timestamp}.jsonl"
                            output_path = ensure_within_directory(
                                outputs_dir, outputs_dir / output_filename
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
                            }
                            yield f"data: {json.dumps(completion_data)}\n\n"

                            # Clean up
                            if config_id in active_generations:
                                del active_generations[config_id]

                        elif item["type"] == "error":
                            yield f"data: {json.dumps(item)}\n\n"
                            # Clean up
                            if config_id in active_generations:
                                del active_generations[config_id]
                        else:
                            # Log message
                            yield f"data: {json.dumps(item)}\n\n"

                    except queue.Empty:
                        break

                await asyncio.sleep(0.1)

            # Process any remaining logs after death
            while not log_queue.empty():
                try:
                    item = log_queue.get_nowait()
                    if item["type"] == "result":
                        # Handle completion
                        dataset_list = item["dataset_list"]
                        column_names = item["column_names"]
                        completion_data = {
                            "type": "complete",
                            "num_samples": len(dataset_list),
                            "num_columns": len(column_names),
                            "output_file": None,
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

            # Clean up
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


@app.post("/api/flow/cancel-generation")
async def cancel_generation(config_id: Optional[str] = None):
    """Cancel generation by killing the worker process."""
    try:
        import os
        import signal

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
                # Use SIGKILL to ensure it stops immediately
                os.kill(pid, signal.SIGKILL)

                active_generation_process["pid"] = None
                active_generation_process["config_id"] = None

                # Clean up from active_generations too
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
                # Clean up from active_generations too
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
            # Clean up from active_generations too
            if config_id and config_id in active_generations:
                del active_generations[config_id]
            return {
                "status": "success",
                "message": "No active generation process found.",
            }

    except Exception as e:
        logger.error(f"Error cancelling generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Checkpoint Management Endpoints
# ============================================================================


@app.get("/api/flow/checkpoints/{config_id}")
async def get_checkpoints(config_id: str):
    """Get checkpoint information for a configuration.

    Returns info about existing checkpoints including:
    - Whether checkpoints exist
    - Number of checkpoint files
    - Approximate number of completed samples
    - Last checkpoint timestamp
    """
    try:
        info = get_checkpoint_info(config_id)
        return {"status": "success", "config_id": config_id, **info}
    except Exception as e:
        logger.error(f"Error getting checkpoint info for {config_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/flow/checkpoints/{config_id}")
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


@app.get("/api/flow/download-generated")
async def download_generated():
    """Download the generated dataset as JSONL."""
    try:
        if not current_config.get("generated_dataset"):
            raise HTTPException(
                status_code=404, detail="No generated dataset available"
            )

        dataset_list = current_config["generated_dataset"]

        # Convert to JSONL
        import io

        output = io.StringIO()
        for item in dataset_list:
            output.write(json.dumps(item) + "\n")

        from fastapi.responses import Response

        # Get dataset info for filename
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


# ============================================================================
# Flow Creation and Management
# ============================================================================


@app.post("/api/flows/create")
async def create_flow(
    flow_yaml: str = Form(...),
    flow_name: str = Form(...),
    prompt_templates: str = Form(None),
    source_flow_name: str = Form(None),
):
    """Save a new custom flow to the flows directory."""
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
        if source_flow_name:
            source_flow_path = FlowRegistry.get_flow_path(source_flow_name)
            if source_flow_path:
                source_dir = Path(source_flow_path).parent

                # Validate source directory is within allowed directories
                if is_path_within_allowed_dirs(source_dir, ALLOWED_FLOW_READ_DIRS):
                    # Copy all YAML files except flow.yaml
                    for yaml_file in source_dir.glob("*.yaml"):
                        if yaml_file.name != "flow.yaml":
                            # Validate source file is within allowed directories (path traversal protection)
                            resolved_yaml_file = yaml_file.resolve()
                            if not is_path_within_allowed_dirs(
                                resolved_yaml_file, ALLOWED_FLOW_READ_DIRS
                            ):
                                logger.warning(
                                    f"Skipping file outside allowed directories: {yaml_file}"
                                )
                                continue
                            # Sanitize destination filename and validate path
                            safe_filename = (
                                sanitize_filename(yaml_file.name) or "prompt.yaml"
                            )
                            dest_file = ensure_within_directory(
                                flow_dir, flow_dir / safe_filename
                            )
                            shutil.copy2(resolved_yaml_file, dest_file)
                            logger.info(f"Copied prompt template: {yaml_file.name}")

        # Save flow.yaml
        flow_yaml_path = ensure_within_directory(flow_dir, flow_dir / "flow.yaml")
        with open(flow_yaml_path, "w") as f:
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

                with open(template_yaml_path, "w") as f:
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


# ============================================================================
# Configuration Management
# ============================================================================


@app.post("/api/config/import")
async def import_config(file: UploadFile = File(...)):
    """Import a previously exported configuration file."""
    try:
        # Read and parse the config file
        content = await file.read()
        config_data = json.loads(content)

        # Extract configuration components
        flow_info = config_data.get("flow")
        model_cfg = config_data.get("model_config")
        dataset_cfg = config_data.get("dataset_config")

        if not flow_info or not flow_info.get("name"):
            raise HTTPException(
                status_code=400,
                detail="Invalid configuration file: missing flow information",
            )

        # Load the flow
        flow_name = flow_info["name"]
        flow_path = FlowRegistry.get_flow_path(flow_name)
        if not flow_path:
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")

        flow = Flow.from_yaml(flow_path)
        current_config["flow"] = flow
        current_config["flow_path"] = flow_path

        logger.info(f"Imported flow: {flow_name}")

        # Apply model configuration if present
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

        # Load dataset if configuration present (using pandas for performance)
        dataset_loaded = False
        if dataset_cfg and dataset_cfg.get("data_files"):
            try:
                data_files = dataset_cfg["data_files"]
                file_format = dataset_cfg.get("file_format", "auto")

                # Use pandas loading helper
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


@app.get("/api/config/current")
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

    # Mask API key in response
    safe_model_config = sanitize_model_config(
        current_config["model_config"], mask_key=True
    )

    return {
        "flow": flow_info,
        "model_config": safe_model_config,
        "dataset_info": current_config["dataset_info"],
    }


@app.post("/api/config/reset")
async def reset_config():
    """Reset the current configuration."""
    current_config["flow"] = None
    current_config["flow_path"] = None
    current_config["model_config"] = {}
    current_config["dataset"] = None
    current_config["dataset_info"] = {}

    logger.info("Configuration reset")

    return {"status": "success", "message": "Configuration reset"}


# ============================================================================
# Block Registry Endpoints
# ============================================================================


@app.get("/api/blocks/list")
async def list_blocks():
    """List all available block types."""
    try:
        blocks = BlockRegistry.list_blocks()
        return {"blocks": blocks}
    except Exception as e:
        logger.error(f"Error listing blocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/blocks/templates")
async def get_block_templates():
    """Get pre-configured block templates from existing flows."""
    try:
        import yaml

        block_templates = []

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
                    logger.warning(f"Skipping flow {flow_name}: path validation failed")
                    continue

                # Read the flow YAML
                with open(validated_path, "r") as f:
                    flow_data = yaml.safe_load(f)

                # Extract block configurations
                blocks = flow_data.get("blocks", [])
                for block in blocks:
                    block_config = block.get("block_config", {})
                    block_name = block_config.get("block_name", "unknown")

                    # Create template entry
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


@app.get("/api/flows/templates")
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

                # Create template entry with full flow configuration
                template = {
                    "id": flow_info["id"],
                    "name": flow_name,
                    "metadata": flow_data.get("metadata", {}),
                    "blocks": flow_data.get("blocks", []),
                    "num_blocks": len(flow_data.get("blocks", [])),
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


# ============================================================================
# Prompt Template Endpoints
# ============================================================================


@app.get("/api/prompts/load")
async def load_prompt_template(prompt_path: str):
    """Load an existing prompt template YAML file."""
    try:
        import yaml

        # Validate and resolve the prompt file path within allowed directories
        prompt_file = resolve_prompt_file(prompt_path)

        # Load YAML
        with open(prompt_file, "r") as f:
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


# ============================================================================
# Flow Runs History Endpoints
# ============================================================================


@app.get("/api/runs/list")
async def list_runs():
    """Get list of all flow runs."""
    try:
        runs = load_runs_history()
        return {"runs": runs}
    except Exception as e:
        logger.error(f"Error loading runs history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/runs/create")
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


@app.put("/api/runs/{run_id}/update")
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


@app.delete("/api/runs/{run_id}")
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


@app.get("/api/runs/{run_id}")
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


@app.get("/api/runs/{run_id}/download")
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

        # Check if file exists
        file_path = Path(output_file)
        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Output file not found: {output_file}"
            )

        # Return file for download
        from fastapi.responses import FileResponse

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


# ============================================================================
# Custom Flow Management
# ============================================================================


@app.post("/api/flows/save-custom")
async def save_custom_flow(flow_data: Dict[str, Any]):
    """Save a custom flow to the custom_flows directory."""
    try:
        import shutil

        import yaml

        CUSTOM_FLOWS_DIR.mkdir(parents=True, exist_ok=True)

        # Create flow directory
        flow_name = flow_data.get("metadata", {}).get("name", "unnamed_flow")
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
        if source_flow_name:
            try:
                # Try to get flow path by name (works for both custom and predefined flows)
                source_flow_path = FlowRegistry.get_flow_path(source_flow_name)
                if source_flow_path:
                    source_flow_dir = Path(source_flow_path).parent.resolve()
                    logger.info(
                        f"Source flow directory from name '{source_flow_name}': {source_flow_dir}"
                    )
            except Exception as e:
                logger.warning(f"Could not get source flow directory by name: {e}")

        # Fall back to source_flow_id if name didn't work
        if not source_flow_dir and source_flow_id:
            try:
                source_flow_path = FlowRegistry.get_flow_path(source_flow_id)
                if source_flow_path:
                    source_flow_dir = Path(source_flow_path).parent.resolve()
                    logger.info(f"Source flow directory from ID: {source_flow_dir}")
            except Exception as e:
                logger.warning(f"Could not get source flow directory by ID: {e}")

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
                    sanitize_filename(old_path_obj.name)
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
                    logger.info(f"💾 Saving wizard prompt for block: {block_name}")
                    with open(new_prompt_file, "w") as f:
                        yaml.dump(
                            wizard_prompts[block_name], f, default_flow_style=False
                        )
                    logger.info(f"✅ Saved wizard prompt: {new_prompt_file}")
                else:
                    # First check if prompt already exists in target flow directory (for editing existing flows)
                    if new_prompt_file.exists():
                        logger.info(
                            f"✓ Prompt already exists in flow directory, skipping copy: {new_prompt_file}"
                        )
                        continue  # Skip to next block - prompt is already in place

                    # Try to copy from temp flow directory first (for newly created prompts)
                    source_file = None

                    # Check temp flow directory first
                    if temp_flow_dir and temp_flow_dir.exists():
                        temp_prompt_file = temp_flow_dir / prompt_filename
                        if temp_prompt_file.exists():
                            source_file = temp_prompt_file
                            logger.info(f"✓ Found prompt in temp flow: {source_file}")

                    # If we have source flow directory, look there
                    if not source_file and source_flow_dir:
                        # Try with the full old_path first
                        source_in_flow_dir = (source_flow_dir / old_path).resolve()
                        if source_in_flow_dir.exists():
                            source_file = source_in_flow_dir
                            logger.info(
                                f"✓ Found prompt in source flow (full path): {source_file}"
                            )
                        else:
                            # Try with just the filename
                            source_in_flow_dir = (
                                source_flow_dir / prompt_filename
                            ).resolve()
                            if source_in_flow_dir.exists():
                                source_file = source_in_flow_dir
                                logger.info(
                                    f"✓ Found prompt in source flow (filename): {source_file}"
                                )

                    if not source_file:
                        logger.warning(
                            f"Prompt source not found for block {block_name} ({old_path})."
                        )

                    # Copy the file or fail
                    if source_file:
                        resolved_source = Path(source_file).resolve()

                        # Build list of allowed directories - start with standard flow directories
                        allowed_prompt_dirs = list(ALLOWED_FLOW_READ_DIRS)

                        # Add current working directories if they exist
                        if flow_dir:
                            allowed_prompt_dirs.append(flow_dir)
                        if temp_flow_dir:
                            allowed_prompt_dirs.append(temp_flow_dir)

                        # When cloning from a source flow, also allow parent directories
                        # because prompts can be shared across flows using relative paths like ../prompt.yaml
                        if source_flow_dir:
                            # Add source flow dir and all its parent directories up to 'flows' or 'sdg_hub'
                            current_dir = source_flow_dir
                            while current_dir and current_dir.name:
                                allowed_prompt_dirs.append(current_dir)
                                # Stop at the 'flows' directory or root
                                if current_dir.name in (
                                    "flows",
                                    "sdg_hub",
                                    "custom_flows",
                                ):
                                    break
                                current_dir = current_dir.parent

                        # Use the standard helper function for path validation (path traversal protection)
                        if not is_path_within_allowed_dirs(
                            resolved_source, allowed_prompt_dirs
                        ):
                            raise HTTPException(
                                status_code=400,
                                detail=f"Prompt source {resolved_source} is outside allowed directories.",
                            )
                        shutil.copy2(resolved_source, new_prompt_file)
                        logger.info(
                            f"✅ Copied prompt file: {resolved_source} -> {new_prompt_file}"
                        )
                    else:
                        # Can't find source - this is an error
                        error_msg = f"❌ CRITICAL: Could not find source prompt file: {old_path} (block: {block_name}, source_flow_id: {source_flow_id})"
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
            logger.info(f"✅ Auto-selected saved custom flow: {flow_name}")
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


@app.post("/api/prompts/save")
async def save_prompt_template(prompt_data: Dict[str, Any]):
    """Save a prompt template YAML file."""
    try:
        import yaml

        prompt_name = prompt_data.get("prompt_name", "custom_prompt")
        prompt_content = prompt_data.get("prompt_content", [])
        flow_name = prompt_data.get("flow_name", "custom_flow")

        # Create custom flows directory if it doesn't exist
        CUSTOM_FLOWS_DIR.mkdir(parents=True, exist_ok=True)

        # Create flow directory
        safe_flow_name = slugify_name(flow_name, prefix="flow")
        flow_dir = ensure_within_directory(
            CUSTOM_FLOWS_DIR, CUSTOM_FLOWS_DIR / safe_flow_name
        )
        flow_dir.mkdir(parents=True, exist_ok=True)

        # Save prompt file
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


# ============================================================================
# Configuration Management
# ============================================================================

# Persistent storage for saved configurations
saved_configurations = {}


def load_saved_configurations_from_disk():
    """Load saved configurations from JSON file."""
    if not SAVED_CONFIG_FILE.exists():
        logger.info("No saved configurations file found; starting fresh.")
        return
    try:
        with open(SAVED_CONFIG_FILE, "r") as f:
            data = json.load(f)
        saved_configurations.clear()
        for item in data:
            try:
                config = SavedConfiguration(**item)
                saved_configurations[config.id] = config
            except Exception as exc:
                logger.warning(f"Skipping invalid saved configuration entry: {exc}")
        logger.info(
            f"Loaded {len(saved_configurations)} saved configurations from disk."
        )
    except Exception as exc:
        logger.error(f"Failed to load saved configurations: {exc}")


def persist_saved_configurations():
    """Persist saved configurations to disk."""
    try:
        SAVED_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = SAVED_CONFIG_FILE.with_suffix(".tmp")
        data = [config.dict() for config in saved_configurations.values()]
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, SAVED_CONFIG_FILE)
        try:
            SAVED_CONFIG_FILE.chmod(0o600)
        except Exception:
            pass
        logger.info(f"Persisted {len(saved_configurations)} configurations to disk.")
    except Exception as exc:
        logger.error(f"Failed to persist configurations: {exc}")


class SavedConfiguration(BaseModel):
    """Saved configuration model."""

    id: str
    flow_name: str
    flow_id: str
    flow_path: str
    model_configuration: Dict[str, Any]
    dataset_configuration: Dict[str, Any]
    dry_run_configuration: Optional[Dict[str, Any]] = None
    tags: List[str] = []
    status: Optional[str] = "configured"  # configured, not_configured, draft
    created_at: str
    updated_at: str


class SaveConfigurationRequest(BaseModel):
    """Request to save a configuration."""

    flow_name: str
    flow_id: str
    flow_path: str
    model_configuration: Dict[str, Any]
    dataset_configuration: Dict[str, Any]
    dry_run_configuration: Optional[Dict[str, Any]] = None
    tags: List[str] = []
    status: Optional[str] = "configured"  # configured, not_configured, draft


@app.post("/api/configurations/save")
async def save_configuration(request: SaveConfigurationRequest):
    """Save a flow configuration (API keys stored locally for convenience)."""
    try:
        from datetime import datetime
        import uuid

        config_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Determine if we are persisting a direct API key (non-env)
        original_api_key = request.model_configuration.get("api_key", "")
        is_direct_api_key = bool(original_api_key) and not (
            original_api_key in ["EMPTY", "NONE", ""]
            or original_api_key.startswith("env:")
        )

        # Store full model configuration (will be masked when returned)
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
        persist_saved_configurations()
        logger.info(f"✅ Saved configuration: {config_id} for flow {request.flow_name}")

        # Return masked version
        response_config = config.dict()
        response_config["model_configuration"] = sanitize_model_config(
            response_config["model_configuration"], mask_key=True
        )

        # Build response with conditional warning
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


@app.get("/api/configurations/list")
async def list_configurations():
    """List all saved configurations."""
    try:
        configs = []
        for config in saved_configurations.values():
            config_dict = config.dict()
            # Mask API key in model configuration
            config_dict["model_configuration"] = sanitize_model_config(
                config_dict["model_configuration"], mask_key=True
            )
            configs.append(config_dict)

        logger.info(f"Listed {len(configs)} configurations")
        return {"status": "success", "configurations": configs}
    except Exception as e:
        logger.error(f"Error listing configurations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/configurations/{config_id}")
async def get_configuration(config_id: str):
    """Get a specific configuration."""
    try:
        if config_id not in saved_configurations:
            raise HTTPException(
                status_code=404, detail=f"Configuration {config_id} not found"
            )

        config = saved_configurations[config_id]
        config_dict = config.dict()

        # Mask API key in response
        config_dict["model_configuration"] = sanitize_model_config(
            config_dict["model_configuration"], mask_key=True
        )

        return {"status": "success", "configuration": config_dict}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/configurations/{config_id}")
async def delete_configuration(config_id: str):
    """Delete a configuration."""
    try:
        if config_id not in saved_configurations:
            raise HTTPException(
                status_code=404, detail=f"Configuration {config_id} not found"
            )

        del saved_configurations[config_id]
        persist_saved_configurations()
        logger.info(f"Deleted configuration: {config_id}")
        return {"status": "success", "message": f"Configuration {config_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/configurations/{config_id}/load")
async def load_configuration(config_id: str):
    """Load a configuration into current state."""
    try:
        if config_id not in saved_configurations:
            raise HTTPException(
                status_code=404, detail=f"Configuration {config_id} not found"
            )

        config = saved_configurations[config_id]

        # Try to load flow from saved path first, otherwise use flow_id to lookup
        flow = None
        flow_path_str = None

        if config.flow_path and config.flow_path != "." and config.flow_path != "":
            flow_path = Path(config.flow_path)
            if flow_path.exists():
                flow = Flow.from_yaml(str(flow_path))
                flow_path_str = config.flow_path

        # If flow_path doesn't work, try to find by flow_id or flow_name
        if flow is None:
            try:
                # Try to get flow path from registry
                if config.flow_id:
                    flow_path_str = FlowRegistry.get_flow_path(config.flow_id)
                    if flow_path_str:
                        flow = Flow.from_yaml(flow_path_str)
            except Exception as e:
                logger.warning(f"Could not find flow by ID, trying by name: {e}")

            # Last resort: search by flow name
            # Note: get_flow_path accepts both flow_id and flow_name for backward compatibility
            if flow is None and config.flow_name:
                flow_path_str = FlowRegistry.get_flow_path(config.flow_name)
                if flow_path_str:
                    flow = Flow.from_yaml(flow_path_str)

        if flow is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not load flow. Path: {config.flow_path}, ID: {config.flow_id}, Name: {config.flow_name}",
            )

        # Update current config
        current_config["flow"] = flow
        current_config["flow_path"] = flow_path_str
        current_config["model_config"] = config.model_configuration

        # Apply model configuration
        if config.model_configuration:
            flow.set_model_config(**config.model_configuration)

        # Note: Dataset is not reloaded automatically - user needs to load it

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


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Apply nest_asyncio before starting uvicorn
    # This allows sdg_hub async blocks to work within FastAPI's async context
    try:
        nest_asyncio.apply()
        logger.info("✅ nest_asyncio applied successfully")
    except Exception as e:
        logger.warning(f"Could not apply nest_asyncio: {e}")

    logger.info("🚀 Starting server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level="info")
