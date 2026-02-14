#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SDG Hub Configuration

Path constants, directory setup, settings, and allowed-directory lists.
Imported by all other backend modules.
"""

from pathlib import Path
from typing import Dict, List
import logging
import os
import re

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
    logger.info(f"🔐 Loaded environment variables from: {env_file}")
else:
    logger.info("ℹ️  No .env file found. Using system environment variables only.")

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------

UPLOADS_DIR = (DATA_DIR / "uploads").resolve()
CUSTOM_FLOWS_DIR = (DATA_DIR / "custom_flows").resolve()
SAVED_CONFIG_FILE = (DATA_DIR / "saved_configurations.json").resolve()
CHECKPOINTS_DIR = (DATA_DIR / "checkpoints").resolve()
OUTPUTS_DIR = (BASE_DIR / "outputs").resolve()

# Ensure required directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_FLOWS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# File handling constants
# ---------------------------------------------------------------------------

FILENAME_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]")
MAX_UPLOAD_SIZE_MB = int(os.getenv("SDG_HUB_MAX_UPLOAD_MB", "512"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".csv", ".parquet", ".pq"}

# ---------------------------------------------------------------------------
# Workspace ID validation
# ---------------------------------------------------------------------------

# Workspace ID pattern: must be temp_ws_ followed by alphanumerics/underscores only
_WORKSPACE_ID_RE = re.compile(r"^temp_ws_[A-Za-z0-9_]+$")

# ---------------------------------------------------------------------------
# Allowed dataset directories
# ---------------------------------------------------------------------------

ALLOWED_DATASET_DIRS: List[Path] = [UPLOADS_DIR, OUTPUTS_DIR]
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

# ---------------------------------------------------------------------------
# Flow source directories
# ---------------------------------------------------------------------------

# Get SDG Hub flows directory (for reading predefined flows)
SDG_HUB_FLOWS_DIR = (
    Path(__file__).parent.parent.parent / "src" / "sdg_hub" / "flows"
).resolve()

# Allowed directories for reading flow files
ALLOWED_FLOW_READ_DIRS: List[Path] = [CUSTOM_FLOWS_DIR, SDG_HUB_FLOWS_DIR]

# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------

RUNS_HISTORY_FILE = DATA_DIR / "runs_history.json"

# ---------------------------------------------------------------------------
# PDF preprocessing directories
# ---------------------------------------------------------------------------

PDF_UPLOADS_DIR = (DATA_DIR / "pdf_uploads").resolve()
PDF_CONVERTED_DIR = (DATA_DIR / "pdf_converted").resolve()
PDF_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PDF_CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

# File for persisting preprocessing jobs
PREPROCESSING_JOBS_FILE = DATA_DIR / "preprocessing_jobs.json"
