# SPDX-License-Identifier: Apache-2.0
"""Checkpoint management utilities."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException

from config import CHECKPOINTS_DIR
from utils.file_handling import sanitize_filename
from utils.security import ensure_within_directory

logger = logging.getLogger(__name__)


def get_checkpoint_dir_for_config(config_id: str) -> Path:
    """Get the checkpoint directory path for a specific configuration.

    Validates that config_id doesn't contain path traversal sequences.
    """
    safe_config_id = sanitize_filename(config_id)
    if not safe_config_id:
        raise HTTPException(status_code=400, detail="Invalid configuration ID")

    checkpoint_dir = CHECKPOINTS_DIR / safe_config_id
    return ensure_within_directory(CHECKPOINTS_DIR, checkpoint_dir)


def get_checkpoint_info(config_id: str) -> Dict[str, Any]:
    """Get information about existing checkpoints for a configuration."""
    checkpoint_dir = get_checkpoint_dir_for_config(config_id)

    if not checkpoint_dir.exists():
        return {
            "has_checkpoints": False,
            "checkpoint_count": 0,
            "samples_completed": 0,
            "last_checkpoint_time": None,
            "checkpoint_dir": str(checkpoint_dir),
        }

    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.jsonl"))

    if not checkpoint_files:
        return {
            "has_checkpoints": False,
            "checkpoint_count": 0,
            "samples_completed": 0,
            "last_checkpoint_time": None,
            "checkpoint_dir": str(checkpoint_dir),
        }

    total_samples = 0
    for cp_file in checkpoint_files:
        try:
            with open(cp_file, "r") as f:
                total_samples += sum(1 for _ in f)
        except Exception:
            pass

    last_checkpoint = checkpoint_files[-1]
    last_modified = last_checkpoint.stat().st_mtime
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
