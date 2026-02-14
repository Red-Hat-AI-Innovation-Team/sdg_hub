# SPDX-License-Identifier: Apache-2.0
"""Preprocessing job persistence utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from config import PREPROCESSING_JOBS_FILE

logger = logging.getLogger(__name__)


def load_preprocessing_jobs(jobs_file: Path = PREPROCESSING_JOBS_FILE) -> Dict[str, Dict[str, Any]]:
    """Load preprocessing jobs from disk.

    Returns:
        Dictionary of job_id -> job_data.
    """
    if jobs_file.exists():
        try:
            with open(jobs_file, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} preprocessing jobs from disk")
            return data
        except Exception as e:
            logger.error(f"Failed to load preprocessing jobs: {e}")
    return {}


def save_preprocessing_jobs(
    jobs: Dict[str, Dict[str, Any]],
    jobs_file: Path = PREPROCESSING_JOBS_FILE,
):
    """Save preprocessing jobs to disk."""
    try:
        with open(jobs_file, "w") as f:
            json.dump(jobs, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save preprocessing jobs: {e}")
