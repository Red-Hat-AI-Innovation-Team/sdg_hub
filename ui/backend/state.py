# SPDX-License-Identifier: Apache-2.0
"""Shared mutable state for the SDG Hub API server.

All module-level dicts/objects that are mutated by endpoints live here
so they can be imported by any module that needs them.
"""

from typing import Any, Dict

from utils.preprocessing_utils import load_preprocessing_jobs as _load_preprocessing_jobs

# Global state for current configuration
current_config: Dict[str, Any] = {
    "flow": None,
    "flow_path": None,
    "model_config": {},
    "dataset": None,
    "dataset_info": {},
    "dataset_load_params": None,
}

# Generation control
generation_cancel_flag: Dict[str, bool] = {"should_cancel": False}
active_generation_process: Dict[str, Any] = {"pid": None, "config_id": None}

# Dry run control
# Tracks the active dry run process for cancellation support (using multiprocessing like generation)
active_dry_run: Dict[str, Any] = {
    "pid": None,
    "config_id": None,
    "start_time": None,
    "process": None,
    "queue": None,
}

# Global log queues per config_id for reconnection support
# Maps config_id -> {"queue": multiprocessing.Queue, "process": Process, "start_time": timestamp}
active_generations: Dict[str, Any] = {}

# Track preprocessing jobs (loaded from disk on startup)
preprocessing_jobs: Dict[str, Dict[str, Any]] = _load_preprocessing_jobs()

# Persistent storage for saved configurations
saved_configurations: Dict[str, Any] = {}
