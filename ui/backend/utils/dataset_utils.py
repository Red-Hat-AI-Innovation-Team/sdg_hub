# SPDX-License-Identifier: Apache-2.0
"""Dataset utilities: duplicate detection, runs history persistence."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from config import RUNS_HISTORY_FILE

logger = logging.getLogger(__name__)


def _make_hashable_for_dedup(x):
    """Convert any value to a hashable representation for duplicate detection.

    Mirrors the logic in sdg_hub.core.utils.datautils._make_hashable so
    the UI's duplicate check/removal is consistent with the library's
    validate_no_duplicates() call that runs before generation.
    """
    import numpy as np

    def _is_hashable(v):
        try:
            hash(v)
            return True
        except TypeError:
            return False

    if _is_hashable(x):
        return x
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return _make_hashable_for_dedup(x.item())
        return tuple(_make_hashable_for_dedup(i) for i in x)
    if isinstance(x, dict):
        return tuple(
            sorted(
                ((k, _make_hashable_for_dedup(v)) for k, v in x.items()),
                key=lambda kv: repr(kv[0]),
            )
        )
    if isinstance(x, (set, frozenset)):
        return frozenset(_make_hashable_for_dedup(i) for i in x)
    if hasattr(x, "__iter__"):
        return tuple(_make_hashable_for_dedup(i) for i in x)
    return repr(x)


def _get_hashable_duplicate_mask(df: pd.DataFrame):
    """Return a boolean mask of duplicate rows using hashable comparison."""
    hashable_df = df.map(_make_hashable_for_dedup)
    return hashable_df.duplicated(keep="first")


def load_runs_history() -> List[Dict[str, Any]]:
    """Load runs history from file."""
    if RUNS_HISTORY_FILE.exists():
        try:
            with open(RUNS_HISTORY_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []
    return []


def save_runs_history(runs: List[Dict[str, Any]]):
    """Save runs history to file."""
    with open(RUNS_HISTORY_FILE, "w") as f:
        json.dump(runs, f, indent=2)
