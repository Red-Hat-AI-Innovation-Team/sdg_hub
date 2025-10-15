# SPDX-License-Identifier: Apache-2.0
"""Utilities for managing temporary files and directories used by the flow."""

from __future__ import annotations

# Standard
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Union


TEMP_ROOT_DIR_NAME = ".tmp_sdg_buffer"


def _get_temp_root() -> Path:
    """Return the root directory for all temporary resources."""
    root = Path.cwd() / TEMP_ROOT_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_temp_dir(prefix: str = "tmp", suffix: str = "") -> Path:
    """Create a unique temporary directory under the common temp root."""
    root = _get_temp_root()
    for _ in range(100):
        candidate = root / f"{prefix}_{uuid.uuid4().hex}{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Failed to create a unique temporary directory.")


def create_temp_file(
    prefix: str = "tmp", suffix: str = "", ensure_parent: bool = True
) -> Path:
    """Create a unique temporary file path under the common temp root."""
    root = _get_temp_root()
    for _ in range(100):
        candidate = root / f"{prefix}_{uuid.uuid4().hex}{suffix}"
        if candidate.exists():
            continue
        if ensure_parent:
            candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.touch()
        return candidate
    raise RuntimeError("Failed to create a unique temporary file.")


def cleanup_path(path: Optional[Union[str, os.PathLike]]) -> None:
    """Remove a temporary file or directory if it exists."""
    if not path:
        return

    target = Path(path)
    if not target.exists():
        return

    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
    else:
        try:
            target.unlink()
        except FileNotFoundError:
            pass