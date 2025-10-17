# SPDX-License-Identifier: Apache-2.0
"""Utilities for managing temporary files and directories used by the flow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

# Standard
import os
import shutil
import tempfile

TEMP_ROOT_DIR_NAME = ".tmp_sdg_buffer"


def _get_temp_root() -> Path:
    root = Path.cwd() / TEMP_ROOT_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _format_prefix(prefix: str) -> str:
    return f"{prefix}_" if prefix and not prefix.endswith("_") else prefix


def create_temp_dir(prefix: str = "tmp", suffix: str = "") -> Path:
    """Create a unique temporary directory."""
    root = _get_temp_root()
    name = tempfile.mkdtemp(prefix=_format_prefix(prefix), suffix=suffix, dir=root)
    return Path(name)


def create_temp_file(prefix: str = "tmp", suffix: str = "") -> Path:
    """Create a unique temporary file."""
    root = _get_temp_root()
    fd, name = tempfile.mkstemp(prefix=_format_prefix(prefix), suffix=suffix, dir=root)
    os.close(fd)
    return Path(name)


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
