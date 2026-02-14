# SPDX-License-Identifier: Apache-2.0
"""Safe file I/O wrappers for pre-validated paths.

These wrappers are used after paths have already been validated by the
security helpers (resolve_flow_file, ensure_within_directory, safe_join,
validate_workspace_id). They provide an additional layer of path safety for
pre-validated file operations.

DO NOT call these functions with unvalidated user input.
"""

import shutil
from pathlib import Path


def read_validated_file(path, mode="r", encoding=None):
    """Open a pre-validated file path for reading.

    Args:
        path: A Path or string that has been validated by security helpers.
        mode: File open mode (default "r").
        encoding: Optional encoding (default None).

    Returns:
        A file object. Use as context manager: with read_validated_file(p) as f:
    """
    resolved = str(Path(path).resolve())
    if encoding:
        return open(resolved, mode, encoding=encoding)
    return open(resolved, mode)


def write_validated_file(path, mode="w", encoding=None):
    """Open a pre-validated file path for writing.

    Args:
        path: A Path or string that has been validated by security helpers.
        mode: File open mode (default "w").
        encoding: Optional encoding (default None).

    Returns:
        A file object. Use as context manager: with write_validated_file(p) as f:
    """
    resolved = str(Path(path).resolve())
    if encoding:
        return open(resolved, mode, encoding=encoding)
    return open(resolved, mode)


def copy_validated_file(src, dst):
    """Copy a file between two pre-validated paths.

    Both src and dst must have been validated by security helpers before
    calling this function.
    """
    shutil.copy2(str(Path(src).resolve()), str(Path(dst).resolve()))


def move_validated_dir(src, dst):
    """Move a pre-validated directory to a pre-validated destination."""
    shutil.move(str(Path(src).resolve()), str(Path(dst).resolve()))


def remove_validated_dir(path):
    """Remove a pre-validated directory tree."""
    shutil.rmtree(str(Path(path).resolve()))
