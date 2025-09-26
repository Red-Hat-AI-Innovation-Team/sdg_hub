"""
Path resolution utilities for SDG Hub.

This module provides utilities for resolving file paths relative to one or more
base directories, with support for both single directory and multiple directory
search paths.
"""

# Standard
from typing import Union
import os


def resolve_path(filename: str, search_dirs: Union[str, list[str]]) -> str:
    """Resolve a file path relative to one or more search directories.

    Files are checked in the following order:
        1. Absolute path is always used as-is
        2. Checked relative to each directory in search_dirs (in order)
        3. If not found, returns the original filename (assumes relative to current directory)

    Parameters
    ----------
    filename : str
        The path to the file to resolve.
    search_dirs : Union[str, List[str]]
        Directory or list of directories in which to search for the file.

    Returns
    -------
    str
        Resolved file path.

    Examples
    --------
    >>> resolve_path("config.yaml", "/path/to/base")
    '/path/to/base/config.yaml'  # if file exists

    >>> resolve_path("config.yaml", ["/path1", "/path2"])
    '/path1/config.yaml'  # if file exists in path1
    '/path2/config.yaml'  # if file exists in path2 but not path1

    >>> resolve_path("/absolute/path/file.yaml", ["/path1", "/path2"])
    '/absolute/path/file.yaml'  # absolute path always used as-is
    """
    # Handle absolute paths - always use as-is
    if os.path.isabs(filename):
        return filename

    # Convert single directory to list for uniform handling
    if isinstance(search_dirs, str):
        search_dirs = [search_dirs]

    # Check each directory in order
    for directory in search_dirs:
        full_file_path = os.path.join(directory, filename)
        if os.path.isfile(full_file_path):
            return full_file_path

    # If not found in any search directory, return the original filename
    # This assumes the path is relative to the current directory
    return filename
