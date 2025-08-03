# SPDX-License-Identifier: Apache-2.0

# This is part of the public API, and used by instructlab
class GenerateException(Exception):
    """An exception raised during generate step."""


from .path_resolution import resolve_path

__all__ = ["GenerateException", "resolve_path"]
