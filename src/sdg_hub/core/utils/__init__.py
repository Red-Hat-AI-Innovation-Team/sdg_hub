# SPDX-License-Identifier: Apache-2.0

# Local
from .path_resolution import resolve_path


# This is part of the public API, and used by instructlab
class GenerateError(Exception):
    """An exception raised during generate step."""


__all__ = ["GenerateError", "resolve_path"]
