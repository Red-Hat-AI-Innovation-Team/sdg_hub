# SPDX-License-Identifier: Apache-2.0
"""New flow implementation for SDG Hub.

This module provides a redesigned Flow class with metadata support
and dual initialization modes.
"""

# Local
from .base import Flow
from .metadata import FlowMetadata
from .registry import FlowRegistry
from .validation import FlowValidator

# Import submodules to make them available for patching in tests
from . import display  # noqa: F401
from . import execution  # noqa: F401
from . import model_config  # noqa: F401
from . import serialization  # noqa: F401

__all__ = [
    "Flow",
    "FlowMetadata",
    "FlowRegistry",
    "FlowValidator",
]
