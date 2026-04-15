# SPDX-License-Identifier: Apache-2.0
"""New flow implementation for SDG Hub.

This module provides a redesigned Flow class with metadata support
and dual initialization modes.
"""

# Local
# Import submodules to make them available for patching in tests.
# Using redundant-alias idiom (import X as X) to signal intentional re-export to ruff
# without promoting these internal modules to public API via __all__.
from . import (
    agent_config as agent_config,
)
from . import (
    display as display,
)
from . import (
    execution as execution,
)
from . import (
    model_config as model_config,
)
from . import (
    serialization as serialization,
)
from .base import Flow
from .metadata import FlowMetadata
from .registry import FlowRegistry
from .validation import FlowValidator

__all__ = [
    "Flow",
    "FlowMetadata",
    "FlowRegistry",
    "FlowValidator",
]
