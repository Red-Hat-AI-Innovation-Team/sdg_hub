# SPDX-License-Identifier: Apache-2.0
"""Verify every concrete BaseBlock subclass is registered in BlockRegistry.

If this test fails, a block class exists that is not decorated with
@BlockRegistry.register(...).  See docs/agent-knowledge/block-invariants.md
for the registration contract.
"""

import importlib
import inspect
import pkgutil

from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry
import sdg_hub.core.blocks as blocks_pkg


def _import_all_block_modules() -> None:
    """Force-import every module under sdg_hub.core.blocks so that all
    @BlockRegistry.register decorators execute."""
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        blocks_pkg.__path__, prefix=blocks_pkg.__name__ + "."
    ):
        importlib.import_module(modname)


def _collect_concrete_subclasses(base: type) -> set[type]:
    """Recursively collect all non-abstract subclasses of *base*."""
    concrete: set[type] = set()
    for sub in base.__subclasses__():
        if not inspect.isabstract(sub):
            concrete.add(sub)
        concrete |= _collect_concrete_subclasses(sub)
    # Filter out abstract classes that were added via recursion
    return {cls for cls in concrete if not inspect.isabstract(cls)}


def test_all_concrete_blocks_are_registered() -> None:
    """Every concrete BaseBlock subclass must appear in BlockRegistry."""
    _import_all_block_modules()

    concrete_classes = _collect_concrete_subclasses(BaseBlock)
    # Exclude test-only mock/dummy blocks defined outside src/
    concrete_classes = {
        cls for cls in concrete_classes if cls.__module__.startswith("sdg_hub")
    }
    registered_classes = {
        BlockRegistry._metadata[name].block_class
        for name in BlockRegistry.list_blocks()
    }

    unregistered = concrete_classes - registered_classes
    assert not unregistered, (
        f"The following concrete BaseBlock subclasses are not registered "
        f"in BlockRegistry: {sorted(cls.__name__ for cls in unregistered)}. "
        f"See docs/agent-knowledge/block-invariants.md for the registration contract."
    )
