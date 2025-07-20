# SPDX-License-Identifier: Apache-2.0
"""Deprecated blocks for backwards compatibility.

This module contains deprecated block implementations that are maintained
for backwards compatibility. These blocks should not be used in new code.
"""

# Local
from .duplicate_columns import DuplicateColumns
from .filter_by_value import FilterByValueBlock
from .flatten_columns import FlattenColumnsBlock
from .rename_columns import RenameColumns
from .set_to_majority_value import SetToMajorityValue

__all__ = [
    "DuplicateColumns",
    "FilterByValueBlock",
    "FlattenColumnsBlock",
    "RenameColumns",
    "SetToMajorityValue",
]