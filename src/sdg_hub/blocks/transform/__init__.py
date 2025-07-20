# SPDX-License-Identifier: Apache-2.0
"""Data transformation blocks for dataset manipulation.

This module provides blocks for transforming datasets including column operations,
wide-to-long transformations, value selection, and majority value assignment.
"""

from .flatten_columns import FlattenColumnsBlock
from .index_based_mapper import IndexBasedMapperBlock
from .uniform_col_val_setter import UniformColumnValueSetter

__all__ = [
    "FlattenColumnsBlock",
    "IndexBasedMapperBlock",
    "UniformColumnValueSetter",
]
