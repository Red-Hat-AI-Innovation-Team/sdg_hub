# SPDX-License-Identifier: Apache-2.0
"""Data transformation blocks for dataset manipulation.

This module provides blocks for transforming datasets including column operations,
wide-to-long transformations, value selection, and majority value assignment.
"""

from .text_concat import TextConcatBlock
from .duplicate_columns import DuplicateColumnsBlock
from .melt_columns import MeltColumnsBlock
from .index_based_mapper import IndexBasedMapperBlock
from .rename_columns import RenameColumnsBlock
from .uniform_col_val_setter import UniformColumnValueSetter

__all__ = [
    "TextConcatBlock",
    "DuplicateColumnsBlock",
    "MeltColumnsBlock",
    "IndexBasedMapperBlock",
    "RenameColumnsBlock",
    "UniformColumnValueSetter",
]
