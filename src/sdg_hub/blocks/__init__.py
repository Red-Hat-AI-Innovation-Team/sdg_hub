"""Block implementations for SDG Hub.

This package provides various block implementations for data generation, processing, and transformation.
"""

# Local
from ..registry import BlockRegistry
from .block import Block
from .deprecated_blocks import DuplicateColumns, FilterByValueBlock, FlattenColumnsBlock, RenameColumns, SetToMajorityValue
from .filtering import ColumnValueFilterBlock
from .llm import LLMChatBlock, PromptBuilderBlock, TextParserBlock
from .llmblock import ConditionalLLMBlock, LLMBlock
from .transform import DuplicateColumnsBlock, MeltColumnsBlock, RenameColumnsBlock, UniformColumnValueSetter
from .utilblocks import (
    CombineColumnsBlock,
    SamplePopulatorBlock,
    SelectorBlock,
)

__all__ = [
    "Block",
    "ColumnValueFilterBlock",
    "DuplicateColumnsBlock",
    "MeltColumnsBlock",
    "RenameColumnsBlock",
    "UniformColumnValueSetter",
    "DuplicateColumns",  # Deprecated
    "FilterByValueBlock",  # Deprecated
    "FlattenColumnsBlock",  # Deprecated
    "RenameColumns",  # Deprecated
    "SetToMajorityValue",  # Deprecated
    "LLMBlock",
    "ConditionalLLMBlock",
    "LLMChatBlock",
    "TextParserBlock",
    "SamplePopulatorBlock",
    "SelectorBlock",
    "CombineColumnsBlock",
    "BlockRegistry",
    "PromptBuilderBlock",
]
