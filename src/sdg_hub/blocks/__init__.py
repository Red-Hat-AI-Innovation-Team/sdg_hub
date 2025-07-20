"""Block implementations for SDG Hub.

This package provides various block implementations for data generation, processing, and transformation.
"""

# Local
from ..registry import BlockRegistry
from .block import Block
from .deprecated_blocks import FilterByValueBlock, FlattenColumnsBlock
from .filtering import ColumnValueFilterBlock
from .llm import LLMChatBlock, PromptBuilderBlock, TextParserBlock
from .llmblock import ConditionalLLMBlock, LLMBlock
from .transform import MeltColumnsBlock
from .utilblocks import (
    CombineColumnsBlock,
    DuplicateColumns,
    RenameColumns,
    SamplePopulatorBlock,
    SelectorBlock,
    SetToMajorityValue,
)

__all__ = [
    "Block",
    "ColumnValueFilterBlock",
    "MeltColumnsBlock",
    "FilterByValueBlock",  # Deprecated
    "FlattenColumnsBlock",  # Deprecated
    "LLMBlock",
    "ConditionalLLMBlock",
    "LLMChatBlock",
    "TextParserBlock",
    "SamplePopulatorBlock",
    "SelectorBlock",
    "CombineColumnsBlock",
    "DuplicateColumns",
    "RenameColumns",
    "SetToMajorityValue",
    "BlockRegistry",
    "PromptBuilderBlock",
]
