"""Block implementations for SDG Hub.

This package provides various block implementations for data generation, processing, and transformation.
"""

# Local
from .agent import AgentBlock
from .base import BaseBlock
from .filtering import ColumnValueFilterBlock
from .llm import (
    LLMChatBlock,
    LLMResponseExtractorBlock,
    PromptBuilderBlock,
)

try:
    from .mcp import MCPAgentBlock
except ImportError as _err:
    if "mcp" not in str(_err).lower():
        raise  # Don't mask unrelated ImportErrors
    MCPAgentBlock = None  # type: ignore[assignment, misc]
from .parsing import JSONParserBlock, RegexParserBlock, TagParserBlock, TextParserBlock
from .registry import BlockRegistry
from .transform import (
    DuplicateColumnsBlock,
    IndexBasedMapperBlock,
    MeltColumnsBlock,
    RenameColumnsBlock,
    TextConcatBlock,
    UniformColumnValueSetter,
)

__all__ = [
    "AgentBlock",
    "BaseBlock",
    "BlockRegistry",
    "ColumnValueFilterBlock",
    "DuplicateColumnsBlock",
    "IndexBasedMapperBlock",
    "JSONParserBlock",
    "MeltColumnsBlock",
    "PromptBuilderBlock",
    "RegexParserBlock",
    "RenameColumnsBlock",
    "TagParserBlock",
    "TextConcatBlock",
    "TextParserBlock",
    "UniformColumnValueSetter",
    "LLMChatBlock",
    "LLMResponseExtractorBlock",
    "MCPAgentBlock",
]
