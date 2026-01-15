# SPDX-License-Identifier: Apache-2.0
"""LLM blocks for provider-agnostic text generation.

This module provides blocks for interacting with language models through
LiteLLM, supporting 100+ providers including OpenAI, Anthropic, Google,
local models (vLLM, Ollama), and more.
"""

# Local
from .error_handler import ErrorCategory, LLMErrorHandler
from .llm_chat_block import LLMChatBlock
from .llm_response_extractor_block import LLMParserBlock, LLMResponseExtractorBlock
from .prompt_builder_block import PromptBuilderBlock
from .text_parser_block import TextParserBlock

__all__ = [
    "LLMErrorHandler",
    "ErrorCategory",
    "LLMChatBlock",
    "LLMParserBlock",  # Deprecated alias for LLMResponseExtractorBlock
    "LLMResponseExtractorBlock",
    "PromptBuilderBlock",
    "TextParserBlock",
]
