# SPDX-License-Identifier: Apache-2.0
"""LLM blocks for provider-agnostic text generation.

This module provides blocks for interacting with language models through
LiteLLM, supporting 100+ providers including OpenAI, Anthropic, Google,
local models (vLLM, Ollama), and more.
"""

# Local
from .client_manager import LLMClientManager
from .config import LLMConfig
from .error_handler import ErrorCategory, LLMErrorHandler
from .llm_chat_block import LLMChatBlock
from .prompt_builder_block import PromptBuilderBlock
from .text_parser_block import TextParserBlock

__all__ = [
    "LLMConfig",
    "LLMClientManager",
    "LLMErrorHandler",
    "ErrorCategory",
    "LLMChatBlock",
    "PromptBuilderBlock",
    "TextParserBlock",
]
