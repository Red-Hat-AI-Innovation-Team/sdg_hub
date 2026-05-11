# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures and helpers for the SDG Hub test suite."""

import pandas as pd
import pytest


class MockMessage:
    """Mock message class that behaves like a LiteLLM message.

    Supports both the dict-like interface used by LLMChatBlock and
    the model_dump() interface used by MCPAgentBlock.
    """

    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def __iter__(self):
        return iter(["content"])

    def __getitem__(self, key):
        if key == "content":
            return self.content
        raise KeyError(key)

    def keys(self):
        return ["content"]

    def values(self):
        return [self.content]

    def items(self):
        return [("content", self.content)]

    def model_dump(self):
        result = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return result


@pytest.fixture
def mock_message_class():
    """Provides the MockMessage class for constructing mock LLM responses."""
    return MockMessage


@pytest.fixture
def sample_dataset():
    """Generic sample dataset for block and flow tests."""
    return pd.DataFrame(
        {
            "question": ["Question 1", "Question 2", "Question 3"],
            "context": ["Context 1", "Context 2", "Context 3"],
            "input": ["test input 1", "test input 2", "test input 3"],
            "label": ["label1", "label2", "label3"],
            "other_col": ["Other 1", "Other 2", "Other 3"],
        }
    )
