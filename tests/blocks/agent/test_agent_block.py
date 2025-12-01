# SPDX-License-Identifier: Apache-2.0
"""Tests for AgentBlock."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sdg_hub.core.blocks.agent import AgentBlock
from sdg_hub.core.utils.error_handling import BlockValidationError


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "messages": [
                [{"role": "user", "content": "Hello"}],
                [{"role": "user", "content": "How are you?"}],
            ]
        }
    )


@pytest.fixture
def mock_langflow_wrapper():
    """Mock LangflowAgentWrapper."""
    with patch(
        "sdg_hub.core.blocks.agent.agent_wrapper.langflow_agent_wrapper.LangflowAgentWrapper"
    ) as mock_wrapper_class:
        mock_instance = MagicMock()
        mock_instance.generate.return_value = {"response": "Test response"}
        mock_wrapper_class.return_value = mock_instance
        yield mock_instance


class TestAgentBlockInit:
    """Test AgentBlock initialization."""

    def test_init_with_required_params(self, mock_langflow_wrapper):
        """Test initialization with required parameters."""
        block = AgentBlock(
            block_name="test_agent",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        assert block.block_name == "test_agent"
        assert block.input_cols == ["messages"]
        assert block.output_cols == ["response"]
        assert block.agent_framework == "langflow"
        assert block.agent_url == "http://localhost:8000"
        assert block.timeout == 120.0  # default
        assert block._agent_wrapper is not None

    def test_init_with_all_params(self, mock_langflow_wrapper):
        """Test initialization with all parameters."""
        block = AgentBlock(
            block_name="test_agent",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
            agent_api_key="test-key",
            timeout=60.0,
        )

        assert block.agent_api_key == "test-key"
        assert block.timeout == 60.0

    def test_init_wrapper_initialized_once(self, mock_langflow_wrapper):
        """Test that wrapper is initialized only once."""
        block = AgentBlock(
            block_name="test_agent",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        wrapper1 = block._agent_wrapper
        wrapper2 = block._agent_wrapper

        assert wrapper1 is wrapper2


class TestAgentBlockValidation:
    """Test AgentBlock field validation."""

    def test_missing_agent_framework(self):
        """Test that missing agent_framework raises ValidationError."""
        with pytest.raises(Exception, match="Field required"):
            AgentBlock(
                block_name="test",
                input_cols="messages",
                output_cols="response",
                agent_url="http://localhost:8000",
            )

    def test_invalid_agent_framework(self):
        """Test that invalid agent_framework raises ValidationError."""
        with pytest.raises(Exception, match="Unsupported agent framework"):
            AgentBlock(
                block_name="test",
                input_cols="messages",
                output_cols="response",
                agent_framework="invalid_framework",
                agent_url="http://localhost:8000",
            )

    def test_missing_agent_url(self):
        """Test that missing agent_url raises ValidationError."""
        with pytest.raises(Exception, match="Field required"):
            AgentBlock(
                block_name="test",
                input_cols="messages",
                output_cols="response",
                agent_framework="langflow",
            )

    def test_multiple_input_cols(self):
        """Test that multiple input columns raises ValidationError."""
        with pytest.raises(Exception, match="exactly one input column"):
            AgentBlock(
                block_name="test",
                input_cols=["messages", "context"],
                output_cols="response",
                agent_framework="langflow",
                agent_url="http://localhost:8000",
            )

    def test_multiple_output_cols(self):
        """Test that multiple output columns raises ValidationError."""
        with pytest.raises(Exception, match="exactly one output column"):
            AgentBlock(
                block_name="test",
                input_cols="messages",
                output_cols=["response", "metadata"],
                agent_framework="langflow",
                agent_url="http://localhost:8000",
            )


class TestMessageConversion:
    """Test message conversion methods."""

    def test_message_to_dict_with_dict(self, mock_langflow_wrapper):
        """Test _message_to_dict() with dict input."""
        block = AgentBlock(
            block_name="test",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        message = {"role": "user", "content": "Hello"}
        result = block._message_to_dict(message)

        assert result == message

    def test_message_to_dict_with_object(self, mock_langflow_wrapper):
        """Test _message_to_dict() with object input."""
        block = AgentBlock(
            block_name="test",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        class Message:
            def __init__(self):
                self.content = "Hello"
                self.role = "user"

        message = Message()
        result = block._message_to_dict(message)

        assert result["content"] == "Hello"

    def test_messages_to_dict_with_list(self, mock_langflow_wrapper):
        """Test _messages_to_dict() with list input."""
        block = AgentBlock(
            block_name="test",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = block._messages_to_dict(messages)

        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Hi"

    def test_messages_to_dict_with_single_dict(self, mock_langflow_wrapper):
        """Test _messages_to_dict() with single dict input."""
        block = AgentBlock(
            block_name="test",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        message = {"role": "user", "content": "Hello"}
        result = block._messages_to_dict(message)

        assert len(result) == 1
        assert result[0] == message


class TestSessionIdGeneration:
    """Test session ID generation."""

    def test_generates_unique_session_ids(self, mock_langflow_wrapper):
        """Test that each call generates a unique session ID."""
        block = AgentBlock(
            block_name="test",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        id1 = block._generate_session_id()
        id2 = block._generate_session_id()

        # Each call should generate a different UUID
        assert id1 != id2
        # Should be valid UUIDs (length check)
        assert len(id1) == 36
        assert len(id2) == 36


class TestGenerate:
    """Test generate() method."""

    def test_generate_success(self, sample_dataframe, mock_langflow_wrapper):
        """Test successful generation."""
        block = AgentBlock(
            block_name="test",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        result = block.generate(sample_dataframe)

        # Check wrapper was called twice (once per row)
        assert mock_langflow_wrapper.generate.call_count == 2

        # Check result DataFrame
        assert "response" in result.columns
        assert len(result) == 2
        assert all(result["response"] == {"response": "Test response"})

    def test_generate_uses_unique_session_ids(
        self, sample_dataframe, mock_langflow_wrapper
    ):
        """Test that generation uses unique session IDs for each sample."""
        block = AgentBlock(
            block_name="test",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        result = block.generate(sample_dataframe)

        # Get all session IDs used
        session_ids = [
            call[0][1] for call in mock_langflow_wrapper.generate.call_args_list
        ]

        # All should be different
        assert len(set(session_ids)) == 2
        assert session_ids[0] != session_ids[1]

    def test_generate_wrapper_error_propagates(
        self, sample_dataframe, mock_langflow_wrapper
    ):
        """Test that wrapper errors propagate correctly."""
        mock_langflow_wrapper.generate.side_effect = BlockValidationError(
            "Wrapper error"
        )

        block = AgentBlock(
            block_name="test",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        with pytest.raises(BlockValidationError, match="Wrapper error"):
            block.generate(sample_dataframe)

    def test_generate_unexpected_error_wrapped(
        self, sample_dataframe, mock_langflow_wrapper
    ):
        """Test that unexpected errors are wrapped in BlockValidationError."""
        mock_langflow_wrapper.generate.side_effect = RuntimeError("Unexpected error")

        block = AgentBlock(
            block_name="test",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        with pytest.raises(BlockValidationError, match="Unexpected error"):
            block.generate(sample_dataframe)

    def test_generate_preserves_original_columns(
        self, sample_dataframe, mock_langflow_wrapper
    ):
        """Test that generate preserves original DataFrame columns."""
        block = AgentBlock(
            block_name="test",
            input_cols="messages",
            output_cols="response",
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        result = block.generate(sample_dataframe)

        # Original column should still be there
        assert "messages" in result.columns
        assert all(result["messages"] == sample_dataframe["messages"])
