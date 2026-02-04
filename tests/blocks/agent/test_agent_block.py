# SPDX-License-Identifier: Apache-2.0
"""Tests for AgentBlock."""

from unittest.mock import MagicMock, patch

from sdg_hub.core.blocks.agent import AgentBlock
from sdg_hub.core.blocks.registry import BlockRegistry
from sdg_hub.core.connectors.exceptions import ConnectorError
import pandas as pd
import pytest


class TestAgentBlockRegistration:
    """Test AgentBlock registration."""

    def test_registered_in_block_registry(self):
        """Test AgentBlock is registered."""
        block_class = BlockRegistry._get("AgentBlock")
        assert block_class == AgentBlock

    def test_registered_in_agent_category(self):
        """Test AgentBlock is in agent category."""
        agent_blocks = BlockRegistry.list_blocks(category="agent")
        assert "AgentBlock" in agent_blocks


class TestAgentBlockConfiguration:
    """Test AgentBlock configuration."""

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValueError):
            AgentBlock(
                block_name="test",
                # Missing agent_framework and agent_url
            )

    def test_create_with_minimal_config(self):
        """Test creating block with minimal config."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=["response"],
        )

        assert block.agent_framework == "langflow"
        assert block.agent_url == "http://localhost:7860"
        assert block.timeout == 120.0
        assert block.max_retries == 3
        assert not block.async_mode

    def test_create_with_full_config(self):
        """Test creating block with full config."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            agent_api_key="secret",
            timeout=60.0,
            max_retries=5,
            session_id_col="session",
            async_mode=True,
            max_concurrency=20,
            input_cols=["messages"],
            output_cols=["response"],
        )

        assert block.agent_api_key == "secret"
        assert block.timeout == 60.0
        assert block.max_retries == 5
        assert block.session_id_col == "session"
        assert block.async_mode
        assert block.max_concurrency == 20

    def test_extract_response_defaults_to_false(self):
        """Test extract_response defaults to False."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=["response"],
        )

        assert not block.extract_response

    def test_extract_response_can_be_set(self):
        """Test extract_response can be set to True."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=["response"],
            extract_response=True,
        )

        assert block.extract_response


class TestAgentBlockHelperMethods:
    """Test AgentBlock helper methods."""

    def test_get_messages_col_from_dict(self):
        """Test getting messages column from dict input_cols."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols={"messages": "question"},
            output_cols=["response"],
        )

        # When input_cols is a dict, the value is the DataFrame column name
        assert block._get_messages_col() == "question"

    def test_get_messages_col_from_dict_fallback(self):
        """Test getting messages column from dict without 'messages' key."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols={"query": "user_query"},
            output_cols=["response"],
        )

        # When input_cols is a dict without 'messages' key, use first key
        assert block._get_messages_col() == "query"

    def test_get_messages_col_from_list(self):
        """Test getting messages column from list input_cols."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["question"],
            output_cols=["response"],
        )

        assert block._get_messages_col() == "question"

    def test_get_messages_col_invalid_raises_error(self):
        """Test error when input_cols is invalid."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=[],  # Empty list
            output_cols=["response"],
        )

        with pytest.raises(ConnectorError, match="input_cols must specify"):
            block._get_messages_col()

    def test_get_messages_col_empty_dict_raises_error(self):
        """Test error when input_cols is empty dict."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols={},  # Empty dict
            output_cols=["response"],
        )

        with pytest.raises(ConnectorError, match="input_cols must specify"):
            block._get_messages_col()

    def test_get_output_col_from_list(self):
        """Test getting output column from list."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=["agent_output"],
        )

        assert block._get_output_col() == "agent_output"

    def test_get_output_col_from_dict(self):
        """Test getting output column from dict."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols={"response": "agent_response_col"},
        )

        assert block._get_output_col() == "response"

    def test_get_output_col_default(self):
        """Test default output column name."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=[],
        )

        assert block._get_output_col() == "agent_response"

    def test_build_messages_from_list(self):
        """Test building messages from list."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=["response"],
        )

        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = block._build_messages(messages)
        assert result == messages

    def test_build_messages_from_dict(self):
        """Test building messages from single dict."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=["response"],
        )

        message = {"role": "user", "content": "Hello"}
        result = block._build_messages(message)
        assert result == [message]

    def test_build_messages_from_string(self):
        """Test building messages from plain string."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=["response"],
        )

        result = block._build_messages("Hello, world!")
        assert result == [{"role": "user", "content": "Hello, world!"}]


class TestAgentBlockGenerate:
    """Test AgentBlock generate method."""

    def test_generate_sync_mode(self):
        """Test generate in sync mode."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["question"],
            output_cols=["answer"],
            async_mode=False,
        )

        df = pd.DataFrame(
            {
                "question": ["What is 2+2?", "What is 3+3?"],
            }
        )

        mock_connector = MagicMock()
        mock_connector.send.side_effect = [
            {"output": "4"},
            {"output": "6"},
        ]

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        assert len(result) == 2
        assert "answer" in result.columns
        assert result["answer"].iloc[0] == {"output": "4"}
        assert result["answer"].iloc[1] == {"output": "6"}

    def test_generate_uses_session_id_column(self):
        """Test that generate uses session_id_col if provided."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["question"],
            output_cols=["answer"],
            session_id_col="session",
        )

        df = pd.DataFrame(
            {
                "question": ["Hello"],
                "session": ["session-123"],
            }
        )

        mock_connector = MagicMock()
        mock_connector.send.return_value = {"output": "Hi"}

        with patch.object(block, "_get_connector", return_value=mock_connector):
            block.generate(df)

        # Check that send was called with the session_id from the column
        call_args = mock_connector.send.call_args
        assert call_args[0][1] == "session-123"

    def test_generate_creates_uuid_session_id(self):
        """Test that generate creates UUID if no session_id_col."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["question"],
            output_cols=["answer"],
        )

        df = pd.DataFrame(
            {
                "question": ["Hello"],
            }
        )

        mock_connector = MagicMock()
        mock_connector.send.return_value = {"output": "Hi"}

        with patch.object(block, "_get_connector", return_value=mock_connector):
            block.generate(df)

        # Check that send was called with a UUID-like string
        call_args = mock_connector.send.call_args
        session_id = call_args[0][1]
        assert len(session_id) == 36  # UUID format
        assert session_id.count("-") == 4

    def test_generate_async_mode(self):
        """Test generate in async mode."""
        from unittest.mock import AsyncMock

        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["question"],
            output_cols=["answer"],
            async_mode=True,
            max_concurrency=2,
        )

        df = pd.DataFrame({"question": ["Q1", "Q2"]})

        mock_connector = MagicMock()
        mock_connector.asend = AsyncMock(
            side_effect=[{"output": "A1"}, {"output": "A2"}]
        )

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        assert len(result) == 2
        assert "answer" in result.columns

    @pytest.mark.asyncio
    async def test_generate_async_mode_from_async_context(self):
        """Test generate in async mode when called from within async context."""
        from unittest.mock import AsyncMock

        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["question"],
            output_cols=["answer"],
            async_mode=True,
            max_concurrency=2,
        )

        df = pd.DataFrame({"question": ["Q1", "Q2"]})

        mock_connector = MagicMock()
        mock_connector.asend = AsyncMock(
            side_effect=[{"output": "A1"}, {"output": "A2"}]
        )

        with patch.object(block, "_get_connector", return_value=mock_connector):
            # This is called from within an async context, testing ThreadPoolExecutor path
            result = block.generate(df)

        assert len(result) == 2
        assert "answer" in result.columns


class TestAgentBlockConnectorIntegration:
    """Test AgentBlock connector integration."""

    def test_get_connector_creates_correct_connector(self):
        """Test that _get_connector creates the right connector type."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            agent_api_key="secret",
            timeout=60.0,
            max_retries=5,
            input_cols=["messages"],
            output_cols=["response"],
        )

        connector = block._get_connector()

        assert connector.__class__.__name__ == "LangflowConnector"
        assert connector.config.url == "http://localhost:7860"
        assert connector.config.api_key == "secret"
        assert connector.config.timeout == 60.0
        assert connector.config.max_retries == 5

    def test_get_connector_caches_instance(self):
        """Test that _get_connector caches the connector."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=["response"],
        )

        connector1 = block._get_connector()
        connector2 = block._get_connector()

        assert connector1 is connector2

    def test_get_connector_invalid_framework_raises_error(self):
        """Test that invalid framework raises ConnectorError."""
        block = AgentBlock(
            block_name="test",
            agent_framework="nonexistent",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=["response"],
        )

        with pytest.raises(ConnectorError, match="not found"):
            block._get_connector()

    def test_get_connector_invalidates_on_config_change(self):
        """Test that _get_connector creates new connector when config changes."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=["response"],
        )

        connector1 = block._get_connector()
        assert connector1.config.url == "http://localhost:7860"

        # Simulate runtime override by changing the URL
        block.agent_url = "http://newhost:8080"
        connector2 = block._get_connector()

        assert connector1 is not connector2
        assert connector2.config.url == "http://newhost:8080"

    def test_get_connector_passes_extract_text(self):
        """Test that _get_connector passes extract_text to config."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=["response"],
            extract_response=True,
        )

        connector = block._get_connector()
        assert connector.config.extract_text is True

    def test_get_connector_invalidates_on_extract_response_change(self):
        """Test that _get_connector invalidates cache when extract_response changes."""
        block = AgentBlock(
            block_name="test",
            agent_framework="langflow",
            agent_url="http://localhost:7860",
            input_cols=["messages"],
            output_cols=["response"],
            extract_response=False,
        )

        connector1 = block._get_connector()
        assert connector1.config.extract_text is False

        # Change extract_response
        block.extract_response = True
        connector2 = block._get_connector()

        assert connector1 is not connector2
        assert connector2.config.extract_text is True
