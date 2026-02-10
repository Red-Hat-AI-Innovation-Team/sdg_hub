# SPDX-License-Identifier: Apache-2.0
"""Tests for the MCPAgentBlock with mocked MCP session."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest


class MockMessage:
    """Mock message class that behaves like LiteLLM message."""

    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

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


class MockToolCall:
    """Mock tool call from LLM response."""

    def __init__(self, tool_id, name, arguments):
        self.id = tool_id
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments


class MockTool:
    """Mock MCP tool."""

    def __init__(self, name, description, input_schema=None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object", "properties": {}}


class MockToolResult:
    """Mock MCP tool result."""

    def __init__(self, text):
        self.content = [MagicMock(text=text)]


class MockToolsResponse:
    """Mock response from session.list_tools()."""

    def __init__(self, tools):
        self.tools = tools


@pytest.fixture
def sample_dataset():
    """Create a sample dataset with queries."""
    return pd.DataFrame({"question": ["What is Python?", "How do I use asyncio?"]})


@pytest.fixture
def mock_mcp_session():
    """Create a mock MCP ClientSession."""
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(
        return_value=MockToolsResponse(
            [
                MockTool(
                    "search",
                    "Search the web",
                    {"type": "object", "properties": {"query": {"type": "string"}}},
                ),
                MockTool(
                    "read_docs",
                    "Read documentation",
                    {"type": "object", "properties": {"topic": {"type": "string"}}},
                ),
            ]
        )
    )
    session.call_tool = AsyncMock(return_value=MockToolResult("Tool result text"))
    return session


@pytest.fixture
def mock_sse_client(mock_mcp_session):
    """Mock the streamablehttp_client context manager."""

    class MockStream:
        pass

    with patch(
        "sdg_hub.core.blocks.mcp.mcp_agent_block.streamablehttp_client"
    ) as mock_http:
        # Create async context manager that yields (read, write, get_session_id)
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = (MockStream(), MockStream(), None)
        mock_cm.__aexit__.return_value = None
        mock_http.return_value = mock_cm
        yield mock_http


@pytest.fixture
def mock_client_session(mock_mcp_session):
    """Mock the ClientSession context manager."""
    with patch("sdg_hub.core.blocks.mcp.mcp_agent_block.ClientSession") as mock_cls:
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_mcp_session
        mock_cm.__aexit__.return_value = None
        mock_cls.return_value = mock_cm
        yield mock_cls, mock_mcp_session


class TestMCPAgentBlockInit:
    """Tests for MCPAgentBlock initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        block = MCPAgentBlock(
            block_name="test_agent",
            mcp_server_url="https://mcp.example.com/mcp",
            model="openai/gpt-4o",
            input_cols="question",
            output_cols="answer",
        )

        assert block.block_name == "test_agent"
        assert block.mcp_server_url == "https://mcp.example.com/mcp"
        assert block.model == "openai/gpt-4o"
        assert block.input_cols == ["question"]
        assert block.output_cols == ["answer"]
        assert block.max_iterations == 10
        assert block.system_prompt is None

    def test_init_with_all_options(self):
        """Test initialization with all options."""
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        block = MCPAgentBlock(
            block_name="full_agent",
            mcp_server_url="https://mcp.example.com/mcp",
            mcp_headers={"Authorization": "Bearer token123"},
            model="openai/gpt-4o",
            api_key="test-key",
            api_base="https://api.example.com/v1",
            max_iterations=5,
            system_prompt="You are a helpful assistant.",
            input_cols=["query"],
            output_cols=["response"],
        )

        assert block.mcp_headers == {"Authorization": "Bearer token123"}
        assert block.api_key.get_secret_value() == "test-key"
        assert block.api_base == "https://api.example.com/v1"
        assert block.max_iterations == 5
        assert block.system_prompt == "You are a helpful assistant."

    def test_init_multiple_input_cols_error(self):
        """Test error when multiple input columns provided."""
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        with pytest.raises(ValueError, match="expects exactly one input column"):
            MCPAgentBlock(
                block_name="test_block",
                mcp_server_url="https://mcp.example.com/mcp",
                model="openai/gpt-4o",
                input_cols=["question1", "question2"],
                output_cols="answer",
            )

    def test_init_multiple_output_cols_error(self):
        """Test error when multiple output columns provided."""
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        with pytest.raises(ValueError, match="expects exactly one output column"):
            MCPAgentBlock(
                block_name="test_block",
                mcp_server_url="https://mcp.example.com/mcp",
                model="openai/gpt-4o",
                input_cols="question",
                output_cols=["answer1", "answer2"],
            )

    def test_init_invalid_max_iterations(self):
        """Test error when max_iterations is invalid."""
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        with pytest.raises(ValueError, match="max_iterations must be at least 1"):
            MCPAgentBlock(
                block_name="test_block",
                mcp_server_url="https://mcp.example.com/mcp",
                model="openai/gpt-4o",
                input_cols="question",
                output_cols="answer",
                max_iterations=0,
            )


class TestMCPAgentBlockGeneration:
    """Tests for MCPAgentBlock generation."""

    def test_generation_no_tool_calls(
        self, sample_dataset, mock_sse_client, mock_client_session
    ):
        """Test generation when LLM responds without tool calls."""
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        mock_cls, mock_session = mock_client_session

        # Mock LLM response without tool calls
        with patch(
            "sdg_hub.core.blocks.mcp.mcp_agent_block.acompletion"
        ) as mock_acompletion:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MockMessage("Python is a programming language."))
            ]

            async def mock_completion(*args, **kwargs):
                return mock_response

            mock_acompletion.side_effect = mock_completion

            block = MCPAgentBlock(
                block_name="test_agent",
                mcp_server_url="https://mcp.example.com/mcp",
                model="openai/gpt-4o",
                input_cols="question",
                output_cols="answer",
            )

            result = block.generate(sample_dataset)

            assert "answer" in result.columns.tolist()
            assert len(result["answer"]) == 2
            # Output is now a trace dict
            trace = result["answer"].iloc[0]
            assert isinstance(trace, dict)
            assert "messages" in trace
            assert "iterations" in trace
            assert trace["max_iterations_reached"] is False
            # Check final assistant message contains the response
            final_msg = trace["messages"][-1]
            assert final_msg["content"] == "Python is a programming language."

    def test_generation_with_tool_calls(self, mock_sse_client, mock_client_session):
        """Test generation when LLM makes tool calls."""
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        mock_cls, mock_session = mock_client_session

        # Create dataset with single query
        dataset = pd.DataFrame({"question": ["What is the weather?"]})

        with patch(
            "sdg_hub.core.blocks.mcp.mcp_agent_block.acompletion"
        ) as mock_acompletion:
            # First response: tool call
            tool_call = MockToolCall("call_123", "search", '{"query": "weather"}')
            first_response = MagicMock()
            first_response.choices = [MagicMock(message=MockMessage(None, [tool_call]))]

            # Second response: final answer
            second_response = MagicMock()
            second_response.choices = [
                MagicMock(message=MockMessage("The weather is sunny."))
            ]

            call_count = 0

            async def mock_completion(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return first_response
                return second_response

            mock_acompletion.side_effect = mock_completion

            block = MCPAgentBlock(
                block_name="test_agent",
                mcp_server_url="https://mcp.example.com/mcp",
                model="openai/gpt-4o",
                input_cols="question",
                output_cols="answer",
            )

            result = block.generate(dataset)

            assert "answer" in result.columns.tolist()
            # Output is now a trace dict
            trace = result["answer"].iloc[0]
            assert isinstance(trace, dict)
            assert "messages" in trace
            assert trace["iterations"] == 2  # One tool call iteration + final response
            # Check final message is the answer
            final_msg = trace["messages"][-1]
            assert final_msg["content"] == "The weather is sunny."
            assert mock_session.call_tool.called

    def test_generation_with_system_prompt(self, mock_sse_client, mock_client_session):
        """Test generation includes system prompt in messages."""
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        mock_cls, mock_session = mock_client_session

        dataset = pd.DataFrame({"question": ["Hello"]})

        with patch(
            "sdg_hub.core.blocks.mcp.mcp_agent_block.acompletion"
        ) as mock_acompletion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MockMessage("Hi there!"))]

            async def mock_completion(*args, **kwargs):
                # Verify system prompt is included
                messages = kwargs.get("messages", [])
                assert len(messages) >= 2
                assert messages[0]["role"] == "system"
                assert messages[0]["content"] == "You are helpful."
                return mock_response

            mock_acompletion.side_effect = mock_completion

            block = MCPAgentBlock(
                block_name="test_agent",
                mcp_server_url="https://mcp.example.com/mcp",
                model="openai/gpt-4o",
                system_prompt="You are helpful.",
                input_cols="question",
                output_cols="answer",
            )

            result = block.generate(dataset)
            trace = result["answer"].iloc[0]
            assert trace["messages"][-1]["content"] == "Hi there!"

    def test_generation_max_iterations_reached(
        self, mock_sse_client, mock_client_session
    ):
        """Test generation handles max iterations limit."""
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        mock_cls, mock_session = mock_client_session

        dataset = pd.DataFrame({"question": ["Infinite loop query"]})

        with patch(
            "sdg_hub.core.blocks.mcp.mcp_agent_block.acompletion"
        ) as mock_acompletion:
            # Always return tool calls (never completes)
            tool_call = MockToolCall("call_123", "search", '{"query": "test"}')
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MockMessage(None, [tool_call]))]

            async def mock_completion(*args, **kwargs):
                return mock_response

            mock_acompletion.side_effect = mock_completion

            block = MCPAgentBlock(
                block_name="test_agent",
                mcp_server_url="https://mcp.example.com/mcp",
                model="openai/gpt-4o",
                max_iterations=3,
                input_cols="question",
                output_cols="answer",
            )

            result = block.generate(dataset)

            # Should return trace with max_iterations_reached flag
            assert "answer" in result.columns.tolist()
            trace = result["answer"].iloc[0]
            assert isinstance(trace, dict)
            assert trace["max_iterations_reached"] is True
            assert trace["iterations"] == 3


class TestMCPAgentBlockHelpers:
    """Tests for MCPAgentBlock helper methods."""

    def test_to_openai_format(self):
        """Test conversion of MCP tools to OpenAI format."""
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        block = MCPAgentBlock(
            block_name="test_agent",
            mcp_server_url="https://mcp.example.com/mcp",
            model="openai/gpt-4o",
            input_cols="question",
            output_cols="answer",
        )

        mcp_tools = [
            MockTool(
                "search",
                "Search the web",
                {"type": "object", "properties": {"query": {"type": "string"}}},
            ),
            MockTool(
                "calculate",
                "Do math",
                {"type": "object", "properties": {"expression": {"type": "string"}}},
            ),
        ]

        openai_tools = block._to_openai_format(mcp_tools)

        assert len(openai_tools) == 2
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "search"
        assert openai_tools[0]["function"]["description"] == "Search the web"
        assert openai_tools[1]["function"]["name"] == "calculate"

    def test_serialize_tool_result_with_text_content(self):
        """Test serialization of tool result to JSON."""
        import json

        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        block = MCPAgentBlock(
            block_name="test_agent",
            mcp_server_url="https://mcp.example.com/mcp",
            model="openai/gpt-4o",
            input_cols="question",
            output_cols="answer",
        )

        result = MockToolResult("This is the tool output")
        serialized = block._serialize_tool_result(result)

        # Should be valid JSON
        parsed = json.loads(serialized)
        assert isinstance(parsed, list)
        assert parsed[0]["type"] == "text"
        assert parsed[0]["text"] == "This is the tool output"

    def test_build_completion_kwargs(self):
        """Test building completion kwargs."""
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        block = MCPAgentBlock(
            block_name="test_agent",
            mcp_server_url="https://mcp.example.com/mcp",
            model="openai/gpt-4o",
            api_key="test-key",
            api_base="https://api.example.com/v1",
            input_cols="question",
            output_cols="answer",
        )

        kwargs = block._build_completion_kwargs()

        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["api_key"] == "test-key"
        assert kwargs["api_base"] == "https://api.example.com/v1"

    def test_repr(self):
        """Test string representation."""
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        block = MCPAgentBlock(
            block_name="my_agent",
            mcp_server_url="https://mcp.example.com/mcp",
            model="openai/gpt-4o",
            max_iterations=5,
            input_cols="question",
            output_cols="answer",
        )

        repr_str = repr(block)

        assert "MCPAgentBlock" in repr_str
        assert "my_agent" in repr_str
        assert "openai/gpt-4o" in repr_str
        assert "https://mcp.example.com/mcp" in repr_str
        assert "max_iterations=5" in repr_str


class TestMCPAgentBlockRegistration:
    """Test block registration."""

    def test_block_registered(self):
        """Test that MCPAgentBlock is properly registered."""
        from sdg_hub import BlockRegistry
        from sdg_hub.core.blocks.mcp import MCPAgentBlock

        assert "MCPAgentBlock" in BlockRegistry._metadata
        assert BlockRegistry._metadata["MCPAgentBlock"].block_class == MCPAgentBlock
        assert BlockRegistry._metadata["MCPAgentBlock"].category == "mcp"
