# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenAI chat completion blocks."""

# Standard
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

# Third Party
from datasets import Dataset
import openai
import pytest

# First Party
from sdg_hub.blocks.openaichatblock import OpenAIAsyncChatBlock, OpenAIChatBlock


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    client = MagicMock(spec=openai.OpenAI)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def mock_async_openai_client():
    """Create a mock async OpenAI client for testing."""
    client = MagicMock(spec=openai.AsyncOpenAI)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test async response"

    # Make the chat.completions.create method async
    async def mock_create(*args, **kwargs):
        return mock_response

    client.chat.completions.create = AsyncMock(side_effect=mock_create)
    return client


@pytest.fixture
def sample_messages():
    """Sample messages in OpenAI format."""
    return [
        [{"role": "user", "content": "Hello, how are you?"}],
        [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is 2+2?"},
        ],
    ]


@pytest.fixture
def sample_dataset(sample_messages):
    """Create a sample dataset with messages."""
    return Dataset.from_dict({"messages": sample_messages})


class TestOpenAIChatBlock:
    """Tests for OpenAIChatBlock."""

    def test_init_basic(self, mock_openai_client):
        """Test basic initialization."""
        block = OpenAIChatBlock(
            block_name="test_block",
            input_cols="messages",
            output_cols="response",
            client=mock_openai_client,
            model_id="gpt-4",
        )

        assert block.block_name == "test_block"
        assert block.input_cols == ["messages"]
        assert block.output_cols == ["response"]
        assert block.client == mock_openai_client
        assert block.model_id == "gpt-4"
        assert block.messages_column == "messages"
        assert block.output_column == "response"
        assert block.gen_kwargs == {}

    def test_init_with_parameters(self, mock_openai_client):
        """Test initialization with generation parameters."""
        block = OpenAIChatBlock(
            block_name="test_block",
            input_cols="messages",
            output_cols="response",
            client=mock_openai_client,
            model_id="gpt-4",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
        )

        expected_kwargs = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
        }
        assert block.gen_kwargs == expected_kwargs

    def test_init_filters_none_parameters(self, mock_openai_client):
        """Test that None parameters are filtered out."""
        block = OpenAIChatBlock(
            block_name="test_block",
            input_cols="messages",
            output_cols="response",
            client=mock_openai_client,
            model_id="gpt-4",
            temperature=0.7,
            max_tokens=None,
            top_p=0.9,
            seed=None,
        )

        expected_kwargs = {
            "temperature": 0.7,
            "top_p": 0.9,
        }
        assert block.gen_kwargs == expected_kwargs

    def test_init_multiple_input_cols_error(self, mock_openai_client):
        """Test error when multiple input columns provided."""
        with pytest.raises(ValueError, match="expects exactly one input column"):
            OpenAIChatBlock(
                block_name="test_block",
                input_cols=["messages1", "messages2"],
                output_cols="response",
                client=mock_openai_client,
                model_id="gpt-4",
            )

    def test_init_multiple_output_cols_error(self, mock_openai_client):
        """Test error when multiple output columns provided."""
        with pytest.raises(ValueError, match="expects exactly one output column"):
            OpenAIChatBlock(
                block_name="test_block",
                input_cols="messages",
                output_cols=["response1", "response2"],
                client=mock_openai_client,
                model_id="gpt-4",
            )

    def test_generate_basic(self, mock_openai_client, sample_dataset):
        """Test basic generation."""
        block = OpenAIChatBlock(
            block_name="test_block",
            input_cols="messages",
            output_cols="response",
            client=mock_openai_client,
            model_id="gpt-4",
        )

        result = block.generate(sample_dataset)

        assert "response" in result.column_names
        assert len(result["response"]) == 2
        assert all(response == "Test response" for response in result["response"])

        # Verify client was called correctly
        assert mock_openai_client.chat.completions.create.call_count == 2
        calls = mock_openai_client.chat.completions.create.call_args_list
        assert calls[0][1]["model"] == "gpt-4"
        assert calls[0][1]["messages"] == sample_dataset["messages"][0]

    def test_generate_with_override_kwargs(self, mock_openai_client, sample_dataset):
        """Test generation with runtime parameter overrides."""
        block = OpenAIChatBlock(
            block_name="test_block",
            input_cols="messages",
            output_cols="response",
            client=mock_openai_client,
            model_id="gpt-4",
            temperature=0.5,
        )

        block.generate(sample_dataset, temperature=0.9, max_tokens=150)

        # Verify override parameters were used
        calls = mock_openai_client.chat.completions.create.call_args_list
        assert calls[0][1]["temperature"] == 0.9  # Override value
        assert calls[0][1]["max_tokens"] == 150  # New parameter
        assert calls[0][1]["model"] == "gpt-4"

    def test_generate_with_invalid_kwargs(self, mock_openai_client, sample_dataset):
        """Test generation with invalid parameters (should be filtered)."""
        block = OpenAIChatBlock(
            block_name="test_block",
            input_cols="messages",
            output_cols="response",
            client=mock_openai_client,
            model_id="gpt-4",
        )

        with patch("sdg_hub.blocks.openaichatblock.logger") as mock_logger:
            block.generate(sample_dataset, invalid_param="value", temperature=0.7)

            # Check warning was logged
            mock_logger.warning.assert_called_once()
            assert "invalid_param" in str(mock_logger.warning.call_args)

            # Valid parameter should still be passed
            calls = mock_openai_client.chat.completions.create.call_args_list
            assert calls[0][1]["temperature"] == 0.7
            assert "invalid_param" not in calls[0][1]

    def test_generate_with_retry_on_rate_limit(
        self, mock_openai_client, sample_dataset
    ):
        """Test retry behavior on rate limit errors."""
        # Create a proper mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success after retry"

        # Create a mock request for the error
        mock_request = MagicMock()
        mock_response_for_error = MagicMock()
        mock_response_for_error.request = mock_request

        # First call raises rate limit error, second succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            openai.RateLimitError(
                "Rate limited", response=mock_response_for_error, body=None
            ),
            mock_response,
        ]

        block = OpenAIChatBlock(
            block_name="test_block",
            input_cols="messages",
            output_cols="response",
            client=mock_openai_client,
            model_id="gpt-4",
        )

        # Create dataset with single message to avoid multiple calls
        single_message_dataset = Dataset.from_dict(
            {"messages": [sample_dataset["messages"][0]]}
        )
        result = block.generate(single_message_dataset)

        assert result["response"][0] == "Success after retry"
        assert mock_openai_client.chat.completions.create.call_count == 2

    def test_generate_all_openai_parameters(self, mock_openai_client, sample_dataset):
        """Test generation with all supported OpenAI parameters."""
        block = OpenAIChatBlock(
            block_name="test_block",
            input_cols="messages",
            output_cols="response",
            client=mock_openai_client,
            model_id="gpt-4",
            frequency_penalty=0.1,
            logit_bias={"50256": -100},
            logprobs=True,
            max_completion_tokens=200,
            max_tokens=150,
            n=1,
            presence_penalty=0.2,
            response_format={"type": "json_object"},
            seed=42,
            stop=["END"],
            stream=False,
            temperature=0.8,
            tool_choice="auto",
            tools=[{"type": "function", "function": {"name": "test"}}],
            top_logprobs=5,
            top_p=0.95,
            user="test_user",
            extra_body={"custom": "value"},
        )

        # Create single message dataset for simpler testing
        single_message_dataset = Dataset.from_dict(
            {"messages": [sample_dataset["messages"][0]]}
        )
        block.generate(single_message_dataset)

        # Verify all parameters were passed
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["frequency_penalty"] == 0.1
        assert call_kwargs["logit_bias"] == {"50256": -100}
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["max_completion_tokens"] == 200
        assert call_kwargs["max_tokens"] == 150
        assert call_kwargs["n"] == 1
        assert call_kwargs["presence_penalty"] == 0.2
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["seed"] == 42
        assert call_kwargs["stop"] == ["END"]
        assert call_kwargs["stream"] is False
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["tool_choice"] == "auto"
        assert call_kwargs["tools"] == [
            {"type": "function", "function": {"name": "test"}}
        ]
        assert call_kwargs["top_logprobs"] == 5
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["user"] == "test_user"
        assert call_kwargs["extra_body"] == {"custom": "value"}


class TestOpenAIAsyncChatBlock:
    """Tests for OpenAIAsyncChatBlock."""

    def test_init_basic(self, mock_async_openai_client):
        """Test basic initialization."""
        block = OpenAIAsyncChatBlock(
            block_name="test_async_block",
            input_cols="messages",
            output_cols="response",
            async_client=mock_async_openai_client,
            model_id="gpt-4",
        )

        assert block.block_name == "test_async_block"
        assert block.input_cols == ["messages"]
        assert block.output_cols == ["response"]
        assert block.async_client == mock_async_openai_client
        assert block.model_id == "gpt-4"
        assert block.messages_column == "messages"
        assert block.output_column == "response"
        assert block.gen_kwargs == {}

    def test_init_with_parameters(self, mock_async_openai_client):
        """Test initialization with generation parameters."""
        block = OpenAIAsyncChatBlock(
            block_name="test_async_block",
            input_cols="messages",
            output_cols="response",
            async_client=mock_async_openai_client,
            model_id="gpt-4",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
        )

        expected_kwargs = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
        }
        assert block.gen_kwargs == expected_kwargs

    def test_init_multiple_input_cols_error(self, mock_async_openai_client):
        """Test error when multiple input columns provided."""
        with pytest.raises(ValueError, match="expects exactly one input column"):
            OpenAIAsyncChatBlock(
                block_name="test_async_block",
                input_cols=["messages1", "messages2"],
                output_cols="response",
                async_client=mock_async_openai_client,
                model_id="gpt-4",
            )

    def test_init_multiple_output_cols_error(self, mock_async_openai_client):
        """Test error when multiple output columns provided."""
        with pytest.raises(ValueError, match="expects exactly one output column"):
            OpenAIAsyncChatBlock(
                block_name="test_async_block",
                input_cols="messages",
                output_cols=["response1", "response2"],
                async_client=mock_async_openai_client,
                model_id="gpt-4",
            )

    def test_generate_basic(self, mock_async_openai_client, sample_dataset):
        """Test basic async generation."""
        block = OpenAIAsyncChatBlock(
            block_name="test_async_block",
            input_cols="messages",
            output_cols="response",
            async_client=mock_async_openai_client,
            model_id="gpt-4",
        )

        result = block.generate(sample_dataset)

        assert "response" in result.column_names
        assert len(result["response"]) == 2
        assert all(response == "Test async response" for response in result["response"])

        # Verify async client was called correctly
        assert mock_async_openai_client.chat.completions.create.call_count == 2

    def test_generate_with_override_kwargs(
        self, mock_async_openai_client, sample_dataset
    ):
        """Test async generation with runtime parameter overrides."""
        block = OpenAIAsyncChatBlock(
            block_name="test_async_block",
            input_cols="messages",
            output_cols="response",
            async_client=mock_async_openai_client,
            model_id="gpt-4",
            temperature=0.5,
        )

        block.generate(sample_dataset, temperature=0.9, max_tokens=150)

        # Verify override parameters were used
        calls = mock_async_openai_client.chat.completions.create.call_args_list
        assert calls[0][1]["temperature"] == 0.9  # Override value
        assert calls[0][1]["max_tokens"] == 150  # New parameter
        assert calls[0][1]["model"] == "gpt-4"

    def test_generate_with_invalid_kwargs(
        self, mock_async_openai_client, sample_dataset
    ):
        """Test async generation with invalid parameters (should be filtered)."""
        block = OpenAIAsyncChatBlock(
            block_name="test_async_block",
            input_cols="messages",
            output_cols="response",
            async_client=mock_async_openai_client,
            model_id="gpt-4",
        )

        with patch("sdg_hub.blocks.openaichatblock.logger") as mock_logger:
            block.generate(sample_dataset, invalid_param="value", temperature=0.7)

            # Check warning was logged
            mock_logger.warning.assert_called_once()
            assert "invalid_param" in str(mock_logger.warning.call_args)

            # Valid parameter should still be passed
            calls = mock_async_openai_client.chat.completions.create.call_args_list
            assert calls[0][1]["temperature"] == 0.7
            assert "invalid_param" not in calls[0][1]

    @pytest.mark.asyncio
    async def test_generate_single_async(
        self, mock_async_openai_client, sample_messages
    ):
        """Test single async generation method."""
        block = OpenAIAsyncChatBlock(
            block_name="test_async_block",
            input_cols="messages",
            output_cols="response",
            async_client=mock_async_openai_client,
            model_id="gpt-4",
        )

        result = await block._generate_single(sample_messages[0], model="gpt-4")

        assert result == "Test async response"
        mock_async_openai_client.chat.completions.create.assert_called_once_with(
            messages=sample_messages[0], model="gpt-4"
        )

    def test_generate_async_retry_on_rate_limit(
        self, mock_async_openai_client, sample_dataset
    ):
        """Test async retry behavior on rate limit errors."""
        # Create proper mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success after retry"

        # Configure side effect for rate limit followed by success
        call_count = 0

        async def side_effect(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                mock_request = MagicMock()
                mock_response_for_error = MagicMock()
                mock_response_for_error.request = mock_request
                raise openai.RateLimitError(
                    "Rate limited", response=mock_response_for_error, body=None
                )
            return mock_response

        mock_async_openai_client.chat.completions.create.side_effect = side_effect

        block = OpenAIAsyncChatBlock(
            block_name="test_async_block",
            input_cols="messages",
            output_cols="response",
            async_client=mock_async_openai_client,
            model_id="gpt-4",
        )

        # Create dataset with single message to test retry more precisely
        single_message_dataset = Dataset.from_dict(
            {"messages": [sample_dataset["messages"][0]]}
        )
        result = block.generate(single_message_dataset)

        assert result["response"][0] == "Success after retry"
        assert call_count == 2

    def test_generate_concurrent_execution(
        self, mock_async_openai_client, sample_dataset
    ):
        """Test that async calls are executed concurrently."""
        # Track call order
        call_order = []

        async def track_calls(*_args, **_kwargs):
            current_call = len(call_order)
            call_order.append(current_call)
            # Simulate some async work
            await asyncio.sleep(0.01)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = f"Response {current_call + 1}"
            return mock_response

        mock_async_openai_client.chat.completions.create.side_effect = track_calls

        block = OpenAIAsyncChatBlock(
            block_name="test_async_block",
            input_cols="messages",
            output_cols="response",
            async_client=mock_async_openai_client,
            model_id="gpt-4",
        )

        result = block.generate(sample_dataset)

        # Verify concurrent execution (all calls should be made)
        assert len(call_order) == 2
        assert len(result["response"]) == 2


class TestErrorHandling:
    """Test error handling for both sync and async blocks."""

    def test_openai_api_timeout_error(self, mock_openai_client, sample_dataset):
        """Test handling of API timeout errors."""
        mock_request = MagicMock()
        mock_openai_client.chat.completions.create.side_effect = openai.APITimeoutError(
            request=mock_request
        )

        block = OpenAIChatBlock(
            block_name="test_block",
            input_cols="messages",
            output_cols="response",
            client=mock_openai_client,
            model_id="gpt-4",
        )

        # Should retry and eventually fail after max attempts
        with pytest.raises(Exception):
            # Use single message dataset to avoid multiple retries
            single_message_dataset = Dataset.from_dict(
                {"messages": [sample_dataset["messages"][0]]}
            )
            block.generate(single_message_dataset)

    def test_openai_api_connection_error(self, mock_openai_client, sample_dataset):
        """Test handling of API connection errors."""
        mock_request = MagicMock()
        mock_openai_client.chat.completions.create.side_effect = (
            openai.APIConnectionError(request=mock_request)
        )

        block = OpenAIChatBlock(
            block_name="test_block",
            input_cols="messages",
            output_cols="response",
            client=mock_openai_client,
            model_id="gpt-4",
        )

        # Should retry and eventually fail after max attempts
        with pytest.raises(Exception):
            single_message_dataset = Dataset.from_dict(
                {"messages": [sample_dataset["messages"][0]]}
            )
            block.generate(single_message_dataset)

    def test_empty_dataset(self, mock_openai_client):
        """Test handling of empty datasets."""
        empty_dataset = Dataset.from_dict({"messages": []})

        block = OpenAIChatBlock(
            block_name="test_block",
            input_cols="messages",
            output_cols="response",
            client=mock_openai_client,
            model_id="gpt-4",
        )

        result = block.generate(empty_dataset)

        assert "response" in result.column_names
        assert len(result["response"]) == 0
        assert mock_openai_client.chat.completions.create.call_count == 0

    def test_malformed_messages(self, mock_openai_client):
        """Test handling of malformed message data."""
        # Create a dataset with mixed valid/invalid message formats
        malformed_dataset = Dataset.from_dict(
            {
                "messages": [
                    [{"role": "user", "content": "valid message"}],  # Valid format
                    [{"invalid": "message"}],  # Missing required role/content
                ]
            }
        )

        block = OpenAIChatBlock(
            block_name="test_block",
            input_cols="messages",
            output_cols="response",
            client=mock_openai_client,
            model_id="gpt-4",
        )

        # The block should pass the data to OpenAI as-is and let OpenAI handle validation
        # If OpenAI raises an error, it should propagate
        mock_request = MagicMock()
        mock_response_for_error = MagicMock()
        mock_response_for_error.request = mock_request
        mock_openai_client.chat.completions.create.side_effect = openai.BadRequestError(
            "Invalid message format", response=mock_response_for_error, body=None
        )

        with pytest.raises(openai.BadRequestError):
            block.generate(malformed_dataset)


class TestRegistration:
    """Test block registration."""

    def test_openai_chat_block_registered(self):
        """Test that OpenAIChatBlock is properly registered."""
        # First Party
        from sdg_hub.registry import BlockRegistry

        assert "OpenAIChatBlock" in BlockRegistry._registry
        assert BlockRegistry._registry["OpenAIChatBlock"] == OpenAIChatBlock

    def test_openai_async_chat_block_registered(self):
        """Test that OpenAIAsyncChatBlock is properly registered."""
        # First Party
        from sdg_hub.registry import BlockRegistry

        assert "OpenAIAsyncChatBlock" in BlockRegistry._registry
        assert BlockRegistry._registry["OpenAIAsyncChatBlock"] == OpenAIAsyncChatBlock
