# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified LLMChatBlock supporting all providers via LiteLLM."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
from datasets import Dataset
import pytest

# First Party
from sdg_hub.blocks.llm import LLMChatBlock, LLMConfig


@pytest.fixture
def mock_litellm_completion():
    """Mock LiteLLM completion function."""
    with patch("sdg_hub.blocks.llm.client_manager.completion") as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response
        yield mock_completion


@pytest.fixture
def mock_litellm_acompletion():
    """Mock LiteLLM async completion function."""
    with patch("sdg_hub.blocks.llm.client_manager.acompletion") as mock_acompletion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test async response"

        async def mock_async_completion(*_args, **_kwargs):
            return mock_response

        mock_acompletion.side_effect = mock_async_completion
        yield mock_acompletion


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


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_basic_config(self):
        """Test basic configuration creation."""
        config = LLMConfig(model="openai/gpt-4")

        assert config.model == "openai/gpt-4"
        assert config.get_provider() == "openai"
        assert config.get_model_name() == "gpt-4"
        assert not config.is_local_model()
        assert config.timeout == 120.0
        assert config.max_retries == 6

    def test_config_with_parameters(self):
        """Test configuration with generation parameters."""
        config = LLMConfig(
            model="anthropic/claude-3-sonnet-20240229",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
        )

        assert config.temperature == 0.7
        assert config.max_tokens == 100
        assert config.top_p == 0.9
        assert config.get_provider() == "anthropic"

    def test_local_model_detection(self):
        """Test local model detection."""
        local_config = LLMConfig(
            model="hosted_vllm/meta-llama/Llama-2-7b-chat-hf",
            api_base="http://localhost:8000/v1",
        )

        assert local_config.is_local_model()
        assert local_config.get_provider() == "hosted_vllm"

    def test_invalid_model_format(self):
        """Test error on invalid model format."""
        with pytest.raises(
            ValueError, match="should be in format 'provider/model-name'"
        ):
            LLMConfig(model="invalid-model-format")

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        config = LLMConfig(
            model="openai/gpt-4", temperature=1.0, max_tokens=100, top_p=0.5
        )
        assert config.temperature == 1.0

        # Invalid temperature
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
            LLMConfig(model="openai/gpt-4", temperature=3.0)

        # Invalid max_tokens
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            LLMConfig(model="openai/gpt-4", max_tokens=-1)

    def test_api_key_resolution(self):
        """Test API key resolution from environment variables."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            config = LLMConfig(model="openai/gpt-4")
            assert config.api_key == "test-key"

    def test_generation_kwargs(self):
        """Test generation kwargs extraction."""
        config = LLMConfig(
            model="openai/gpt-4",
            temperature=0.7,
            max_tokens=100,
            seed=42,
            extra_headers={"x-custom": "value"},
        )

        kwargs = config.get_generation_kwargs()
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100
        assert kwargs["seed"] == 42
        assert kwargs["extra_headers"] == {"x-custom": "value"}

    def test_merge_overrides(self):
        """Test configuration merging with overrides."""
        base_config = LLMConfig(model="openai/gpt-4", temperature=0.5, max_tokens=100)

        new_config = base_config.merge_overrides(temperature=0.9, seed=42)

        assert new_config.temperature == 0.9  # Overridden
        assert new_config.max_tokens == 100  # Preserved
        assert new_config.seed == 42  # Added
        assert base_config.temperature == 0.5  # Original unchanged


class TestLLMChatBlock:
    """Tests for LLMChatBlock."""

    def test_init_openai_model(self, mock_litellm_completion):
        """Test initialization with OpenAI model."""
        block = LLMChatBlock(
            block_name="test_openai",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
            temperature=0.7,
        )

        assert block.block_name == "test_openai"
        assert block.messages_column == "messages"
        assert block.output_column == "response"
        assert block.config.model == "openai/gpt-4"
        assert block.config.temperature == 0.7
        assert not block.async_mode

    def test_init_anthropic_model(self, mock_litellm_completion):
        """Test initialization with Anthropic model."""
        block = LLMChatBlock(
            block_name="test_anthropic",
            input_cols="messages",
            output_cols="response",
            model="anthropic/claude-3-sonnet-20240229",
            temperature=0.5,
        )

        assert block.config.model == "anthropic/claude-3-sonnet-20240229"
        assert block.config.get_provider() == "anthropic"

    def test_init_local_model(self, mock_litellm_completion):
        """Test initialization with local vLLM model."""
        block = LLMChatBlock(
            block_name="test_local",
            input_cols="messages",
            output_cols="response",
            model="hosted_vllm/meta-llama/Llama-2-7b-chat-hf",
            api_base="http://localhost:8000/v1",
        )

        assert block.config.is_local_model()
        assert block.config.api_base == "http://localhost:8000/v1"

    def test_init_async_mode(self, mock_litellm_acompletion):
        """Test initialization with async mode."""
        block = LLMChatBlock(
            block_name="test_async",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            async_mode=True,
        )

        assert block.async_mode is True

    def test_init_multiple_input_cols_error(self):
        """Test error when multiple input columns provided."""
        with pytest.raises(ValueError, match="expects exactly one input column"):
            LLMChatBlock(
                block_name="test_block",
                input_cols=["messages1", "messages2"],
                output_cols="response",
                model="openai/gpt-4",
            )

    def test_init_multiple_output_cols_error(self):
        """Test error when multiple output columns provided."""
        with pytest.raises(ValueError, match="expects exactly one output column"):
            LLMChatBlock(
                block_name="test_block",
                input_cols="messages",
                output_cols=["response1", "response2"],
                model="openai/gpt-4",
            )

    def test_sync_generation(self, mock_litellm_completion, sample_dataset):
        """Test synchronous generation."""
        block = LLMChatBlock(
            block_name="test_sync",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
        )

        result = block.generate(sample_dataset)

        assert "response" in result.column_names
        assert len(result["response"]) == 2
        assert all(response == "Test response" for response in result["response"])
        assert mock_litellm_completion.call_count == 2

    def test_async_generation(self, mock_litellm_acompletion, sample_dataset):
        """Test asynchronous generation."""
        block = LLMChatBlock(
            block_name="test_async",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
            async_mode=True,
        )

        result = block.generate(sample_dataset)

        assert "response" in result.column_names
        assert len(result["response"]) == 2
        assert all(response == "Test async response" for response in result["response"])
        assert (
            mock_litellm_acompletion.call_count == 2
        )  # Concurrent calls, one per message

    def test_generation_with_overrides(self, mock_litellm_completion, sample_dataset):
        """Test generation with runtime parameter overrides."""
        block = LLMChatBlock(
            block_name="test_overrides",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
            temperature=0.5,
        )

        block.generate(sample_dataset, temperature=0.9, max_tokens=150)

        # Verify override parameters were used
        calls = mock_litellm_completion.call_args_list
        assert calls[0][1]["temperature"] == 0.9  # Override value
        assert calls[0][1]["max_tokens"] == 150  # New parameter
        assert calls[0][1]["model"] == "openai/gpt-4"

    def test_generation_all_providers(self, mock_litellm_completion, sample_dataset):
        """Test generation works with different providers."""
        providers_models = [
            "openai/gpt-4",
            "anthropic/claude-3-sonnet-20240229",
            "google/gemini-pro",
            "hosted_vllm/meta-llama/Llama-2-7b-chat-hf",
        ]

        for model in providers_models:
            block = LLMChatBlock(
                block_name=f"test_{model.replace('/', '_')}",
                input_cols="messages",
                output_cols="response",
                model=model,
                api_key="test-key",
            )

            # Create single message dataset for simpler testing
            single_dataset = Dataset.from_dict(
                {"messages": [sample_dataset["messages"][0]]}
            )
            result = block.generate(single_dataset)

            assert "response" in result.column_names
            assert len(result["response"]) == 1
            assert result["response"][0] == "Test response"

    def test_generation_with_all_parameters(
        self, mock_litellm_completion, sample_dataset
    ):
        """Test generation with all supported parameters."""
        block = LLMChatBlock(
            block_name="test_all_params",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
            frequency_penalty=0.1,
            max_tokens=150,
            n=1,
            presence_penalty=0.2,
            response_format={"type": "json_object"},
            seed=42,
            stop=["END"],
            stream=False,
            temperature=0.8,
            top_p=0.95,
            user="test_user",
        )

        # Create single message dataset for simpler testing
        single_dataset = Dataset.from_dict(
            {"messages": [sample_dataset["messages"][0]]}
        )
        block.generate(single_dataset)

        # Verify all parameters were passed
        call_kwargs = mock_litellm_completion.call_args[1]
        assert call_kwargs["frequency_penalty"] == 0.1
        assert call_kwargs["max_tokens"] == 150
        assert call_kwargs["n"] == 1
        assert call_kwargs["presence_penalty"] == 0.2
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["seed"] == 42
        assert call_kwargs["stop"] == ["END"]
        assert call_kwargs["stream"] is False
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["user"] == "test_user"

    def test_input_validation(self, mock_litellm_completion):
        """Test input validation functionality."""
        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
        )

        # Valid sample
        valid_sample = {"messages": [{"role": "user", "content": "Hello!"}]}
        assert block.validate_input(valid_sample) is True

        # Invalid sample - missing messages
        invalid_sample = {"other_field": "value"}
        assert block.validate_input(invalid_sample) is False

        # Invalid sample - wrong message format
        invalid_format = {
            "messages": [
                {"role": "user"}  # Missing content
            ]
        }
        assert block.validate_input(invalid_format) is False

        # Invalid sample - messages not a list
        invalid_type = {"messages": "not a list"}
        assert block.validate_input(invalid_type) is False

        # Invalid sample - empty messages
        empty_messages = {"messages": []}
        assert block.validate_input(empty_messages) is False

    def test_model_info(self, mock_litellm_completion):
        """Test model info functionality."""
        block = LLMChatBlock(
            block_name="test_info",
            input_cols="messages",
            output_cols="response",
            model="anthropic/claude-3-sonnet-20240229",
            temperature=0.5,
        )

        info = block.get_model_info()
        assert info["model"] == "anthropic/claude-3-sonnet-20240229"
        assert info["provider"] == "anthropic"
        assert info["block_name"] == "test_info"
        assert info["input_column"] == "messages"
        assert info["output_column"] == "response"
        assert not info["is_local"]

    def test_empty_dataset(self, mock_litellm_completion):
        """Test handling of empty datasets."""
        empty_dataset = Dataset.from_dict({"messages": []})

        block = LLMChatBlock(
            block_name="test_empty",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
        )

        result = block.generate(empty_dataset)

        assert "response" in result.column_names
        assert len(result["response"]) == 0
        assert mock_litellm_completion.call_count == 0


class TestErrorHandling:
    """Test error handling for LLMChatBlock."""

    def test_litellm_rate_limit_error(self, sample_dataset):
        """Test handling of LiteLLM rate limit errors."""
        with patch("sdg_hub.blocks.llm.client_manager.completion") as mock_completion:
            # First Party
            from sdg_hub.blocks.llm.error_handler import RateLimitError

            # Mock successful response for retry
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Success after retry"

            # First call raises rate limit error, second succeeds
            mock_completion.side_effect = [
                RateLimitError("Rate limited", llm_provider="openai", model="gpt-4"),
                mock_response,
            ]

            block = LLMChatBlock(
                block_name="test_retry",
                input_cols="messages",
                output_cols="response",
                model="openai/gpt-4",
                api_key="test-key",
            )

            # Create single message dataset to test retry behavior
            single_dataset = Dataset.from_dict(
                {"messages": [sample_dataset["messages"][0]]}
            )
            result = block.generate(single_dataset)

            assert result["response"][0] == "Success after retry"
            assert mock_completion.call_count == 2

    def test_litellm_authentication_error(self, sample_dataset):
        """Test handling of authentication errors (non-retryable)."""
        with patch("sdg_hub.blocks.llm.client_manager.completion") as mock_completion:
            # First Party
            from sdg_hub.blocks.llm.error_handler import AuthenticationError

            mock_completion.side_effect = AuthenticationError(
                "Invalid API key", llm_provider="openai", model="gpt-4"
            )

            block = LLMChatBlock(
                block_name="test_auth_error",
                input_cols="messages",
                output_cols="response",
                model="openai/gpt-4",
                api_key="invalid-key",
            )

            # Should fail immediately without retries
            with pytest.raises(AuthenticationError):
                single_dataset = Dataset.from_dict(
                    {"messages": [sample_dataset["messages"][0]]}
                )
                block.generate(single_dataset)

    def test_litellm_context_window_error(self, sample_dataset):
        """Test handling of context window exceeded errors."""
        with patch("sdg_hub.blocks.llm.client_manager.completion") as mock_completion:
            # First Party
            from sdg_hub.blocks.llm.error_handler import ContextWindowExceededError

            mock_completion.side_effect = ContextWindowExceededError(
                "Context window exceeded", llm_provider="openai", model="gpt-4"
            )

            block = LLMChatBlock(
                block_name="test_context_error",
                input_cols="messages",
                output_cols="response",
                model="openai/gpt-4",
                api_key="test-key",
            )

            # Should fail immediately without retries
            with pytest.raises(ContextWindowExceededError):
                single_dataset = Dataset.from_dict(
                    {"messages": [sample_dataset["messages"][0]]}
                )
                block.generate(single_dataset)


class TestRegistration:
    """Test block registration."""

    def test_llm_chat_block_registered(self):
        """Test that LLMChatBlock is properly registered."""
        # First Party
        from sdg_hub.registry import BlockRegistry

        assert "LLMChatBlock" in BlockRegistry._registry
        assert BlockRegistry._registry["LLMChatBlock"] == LLMChatBlock
