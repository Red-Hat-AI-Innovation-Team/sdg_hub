# SPDX-License-Identifier: Apache-2.0
"""Tests for LLMChatWithParsingRetryBlock composite block."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.llm import LLMChatWithParsingRetryBlock
from sdg_hub.core.blocks.llm.llm_chat_with_parsing_retry_block import (
    MaxRetriesExceededError,
)
from sdg_hub.core.utils.error_handling import BlockValidationError
import pytest


class MockMessage:
    """Mock message class that behaves like LiteLLM message for dict() conversion."""

    def __init__(self, content):
        self.content = content

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


def create_mock_response(contents):
    """Helper to create a mock response with multiple choices."""
    mock_response = MagicMock()
    choices = []
    if isinstance(contents, str):
        contents = [contents]

    for content in contents:
        choice = MagicMock()
        choice.message = MockMessage(content)
        choices.append(choice)

    mock_response.choices = choices
    return mock_response


@pytest.fixture
def mock_litellm_completion():
    """Mock LiteLLM completion function for successful responses."""
    with patch("sdg_hub.core.blocks.llm.llm_chat_block.completion") as mock_completion:
        mock_response = MagicMock()
        choice = MagicMock()
        choice.message = MockMessage("<answer>Test response</answer>")
        mock_response.choices = [choice]
        mock_completion.return_value = mock_response
        yield mock_completion


@pytest.fixture
def mock_litellm_completion_multiple():
    """Mock LiteLLM completion function for multiple responses (n > 1)."""
    with patch("sdg_hub.core.blocks.llm.llm_chat_block.completion") as mock_completion:
        mock_response = MagicMock()
        choices = []
        for i in range(3):
            choice = MagicMock()
            choice.message = MockMessage(f"<answer>Response {i + 1}</answer>")
            choices.append(choice)
        mock_response.choices = choices
        mock_completion.return_value = mock_response
        yield mock_completion


@pytest.fixture
def mock_litellm_completion_partial():
    """Mock LiteLLM completion that returns some parseable and some unparseable responses."""
    with patch("sdg_hub.core.blocks.llm.llm_chat_block.completion") as mock_completion:
        mock_response = MagicMock()
        choice1 = MagicMock()
        choice1.message = MockMessage("<answer>Good response</answer>")
        choice2 = MagicMock()
        choice2.message = MockMessage("Unparseable response")
        mock_response.choices = [choice1, choice2]
        mock_completion.return_value = mock_response
        yield mock_completion


@pytest.fixture
def mock_litellm_completion_unparseable():
    """Mock LiteLLM completion that always returns unparseable responses."""
    with patch("sdg_hub.core.blocks.llm.llm_chat_block.completion") as mock_completion:
        mock_response = MagicMock()
        choice = MagicMock()
        choice.message = MockMessage("No tags in this response")
        mock_response.choices = [choice]
        mock_completion.return_value = mock_response
        yield mock_completion


@pytest.fixture
def sample_messages():
    """Sample messages in OpenAI format."""
    return [
        [{"role": "user", "content": "Please provide an answer in <answer> tags."}],
        [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is 2+2? Use <answer> tags."},
        ],
    ]


@pytest.fixture
def sample_dataset(sample_messages):
    """Create a sample dataset with messages."""
    return Dataset.from_dict({"messages": sample_messages})


class TestLLMChatWithParsingRetryBlockInitialization:
    """Test block initialization and configuration validation."""

    def test_basic_initialization_tag_parsing(self, mock_litellm_completion):
        """Test basic initialization with tag-based parsing."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_retry_block",
            input_cols="messages",
            output_cols="parsed_answer",
            model="openai/gpt-4",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
            parsing_max_retries=3,
        )

        assert block.block_name == "test_retry_block"
        assert block.input_cols == ["messages"]
        assert block.output_cols == ["parsed_answer"]
        assert block.model == "openai/gpt-4"
        assert block.parsing_max_retries == 3
        assert block.start_tags == ["<answer>"]
        assert block.end_tags == ["</answer>"]

        # Check internal blocks are created
        assert block.llm_chat is not None
        assert block.text_parser is not None
        assert block.llm_chat.block_name == "test_retry_block_llm_chat"
        assert block.text_parser.block_name == "test_retry_block_text_parser"

    def test_initialization_regex_parsing(self, mock_litellm_completion):
        """Test initialization with regex-based parsing."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_regex_retry",
            input_cols="messages",
            output_cols="result",
            model="anthropic/claude-3-sonnet-20240229",
            parsing_pattern=r'"result":\s*"([^"]*)"',
            parsing_max_retries=5,
        )

        assert block.parsing_pattern == r'"result":\s*"([^"]*)"'
        assert block.parsing_max_retries == 5
        assert block.text_parser.parsing_pattern == r'"result":\s*"([^"]*)"'

    def test_initialization_multiple_output_columns(self, mock_litellm_completion):
        """Test initialization with multiple output columns."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_multi_output",
            input_cols="messages",
            output_cols=["explanation", "answer"],
            model="openai/gpt-4",
            start_tags=["<explanation>", "<answer>"],
            end_tags=["</explanation>", "</answer>"],
        )

        assert len(block.output_cols) == 2
        assert block.output_cols == ["explanation", "answer"]
        assert len(block.start_tags) == 2
        assert len(block.end_tags) == 2

    def test_initialization_all_llm_parameters(self, mock_litellm_completion):
        """Test initialization with all LLM generation parameters."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_all_params",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            temperature=0.8,
            max_tokens=150,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop=["END"],
            seed=42,
            n=2,
            start_tags=["<response>"],
            end_tags=["</response>"],
            parsing_max_retries=4,
        )

        # Check that parameters are passed to internal LLM block
        assert block.llm_chat.temperature == 0.8
        assert block.llm_chat.max_tokens == 150
        assert block.llm_chat.top_p == 0.9
        assert block.llm_chat.n == 2
        assert block.llm_chat.seed == 42

    def test_input_column_validation(self):
        """Test validation of input columns."""
        # Multiple input columns should raise error
        with pytest.raises(ValueError, match="exactly one input column"):
            LLMChatWithParsingRetryBlock(
                block_name="test_invalid",
                input_cols=["messages1", "messages2"],
                output_cols="response",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
            )

    def test_parsing_max_retries_validation(self):
        """Test validation of parsing_max_retries parameter."""
        # Negative retries should raise error
        with pytest.raises(ValueError, match="parsing_max_retries must be at least 1"):
            LLMChatWithParsingRetryBlock(
                block_name="test_invalid",
                input_cols="messages",
                output_cols="response",
                model="openai/gpt-4",
                parsing_max_retries=0,
                start_tags=["<answer>"],
                end_tags=["</answer>"],
            )

    def test_parsing_configuration_validation(self):
        """Test validation of parsing configuration."""
        # No parsing method should raise error
        with pytest.raises(ValueError, match="at least one parsing method"):
            LLMChatWithParsingRetryBlock(
                block_name="test_invalid",
                input_cols="messages",
                output_cols="response",
                model="openai/gpt-4",
                # No parsing_pattern, start_tags, or end_tags
            )

    def test_model_configuration_requirement(self, mock_litellm_completion):
        """Test that model configuration is required for generation."""
        # Create block without model
        block = LLMChatWithParsingRetryBlock(
            block_name="test_no_model",
            input_cols="messages",
            output_cols="response",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
            # No model specified
        )

        dataset = Dataset.from_dict(
            {"messages": [[{"role": "user", "content": "test"}]]}
        )

        # Should raise BlockValidationError when trying to generate
        with pytest.raises(BlockValidationError, match="Model not configured"):
            block.generate(dataset)


class TestLLMChatWithParsingRetryBlockSuccessfulGeneration:
    """Test successful parsing scenarios with retry logic."""

    def test_successful_generation_first_attempt(
        self, mock_litellm_completion, sample_dataset
    ):
        """Test successful generation on first attempt."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_success",
            input_cols="messages",
            output_cols="answer",
            model="openai/gpt-4",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
            parsing_max_retries=3,
        )

        result = block.generate(sample_dataset)

        # Should succeed on first attempt
        assert len(result) == 2  # Two input samples
        assert all("answer" in row for row in result)
        assert all(row["answer"] == "Test response" for row in result)

        # LLM should be called once per sample (no retries needed)
        assert mock_litellm_completion.call_count == 2

    def test_successful_generation_with_n_parameter(
        self, mock_litellm_completion_multiple, sample_dataset
    ):
        """Test successful generation with n > 1."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_multiple",
            input_cols="messages",
            output_cols="answer",
            model="openai/gpt-4",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
            n=3,  # Generate 3 responses per sample
            parsing_max_retries=2,
        )

        result = block.generate(sample_dataset)

        # Should generate 3 responses per input sample = 6 total
        assert len(result) == 6
        expected_responses = ["Response 1", "Response 2", "Response 3"] * 2
        actual_responses = [row["answer"] for row in result]
        assert actual_responses == expected_responses

    def test_successful_generation_multiple_output_columns(
        self, mock_litellm_completion, sample_dataset
    ):
        """Test successful generation with multiple output columns."""
        # Mock response with multiple tags
        content = "<explanation>This is an explanation</explanation><answer>42</answer>"
        mock_litellm_completion.return_value = create_mock_response(content)

        block = LLMChatWithParsingRetryBlock(
            block_name="test_multi_cols",
            input_cols="messages",
            output_cols=["explanation", "answer"],
            model="openai/gpt-4",
            start_tags=["<explanation>", "<answer>"],
            end_tags=["</explanation>", "</answer>"],
            parsing_max_retries=3,
        )

        result = block.generate(sample_dataset)

        assert len(result) == 2
        for row in result:
            assert row["explanation"] == "This is an explanation"
            assert row["answer"] == "42"

    def test_successful_generation_after_retry(self, sample_dataset):
        """Test successful generation after initial parsing failures."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            # First call returns unparseable, second returns parseable
            mock_response_bad = create_mock_response("No tags here")
            mock_response_good = create_mock_response("<answer>Good response</answer>")

            # Alternate between bad and good responses
            mock_completion.side_effect = [
                mock_response_bad,
                mock_response_good,  # For first sample
                mock_response_bad,
                mock_response_good,  # For second sample
            ]

            block = LLMChatWithParsingRetryBlock(
                block_name="test_retry_success",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                parsing_max_retries=3,
            )

            result = block.generate(sample_dataset)

            # Should succeed after retry
            assert len(result) == 2
            assert all(row["answer"] == "Good response" for row in result)

            # Should have called LLM twice per sample (1 retry each)
            assert mock_completion.call_count == 4

    def test_partial_success_accumulation(self, sample_dataset):
        """Test accumulation of partial successes across retries."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            # First call returns 1 parseable out of 2, second call returns 1 more
            mock_response_1 = create_mock_response(
                ["<answer>First good</answer>", "Unparseable"]
            )
            mock_response_2 = create_mock_response(
                ["<answer>Second good</answer>", "Also unparseable"]
            )

            mock_completion.side_effect = [mock_response_1, mock_response_2] * 2

            block = LLMChatWithParsingRetryBlock(
                block_name="test_accumulate",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                n=2,  # Want 2 responses per sample
                parsing_max_retries=3,
            )

            result = block.generate(sample_dataset)

            # Should accumulate 2 responses per sample = 4 total
            assert len(result) == 4
            expected_answers = ["First good", "Second good"] * 2
            actual_answers = [row["answer"] for row in result]
            assert actual_answers == expected_answers

    def test_excess_results_trimming(self, sample_dataset):
        """Test trimming results when exceeding target count."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            # Return 3 parseable responses when only 2 are needed
            mock_completion.return_value = create_mock_response(
                [
                    "<answer>Response 1</answer>",
                    "<answer>Response 2</answer>",
                    "<answer>Response 3</answer>",
                ]
            )

            block = LLMChatWithParsingRetryBlock(
                block_name="test_trim",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                n=2,  # Only want 2 responses
                parsing_max_retries=3,
            )

            result = block.generate(sample_dataset)

            # Should trim to exactly 2 responses per sample = 4 total
            assert len(result) == 4
            expected_answers = ["Response 1", "Response 2"] * 2
            actual_answers = [row["answer"] for row in result]
            assert actual_answers == expected_answers


class TestLLMChatWithParsingRetryBlockMaxRetriesExceeded:
    """Test MaxRetriesExceededError scenarios."""

    def test_max_retries_exceeded_no_successful_parses(
        self, mock_litellm_completion_unparseable, sample_dataset
    ):
        """Test MaxRetriesExceededError when no responses can be parsed."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_max_retries",
            input_cols="messages",
            output_cols="answer",
            model="openai/gpt-4",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
            parsing_max_retries=2,
        )

        # Create single sample dataset to test specific error message
        single_dataset = Dataset.from_dict(
            {"messages": [sample_dataset["messages"][0]]}
        )

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            block.generate(single_dataset)

        error = exc_info.value
        assert error.target_count == 1
        assert error.actual_count == 0
        assert error.max_retries == 2
        assert "Failed to achieve target count 1 after 2 retries" in str(error)

    def test_max_retries_exceeded_partial_success(self, sample_dataset):
        """Test MaxRetriesExceededError when some but not all responses are parsed."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            # Always return 1 parseable out of 3 needed
            mock_completion.return_value = create_mock_response(
                ["<answer>Only one good</answer>", "Unparseable", "Also unparseable"]
            )

            block = LLMChatWithParsingRetryBlock(
                block_name="test_partial_failure",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                n=3,  # Need 3 responses
                parsing_max_retries=2,
            )

            # Test with single sample for clearer error checking
            single_dataset = Dataset.from_dict(
                {"messages": [sample_dataset["messages"][0]]}
            )

            with pytest.raises(MaxRetriesExceededError) as exc_info:
                block.generate(single_dataset)

            error = exc_info.value
            assert error.target_count == 3
            assert error.actual_count == 2  # Got 1 per retry attempt × 2 attempts
            assert error.max_retries == 2

    def test_max_retries_exceeded_error_details(
        self, mock_litellm_completion_unparseable, sample_dataset
    ):
        """Test detailed error information in MaxRetriesExceededError."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_error_details",
            input_cols="messages",
            output_cols="answer",
            model="openai/gpt-4",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
            n=5,  # High target to test error details
            parsing_max_retries=3,
        )

        single_dataset = Dataset.from_dict(
            {"messages": [sample_dataset["messages"][0]]}
        )

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            block.generate(single_dataset)

        error = exc_info.value
        assert hasattr(error, "target_count")
        assert hasattr(error, "actual_count")
        assert hasattr(error, "max_retries")
        assert error.target_count == 5
        assert error.actual_count == 0
        assert error.max_retries == 3

    def test_different_target_counts_per_sample(self, sample_dataset):
        """Test retry logic with runtime n parameter override."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            # Return 1 parseable response per call
            mock_completion.return_value = create_mock_response(
                "<answer>Single response</answer>"
            )

            block = LLMChatWithParsingRetryBlock(
                block_name="test_override_n",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                n=1,  # Default to 1
                parsing_max_retries=2,
            )

            # Override n to 2 at runtime
            result = block.generate(sample_dataset, n=2)

            # Should successfully get 2 responses per sample = 4 total
            assert len(result) == 4
            # Should have called LLM twice per sample to get 2 responses each
            assert mock_completion.call_count == 4


class TestLLMChatWithParsingRetryBlockEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self, mock_litellm_completion):
        """Test handling of empty datasets."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_empty",
            input_cols="messages",
            output_cols="answer",
            model="openai/gpt-4",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
        )

        empty_dataset = Dataset.from_dict({"messages": []})
        result = block.generate(empty_dataset)

        assert len(result) == 0
        assert mock_litellm_completion.call_count == 0

    def test_llm_generation_error_handling(self, sample_dataset):
        """Test handling of LLM generation errors."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            # First call raises exception, continue to next attempt
            mock_completion.side_effect = [
                Exception("Network error"),
                Exception("Another error"),
                Exception("Final error"),
            ]

            block = LLMChatWithParsingRetryBlock(
                block_name="test_llm_error",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                parsing_max_retries=3,
            )

            single_dataset = Dataset.from_dict(
                {"messages": [sample_dataset["messages"][0]]}
            )

            # Should eventually raise MaxRetriesExceededError after exhausting attempts
            with pytest.raises(MaxRetriesExceededError):
                block.generate(single_dataset)

    def test_mixed_success_failure_across_attempts(self, sample_dataset):
        """Test mixed success/failure scenarios across retry attempts."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            # Simulate pattern: error, success, error, success
            mock_response_good = create_mock_response("<answer>Success</answer>")

            mock_completion.side_effect = [
                Exception("First error"),
                mock_response_good,
                Exception("Second error"),
                mock_response_good,
            ]

            block = LLMChatWithParsingRetryBlock(
                block_name="test_mixed",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                parsing_max_retries=4,
            )

            result = block.generate(sample_dataset)

            # Should get 1 successful response per sample = 2 total
            assert len(result) == 2
            assert all(row["answer"] == "Success" for row in result)

    def test_internal_block_validation(self, mock_litellm_completion):
        """Test validation of internal blocks."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="answer",
            model="openai/gpt-4",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
        )

        # Valid dataset should pass validation
        valid_dataset = Dataset.from_dict(
            {"messages": [[{"role": "user", "content": "test"}]]}
        )

        # Should not raise exception
        block._validate_custom(valid_dataset)

        # Invalid dataset should fail validation
        invalid_dataset = Dataset.from_dict({"wrong_column": ["test"]})

        with pytest.raises(ValueError, match="Required input column"):
            block._validate_custom(invalid_dataset)

    def test_get_internal_blocks_info(self, mock_litellm_completion):
        """Test getting information about internal blocks."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_info",
            input_cols="messages",
            output_cols="answer",
            model="openai/gpt-4",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
        )

        info = block.get_internal_blocks_info()

        assert "llm_chat" in info
        assert "text_parser" in info
        assert info["llm_chat"] is not None
        assert info["text_parser"] is not None

    def test_repr_string(self, mock_litellm_completion):
        """Test string representation of the block."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_repr",
            input_cols="messages",
            output_cols="answer",
            model="openai/gpt-4",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
            parsing_max_retries=5,
        )

        repr_str = repr(block)
        assert "LLMChatWithParsingRetryBlock" in repr_str
        assert "test_repr" in repr_str
        assert "openai/gpt-4" in repr_str
        assert "parsing_max_retries=5" in repr_str

    def test_parameter_forwarding_to_internal_blocks(self, mock_litellm_completion):
        """Test parameter forwarding to internal blocks via __setattr__."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_forwarding",
            input_cols="messages",
            output_cols="answer",
            model="openai/gpt-4",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
        )

        # Change model configuration (simulates Flow.set_model_config)
        block.model = "anthropic/claude-3-sonnet-20240229"
        block.api_key = "new-api-key"
        block.temperature = 0.7

        # Verify internal LLM chat block was updated via __setattr__
        assert block.llm_chat.model == "anthropic/claude-3-sonnet-20240229"
        assert block.llm_chat.api_key == "new-api-key"
        assert block.llm_chat.temperature == 0.7

    def test_regex_parsing_configuration(self, mock_litellm_completion, sample_dataset):
        """Test regex-based parsing configuration and execution."""
        # Mock JSON-like response
        content = 'Here is the result: "answer": "42" and more text'
        mock_litellm_completion.return_value = create_mock_response(content)

        block = LLMChatWithParsingRetryBlock(
            block_name="test_regex",
            input_cols="messages",
            output_cols="result",
            model="openai/gpt-4",
            parsing_pattern=r'"answer":\s*"([^"]*)"',
            parsing_max_retries=2,
        )

        result = block.generate(sample_dataset)

        assert len(result) == 2
        assert all(row["result"] == "42" for row in result)

    def test_async_mode_configuration(self, mock_litellm_completion):
        """Test async mode configuration passed to internal blocks."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_async",
            input_cols="messages",
            output_cols="answer",
            model="openai/gpt-4",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
            async_mode=True,
        )

        # Verify async mode is passed to internal LLM block
        assert block.llm_chat.async_mode is True
        assert block.async_mode is True


class TestLLMChatWithParsingRetryBlockRegistration:
    """Test block registration."""

    def test_block_registered(self):
        """Test that LLMChatWithParsingRetryBlock is properly registered."""
        from sdg_hub import BlockRegistry

        assert "LLMChatWithParsingRetryBlock" in BlockRegistry._metadata
        assert (
            BlockRegistry._metadata["LLMChatWithParsingRetryBlock"].block_class
            == LLMChatWithParsingRetryBlock
        )


class TestLLMChatWithParsingRetryBlockExpandListsFalse:
    """Test expand_lists=False behavior for retry counting."""

    def test_expand_lists_false_successful_first_attempt(
        self, mock_litellm_completion_multiple, sample_dataset
    ):
        """Test expand_lists=False with successful parsing on first attempt."""
        block = LLMChatWithParsingRetryBlock(
            block_name="test_expand_false",
            input_cols="messages",
            output_cols="answer",
            model="openai/gpt-4",
            start_tags=["<answer>"],
            end_tags=["</answer>"],
            expand_lists=False,  # Key: disable list expansion
            n=3,  # Request 3 responses
            parsing_max_retries=2,
        )

        result = block.generate(sample_dataset)

        # Should return 2 rows (one per input sample) with lists as values
        assert len(result) == 2
        for row in result:
            assert "answer" in row
            assert isinstance(row["answer"], list)
            assert len(row["answer"]) == 3  # All 3 responses parsed successfully
            assert row["answer"] == ["Response 1", "Response 2", "Response 3"]

        # Should only call LLM once per sample (no retries needed)
        assert mock_litellm_completion_multiple.call_count == 2

    def test_expand_lists_false_partial_success_with_retry(self, sample_dataset):
        """Test expand_lists=False with partial success requiring retries."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            # First call: 2 responses, only 1 parseable
            mock_response_1 = create_mock_response(
                ["<answer>First good</answer>", "Unparseable response"]
            )

            # Second call: 2 responses, only 1 parseable (should reach target of 2)
            mock_response_2 = create_mock_response(
                ["<answer>Second good</answer>", "Also unparseable"]
            )

            mock_completion.side_effect = [mock_response_1, mock_response_2] * 2

            block = LLMChatWithParsingRetryBlock(
                block_name="test_expand_false_retry",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                expand_lists=False,
                n=2,  # Need 2 successful parses
                parsing_max_retries=3,
            )

            result = block.generate(sample_dataset)

            # Should return 2 rows with accumulated successful parses as lists
            assert len(result) == 2
            for row in result:
                assert isinstance(row["answer"], list)
                assert len(row["answer"]) == 2
                assert row["answer"] == ["First good", "Second good"]

            # Should call LLM twice per sample (1 retry each)
            assert mock_completion.call_count == 4

    def test_expand_lists_false_max_retries_exceeded(self, sample_dataset):
        """Test expand_lists=False with MaxRetriesExceededError."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            # Always return 1 parseable out of 3 needed
            mock_completion.return_value = create_mock_response(
                ["<answer>Only one good</answer>", "Unparseable"]
            )

            block = LLMChatWithParsingRetryBlock(
                block_name="test_expand_false_failure",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                expand_lists=False,
                n=3,  # Need 3 successful parses
                parsing_max_retries=2,
            )

            single_dataset = Dataset.from_dict(
                {"messages": [sample_dataset["messages"][0]]}
            )

            with pytest.raises(MaxRetriesExceededError) as exc_info:
                block.generate(single_dataset)

            error = exc_info.value
            assert error.target_count == 3
            assert error.actual_count == 2  # Got 1 per retry × 2 retries
            assert error.max_retries == 2

    def test_expand_lists_false_vs_true_comparison(self, sample_dataset):
        """Test that expand_lists=False vs True produce different output structures."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            mock_completion.return_value = create_mock_response(
                ["<answer>Response 1</answer>", "<answer>Response 2</answer>"]
            )

            # Test expand_lists=False
            block_false = LLMChatWithParsingRetryBlock(
                block_name="test_false",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                expand_lists=False,
                n=2,
            )

            # Test expand_lists=True (default)
            block_true = LLMChatWithParsingRetryBlock(
                block_name="test_true",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                expand_lists=True,
                n=2,
            )

            single_dataset = Dataset.from_dict(
                {"messages": [sample_dataset["messages"][0]]}
            )

            result_false = block_false.generate(single_dataset)
            mock_completion.reset_mock()  # Reset call count
            result_true = block_true.generate(single_dataset)

            # expand_lists=False: 1 row with list values
            assert len(result_false) == 1
            assert isinstance(result_false[0]["answer"], list)
            assert result_false[0]["answer"] == ["Response 1", "Response 2"]

            # expand_lists=True: 2 rows with individual values
            assert len(result_true) == 2
            assert result_true[0]["answer"] == "Response 1"
            assert result_true[1]["answer"] == "Response 2"

    def test_expand_lists_flag_restored_on_exception(self, sample_dataset):
        """Test that expand_lists flag is properly restored even when parsing throws an exception."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            # Mock successful LLM response
            mock_completion.return_value = create_mock_response(
                "<answer>Response</answer>"
            )

            block = LLMChatWithParsingRetryBlock(
                block_name="test_flag_restore",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                expand_lists=False,  # Original setting should be restored
                n=1,
                parsing_max_retries=2,
            )

            # Verify initial state
            assert block.llm_parser.expand_lists is False

            # Mock the text parser to throw an exception during generate

            def mock_generate_with_exception(*args, **kwargs):
                # Always throw exception to test exception handling
                raise ValueError("Simulated parsing exception")

            with patch.object(
                block.text_parser, "generate", side_effect=mock_generate_with_exception
            ):
                single_dataset = Dataset.from_dict(
                    {"messages": [sample_dataset["messages"][0]]}
                )

                # This should fail due to parsing exceptions, but expand_lists should be restored
                with pytest.raises(MaxRetriesExceededError):
                    block.generate(single_dataset)

            # Critical assertion: expand_lists should remain unchanged
            assert block.llm_parser.expand_lists is False

    def test_partial_parses_rejected_expand_lists_false(self, sample_dataset):
        """Test that partial parses (missing some output columns) are rejected when expand_lists=False."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            # Mock responses where some have complete parses, others have partial parses
            mock_completion.return_value = create_mock_response(
                [
                    "<explanation>Complete explanation</explanation><answer>Complete answer</answer>",  # Complete
                    "<explanation>Partial explanation</explanation>No answer tag here",  # Partial - missing answer
                    "<explanation>Another explanation</explanation><answer>Another answer</answer>",  # Complete
                ]
            )

            block = LLMChatWithParsingRetryBlock(
                block_name="test_partial_reject",
                input_cols="messages",
                output_cols=["explanation", "answer"],  # Both columns required
                model="openai/gpt-4",
                start_tags=["<explanation>", "<answer>"],
                end_tags=["</explanation>", "</answer>"],
                expand_lists=False,
                n=2,  # Want 2 complete parses
                parsing_max_retries=3,
            )

            single_dataset = Dataset.from_dict(
                {"messages": [sample_dataset["messages"][0]]}
            )

            result = block.generate(single_dataset)

            # Should have 1 row with lists containing only the 2 complete parses
            assert len(result) == 1
            row = result[0]

            # Both lists should have exactly 2 items (only complete parses counted)
            assert len(row["explanation"]) == 2
            assert len(row["answer"]) == 2

            # Should contain only the complete parses, skipping the partial one
            assert row["explanation"] == ["Complete explanation", "Another explanation"]
            assert row["answer"] == ["Complete answer", "Another answer"]

            # Verify lists are properly aligned (same indices correspond to same response)
            assert (
                "Complete" in row["explanation"][0] and "Complete" in row["answer"][0]
            )
            assert "Another" in row["explanation"][1] and "Another" in row["answer"][1]

    def test_non_list_columns_preserved_both_modes(self):
        """Test that non-output columns are preserved with correct types in both expand_lists modes."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            mock_completion.return_value = create_mock_response(
                ["<answer>Response 1</answer>", "<answer>Response 2</answer>"]
            )

            # Create dataset with various column types that should be preserved
            test_dataset = Dataset.from_dict(
                {
                    "messages": [[{"role": "user", "content": "test"}]],
                    "context_id": [123],  # integer
                    "user_name": ["alice"],  # string
                    "is_premium": [True],  # boolean
                    "metadata": [{"key": "value"}],  # dict
                }
            )

            # Test expand_lists=False
            block_false = LLMChatWithParsingRetryBlock(
                block_name="test_preserve_false",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                expand_lists=False,
                n=2,
            )

            result_false = block_false.generate(test_dataset)

            # Should have 1 row (expand_lists=False)
            assert len(result_false) == 1
            row = result_false[0]

            # New parsed output column should be a list
            assert isinstance(row["answer"], list)
            assert row["answer"] == ["Response 1", "Response 2"]

            # All original columns should be preserved with original types
            assert row["context_id"] == 123
            assert isinstance(row["context_id"], int)
            assert row["user_name"] == "alice"
            assert isinstance(row["user_name"], str)
            assert row["is_premium"] is True
            assert isinstance(row["is_premium"], bool)
            assert row["metadata"] == {"key": "value"}
            assert isinstance(row["metadata"], dict)
            assert row["messages"] == [{"role": "user", "content": "test"}]
            assert isinstance(row["messages"], list)

            mock_completion.reset_mock()

            # Test expand_lists=True
            block_true = LLMChatWithParsingRetryBlock(
                block_name="test_preserve_true",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
                expand_lists=True,
                n=2,
            )

            result_true = block_true.generate(test_dataset)

            # Should have 2 rows (expand_lists=True)
            assert len(result_true) == 2

            for i, row in enumerate(result_true):
                # New parsed output column should be individual strings
                expected_answer = f"Response {i + 1}"
                assert row["answer"] == expected_answer
                assert isinstance(row["answer"], str)

                # All original columns should be preserved with original types
                assert row["context_id"] == 123
                assert isinstance(row["context_id"], int)
                assert row["user_name"] == "alice"
                assert isinstance(row["user_name"], str)
                assert row["is_premium"] is True
                assert isinstance(row["is_premium"], bool)
                assert row["metadata"] == {"key": "value"}
                assert isinstance(row["metadata"], dict)
                assert row["messages"] == [{"role": "user", "content": "test"}]
                assert isinstance(row["messages"], list)


class TestLLMChatWithParsingRetryBlockIntegration:
    """Integration tests with real internal block behavior."""

    def test_full_pipeline_integration(self, mock_litellm_completion, sample_dataset):
        """Test full pipeline integration between LLM and parser blocks."""
        # Configure complex response that tests both blocks
        content = (
            "Here's my analysis:\n"
            "<explanation>This is a detailed explanation of the problem.</explanation>\n"
            "<answer>The final answer is 42.</answer>\n"
            "Additional text that should be ignored."
        )
        mock_litellm_completion.return_value = create_mock_response(content)

        block = LLMChatWithParsingRetryBlock(
            block_name="test_integration",
            input_cols="messages",
            output_cols=["explanation", "answer"],
            model="openai/gpt-4",
            api_key="test-key",
            start_tags=["<explanation>", "<answer>"],
            end_tags=["</explanation>", "</answer>"],
            temperature=0.7,
            max_tokens=200,
            parsing_max_retries=3,
        )

        result = block.generate(sample_dataset)

        # Verify complete pipeline works
        assert len(result) == 2
        for row in result:
            assert (
                row["explanation"] == "This is a detailed explanation of the problem."
            )
            assert row["answer"] == "The final answer is 42."
            # Original message data should be preserved
            assert "messages" in row

        # Verify LLM was called with correct parameters
        call_kwargs = mock_litellm_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 200

    def test_cleanup_tags_integration(self, mock_litellm_completion, sample_dataset):
        """Test integration with parser cleanup tags using regex parsing."""
        # Use regex parsing since cleanup tags only work with regex, not tag parsing
        content = "Answer: This has <br>line breaks</br> to clean"
        mock_litellm_completion.return_value = create_mock_response(content)

        block = LLMChatWithParsingRetryBlock(
            block_name="test_cleanup",
            input_cols="messages",
            output_cols="clean_answer",
            model="openai/gpt-4",
            api_key="test-key",
            parsing_pattern=r"Answer: (.*?)(?:\n|$)",
            parser_cleanup_tags=["<br>", "</br>"],
        )

        result = block.generate(sample_dataset)

        assert len(result) == 2
        # The cleanup should remove <br> and </br> tags from regex parsing
        assert all(
            row["clean_answer"] == "This has line breaks to clean" for row in result
        )

    def test_error_propagation_from_internal_blocks(self, sample_dataset):
        """Test that errors from internal blocks are properly propagated."""
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.completion"
        ) as mock_completion:
            # Make LLM block raise a specific error
            from sdg_hub.core.blocks.llm.error_handler import AuthenticationError

            mock_completion.side_effect = AuthenticationError(
                "Invalid API key", llm_provider="openai", model="gpt-4"
            )

            block = LLMChatWithParsingRetryBlock(
                block_name="test_error_prop",
                input_cols="messages",
                output_cols="answer",
                model="openai/gpt-4",
                start_tags=["<answer>"],
                end_tags=["</answer>"],
            )

            single_dataset = Dataset.from_dict(
                {"messages": [sample_dataset["messages"][0]]}
            )

            # Error should propagate through and eventually cause MaxRetriesExceededError
            with pytest.raises(MaxRetriesExceededError):
                block.generate(single_dataset)

    def test_llm_chat_with_parsing_retry_parameter_forwarding(self):
        """Test parameter forwarding for LLMChatWithParsingRetryBlock.

        This block has a different structure but must follow the same parameter
        forwarding pattern as evaluation blocks.
        """
        block = LLMChatWithParsingRetryBlock(
            block_name="test_retry",
            input_cols=["messages"],
            output_cols=["parsed_output"],
            parsing_pattern=r"test pattern",  # Required for TextParser
            parsing_max_retries=3,
        )

        # Test LLM parameters
        llm_params = {
            "model": "test-model",
            "extra_body": {"test": "value"},
            "extra_headers": {"X-Test": "header"},
            "temperature": 0.8,
        }

        # Test parser parameters
        parser_params = {
            "start_tags": ["<start>"],
            "end_tags": ["<end>"],
        }

        all_params = {**llm_params, **parser_params}

        # hasattr() must work for Flow detection
        for param_name in all_params:
            assert hasattr(
                block, param_name
            ), f"LLMChatWithParsingRetryBlock must have attribute '{param_name}'"

        # Parameter setting must work
        for param_name, param_value in all_params.items():
            setattr(block, param_name, param_value)

        # Parameters must be accessible
        for param_name, expected_value in all_params.items():
            actual_value = getattr(block, param_name)
            assert actual_value == expected_value

        # LLM parameters must forward to internal LLM block
        for param_name, expected_value in llm_params.items():
            internal_value = getattr(block.llm_chat, param_name)
            assert (
                internal_value == expected_value
            ), f"LLM parameter {param_name} not forwarded to internal LLM block"

        # Parser parameters must forward to internal parser block
        for param_name, expected_value in parser_params.items():
            internal_value = getattr(block.text_parser, param_name)
            assert (
                internal_value == expected_value
            ), f"Parser parameter {param_name} not forwarded to internal parser block"

    def test_llm_chat_with_parsing_retry_validation_requirements(self):
        """Test that LLMChatWithParsingRetryBlock properly validates parsing requirements."""
        # Should work with parsing_pattern
        block1 = LLMChatWithParsingRetryBlock(
            block_name="test1",
            input_cols=["messages"],
            output_cols=["output"],
            parsing_pattern=r"test",
        )
        assert block1.parsing_pattern == r"test"

        # Should work with start_tags/end_tags
        block2 = LLMChatWithParsingRetryBlock(
            block_name="test2",
            input_cols=["messages"],
            output_cols=["output"],
            start_tags=["<start>"],
            end_tags=["<end>"],
        )
        assert block2.start_tags == ["<start>"]
        assert block2.end_tags == ["<end>"]

        # Should work with both (parsing_pattern takes precedence)
        block3 = LLMChatWithParsingRetryBlock(
            block_name="test3",
            input_cols=["messages"],
            output_cols=["output"],
            parsing_pattern=r"test",
            start_tags=["<start>"],
            end_tags=["<end>"],
        )
        assert block3.parsing_pattern == r"test"
        assert block3.start_tags == ["<start>"]
