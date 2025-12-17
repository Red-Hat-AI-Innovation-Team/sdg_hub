# SPDX-License-Identifier: Apache-2.0
"""Tests for TranslationBlock."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pandas as pd
import pytest

# First Party
from sdg_hub.core.blocks.llm import TranslationBlock
from sdg_hub.core.utils.error_handling import BlockValidationError


class MockMessage:
    """Mock message class that behaves like LiteLLM message."""

    def __init__(self, content):
        self.content = content


@pytest.fixture
def mock_litellm_completion():
    """Mock LiteLLM completion function."""
    with patch(
        "sdg_hub.core.blocks.llm.translation_block.completion"
    ) as mock_completion:
        mock_response = MagicMock()
        choice = MagicMock()
        choice.message = MockMessage("Traducido exitosamente")  # Translated text
        mock_response.choices = [choice]
        mock_completion.return_value = mock_response
        yield mock_completion


@pytest.fixture
def mock_litellm_acompletion():
    """Mock LiteLLM async completion function."""
    with patch(
        "sdg_hub.core.blocks.llm.translation_block.acompletion"
    ) as mock_acompletion:
        mock_response = MagicMock()
        choice = MagicMock()
        choice.message = MockMessage("Traducido async")  # Translated text
        mock_response.choices = [choice]
        mock_acompletion.return_value = mock_response
        yield mock_acompletion


class TestTranslationBlockInitialization:
    """Test TranslationBlock initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        block = TranslationBlock(
            block_name="test_translation",
            input_cols="english_text",
            output_cols="spanish_text",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        assert block.block_name == "test_translation"
        assert block.input_cols == ["english_text"]
        assert block.output_cols == ["spanish_text"]
        assert block.source_language == "en"
        assert block.target_language == "es"
        assert block.model == "openai/gpt-4"

    def test_init_with_unsupported_language_code(self, caplog):
        """Test initialization with unsupported language code logs warning."""
        block = TranslationBlock(
            block_name="test_translation",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="xyz",  # Unsupported code
            model="openai/gpt-4",
        )

        assert block.target_language == "xyz"
        # Check that warning was logged
        assert "Language code 'xyz' not in predefined list" in caplog.text

    def test_init_normalizes_input_cols(self):
        """Test that input_cols is normalized to list."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",  # String
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        assert block.input_cols == ["text"]

    def test_init_normalizes_output_cols(self):
        """Test that output_cols is normalized to list."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",  # String
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        assert block.output_cols == ["translated"]

    def test_init_rejects_multiple_input_cols(self):
        """Test that multiple input columns are rejected."""
        with pytest.raises(ValueError, match="exactly one input column"):
            TranslationBlock(
                block_name="test",
                input_cols=["col1", "col2"],
                output_cols="translated",
                source_language="en",
                target_language="es",
                model="openai/gpt-4",
            )

    def test_init_rejects_multiple_output_cols(self):
        """Test that multiple output columns are rejected."""
        with pytest.raises(ValueError, match="exactly one output column"):
            TranslationBlock(
                block_name="test",
                input_cols="text",
                output_cols=["col1", "col2"],
                source_language="en",
                target_language="es",
                model="openai/gpt-4",
            )

    def test_init_with_litellm_params(self):
        """Test initialization with LiteLLM parameters."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
            temperature=0.3,
            max_tokens=2048,
            top_p=0.9,
        )

        # These should be stored via extra="allow"
        assert hasattr(block, "temperature")
        assert hasattr(block, "max_tokens")
        assert hasattr(block, "top_p")


class TestTranslationBlockFormatDetection:
    """Test input format detection."""

    def test_detect_messages_format(self):
        """Test detection of messages format."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello world"},
        ]

        assert block._detect_input_format(messages) == "messages"

    def test_detect_text_format(self):
        """Test detection of text format."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        text = "Hello world"

        assert block._detect_input_format(text) == "text"

    def test_detect_invalid_format(self):
        """Test detection of invalid format."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        with pytest.raises(BlockValidationError, match="Unsupported input format"):
            block._detect_input_format({"invalid": "format"})


class TestTranslationBlockTextTranslation:
    """Test translation of plain text."""

    def test_translate_text_sync(self, mock_litellm_completion):
        """Test synchronous text translation."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        text = "Hello world"
        completion_kwargs = {"model": "openai/gpt-4"}

        result = block._translate_text(text, completion_kwargs)

        assert result == "Traducido exitosamente"
        mock_litellm_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_translate_text_async(self, mock_litellm_acompletion):
        """Test asynchronous text translation."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        text = "Hello world"
        completion_kwargs = {"model": "openai/gpt-4"}

        result = await block._translate_text_async(text, completion_kwargs)

        assert result == "Traducido async"
        mock_litellm_acompletion.assert_called_once()


class TestTranslationBlockMessagesTranslation:
    """Test translation of messages format."""

    def test_translate_messages_sync(self, mock_litellm_completion):
        """Test synchronous messages translation."""
        block = TranslationBlock(
            block_name="test",
            input_cols="messages",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello world"},
        ]
        completion_kwargs = {"model": "openai/gpt-4"}

        result = block._translate_messages(messages, completion_kwargs)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Traducido exitosamente"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Traducido exitosamente"
        assert mock_litellm_completion.call_count == 2  # Once per message

    @pytest.mark.asyncio
    async def test_translate_messages_async(self, mock_litellm_acompletion):
        """Test asynchronous messages translation."""
        block = TranslationBlock(
            block_name="test",
            input_cols="messages",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello world"},
        ]
        completion_kwargs = {"model": "openai/gpt-4"}

        result = await block._translate_messages_async(messages, completion_kwargs)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Traducido async"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Traducido async"
        assert mock_litellm_acompletion.call_count == 2


class TestTranslationBlockGenerate:
    """Test the main generate() method."""

    def test_generate_with_text_format(self, mock_litellm_completion):
        """Test generate with text format input."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        df = pd.DataFrame({"text": ["Hello", "World"]})

        result = block.generate(df)

        assert "translated" in result.columns
        assert len(result) == 2
        assert result["translated"].tolist() == [
            "Traducido exitosamente",
            "Traducido exitosamente",
        ]

    def test_generate_with_messages_format(self, mock_litellm_completion):
        """Test generate with messages format input."""
        block = TranslationBlock(
            block_name="test",
            input_cols="messages",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [{"role": "user", "content": "World"}]

        df = pd.DataFrame({"messages": [messages1, messages2]})

        result = block.generate(df)

        assert "translated" in result.columns
        assert len(result) == 2
        assert all(isinstance(msg, list) for msg in result["translated"])

    def test_generate_without_model_raises_error(self):
        """Test that generate() without model raises error."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            # No model specified
        )

        df = pd.DataFrame({"text": ["Hello"]})

        with pytest.raises(BlockValidationError, match="Model not configured"):
            block.generate(df)

    def test_generate_preserves_original_columns(self, mock_litellm_completion):
        """Test that generate preserves original DataFrame columns."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        df = pd.DataFrame({"text": ["Hello"], "other_col": ["data"]})

        result = block.generate(df)

        assert "text" in result.columns
        assert "other_col" in result.columns
        assert "translated" in result.columns


class TestTranslationBlockValidation:
    """Test custom validation logic."""

    def test_validate_valid_text_format(self, mock_litellm_completion):
        """Test validation accepts valid text format."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        df = pd.DataFrame({"text": ["Hello", "World"]})

        # Should not raise
        block._validate_custom(df)

    def test_validate_valid_messages_format(self, mock_litellm_completion):
        """Test validation accepts valid messages format."""
        block = TranslationBlock(
            block_name="test",
            input_cols="messages",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [{"role": "user", "content": "World"}]

        df = pd.DataFrame({"messages": [messages1, messages2]})

        # Should not raise
        block._validate_custom(df)

    def test_validate_invalid_format_raises_error(self, mock_litellm_completion):
        """Test validation rejects invalid format."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        df = pd.DataFrame({"text": [{"invalid": "format"}]})

        with pytest.raises(BlockValidationError, match="Invalid prompt format"):
            block._validate_custom(df)


class TestTranslationBlockPromptBuilding:
    """Test translation prompt building."""

    def test_build_translation_prompt(self):
        """Test building translation prompt from template."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        content = "Hello world"
        messages = block._build_translation_prompt(content)

        assert isinstance(messages, list)
        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "English" in messages[1]["content"]
        assert "Spanish" in messages[1]["content"]
        assert "Hello world" in messages[1]["content"]

    def test_build_translation_prompt_with_unsupported_language(self):
        """Test prompt building with unsupported language code."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="xyz",  # Unsupported
            target_language="abc",  # Unsupported
            model="openai/gpt-4",
        )

        content = "Hello world"
        messages = block._build_translation_prompt(content)

        # Should still work, using title-cased version of code
        assert "Xyz" in messages[1]["content"]
        assert "Abc" in messages[1]["content"]


class TestTranslationBlockCompletionKwargs:
    """Test building completion kwargs."""

    def test_build_completion_kwargs_basic(self):
        """Test building basic completion kwargs."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
            temperature=0.3,
        )

        kwargs = block._build_completion_kwargs()

        assert kwargs["model"] == "openai/gpt-4"
        assert kwargs["temperature"] == 0.3
        assert kwargs["drop_params"] is True
        assert "source_language" not in kwargs  # Block-only field
        assert "target_language" not in kwargs  # Block-only field

    def test_build_completion_kwargs_with_overrides(self):
        """Test building completion kwargs with runtime overrides."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
            temperature=0.3,
        )

        kwargs = block._build_completion_kwargs(temperature=0.7, max_tokens=1000)

        assert kwargs["temperature"] == 0.7  # Overridden
        assert kwargs["max_tokens"] == 1000  # New param

    def test_build_completion_kwargs_excludes_block_fields(self):
        """Test that block-only fields are excluded from completion kwargs."""
        block = TranslationBlock(
            block_name="test",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
        )

        kwargs = block._build_completion_kwargs()

        assert "block_name" not in kwargs
        assert "input_cols" not in kwargs
        assert "output_cols" not in kwargs
        assert "source_language" not in kwargs
        assert "target_language" not in kwargs
        assert "async_mode" not in kwargs


class TestTranslationBlockRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ output."""
        block = TranslationBlock(
            block_name="test_translation",
            input_cols="text",
            output_cols="translated",
            source_language="en",
            target_language="es",
            model="openai/gpt-4",
            async_mode=True,
        )

        repr_str = repr(block)

        assert "TranslationBlock" in repr_str
        assert "test_translation" in repr_str
        assert "en→es" in repr_str
        assert "openai/gpt-4" in repr_str
        assert "async_mode=True" in repr_str
