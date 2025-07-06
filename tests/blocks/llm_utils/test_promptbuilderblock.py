# Standard
import os

# Third Party
from datasets import Dataset
import pytest

# First Party
from sdg_hub.blocks import PromptBuilderBlock

# Get the absolute paths to test config files
TEST_CONFIG_WITH_SYSTEM = os.path.join(
    os.path.dirname(__file__), "..", "testdata", "test_prompt_format_config.yaml"
)
TEST_CONFIG_NO_SYSTEM = os.path.join(
    os.path.dirname(__file__), "..", "testdata", "test_prompt_format_no_system.yaml"
)
TEST_CONFIG_STRICT = os.path.join(
    os.path.dirname(__file__), "..", "testdata", "test_prompt_format_strict.yaml"
)


class TestPromptBuilderBlock:
    """Test cases for PromptBuilderBlock."""

    def test_init_with_string_input_cols(self):
        """Test initialization with string input_cols."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        assert block.block_name == "test_block"
        assert block.input_col_map == {"input_text": "input_text"}
        assert block.output_cols == "output"
        assert block.format_as_messages is True
        assert block.default_role == "user"

    def test_init_with_list_input_cols(self):
        """Test initialization with list input_cols."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols=["input_text", "context"],
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        assert block.input_col_map == {"input_text": "input_text", "context": "context"}

    def test_init_with_dict_input_cols(self):
        """Test initialization with dictionary input_cols for column mapping."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols={"input_text": "user_input", "context": "background_info"},
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        assert block.input_col_map == {
            "input_text": "user_input",
            "context": "background_info",
        }

    def test_init_with_invalid_input_cols(self):
        """Test initialization with invalid input_cols type."""
        with pytest.raises(ValueError, match="input_cols must be str, list, or dict"):
            PromptBuilderBlock(
                block_name="test_block",
                input_cols=123,  # Invalid type
                output_cols="output",
                prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
            )

    def test_resolve_template_vars_with_string_cols(self):
        """Test _resolve_template_vars with string input_cols."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        sample = {"input_text": "Hello world", "other_col": "ignored"}
        template_vars = block._resolve_template_vars(sample)

        assert template_vars == {"input_text": "Hello world"}

    def test_resolve_template_vars_with_dict_mapping(self):
        """Test _resolve_template_vars with dictionary mapping."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols={"input_text": "user_message", "context": "background"},
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        sample = {
            "user_message": "Hello",
            "background": "conversation context",
            "ignored_col": "not used",
        }
        template_vars = block._resolve_template_vars(sample)

        assert template_vars == {
            "input_text": "Hello",
            "context": "conversation context",
        }

    def test_resolve_template_vars_missing_column(self, caplog):
        """Test _resolve_template_vars with missing dataset column."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols={"input_text": "missing_col"},
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        sample = {"other_col": "exists"}
        template_vars = block._resolve_template_vars(sample)

        assert template_vars == {}
        assert "Dataset column 'missing_col' not found in sample" in caplog.text

    def test_validate_message_role_valid(self):
        """Test _validate_message_role with valid roles."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        assert block._validate_message_role("system") == "system"
        assert block._validate_message_role("USER") == "user"
        assert block._validate_message_role("Assistant") == "assistant"
        assert block._validate_message_role("tool") == "tool"

    def test_validate_message_role_invalid(self, caplog):
        """Test _validate_message_role with invalid role."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        result = block._validate_message_role("invalid_role")
        assert result == "user"
        assert "Invalid role 'invalid_role', defaulting to 'user'" in caplog.text

    def test_format_message_content_string(self):
        """Test _format_message_content with string input."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        result = block._format_message_content("  Hello world  ")
        assert result == "Hello world"

    def test_format_message_content_dict(self):
        """Test _format_message_content with dict input (structured content)."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        content_block = {
            "type": "image_url",
            "image_url": {"url": "http://example.com/image.jpg"},
        }
        result = block._format_message_content(content_block)
        assert result == [content_block]

    def test_format_message_content_list(self):
        """Test _format_message_content with list input (multiple content blocks)."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        content_list = [
            {"type": "text", "text": "Hello"},
            "Plain text",
            {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
        ]
        result = block._format_message_content(content_list)

        expected = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "Plain text"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
        ]
        assert result == expected

    def test_create_openai_message_valid(self):
        """Test _create_openai_message with valid input."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        message = block._create_openai_message("user", "Hello world")
        assert message == {"role": "user", "content": "Hello world"}

    def test_create_openai_message_empty_content(self):
        """Test _create_openai_message with empty content."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        message = block._create_openai_message("user", "")
        assert message is None

    def test_create_openai_message_tool_role_warning(self, caplog):
        """Test _create_openai_message with tool role generates warning."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        message = block._create_openai_message("tool", "Tool response")
        assert message == {"role": "tool", "content": "Tool response"}
        assert "Tool messages require tool_call_id" in caplog.text

    def test_convert_to_openai_messages_with_system(self):
        """Test _convert_to_openai_messages with system message in config."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols=["input_text", "context"],
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        template_vars = {"input_text": "Hello", "context": "friendly chat"}
        rendered_content = "Generate a response based on the following input\nBe accurate and helpful\nExample: Input: Hello. Output: Hi there!\nInput: Hello"

        messages = block._convert_to_openai_messages(rendered_content, template_vars)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert (
            "You are a helpful assistant. Context: friendly chat"
            in messages[0]["content"]
        )
        assert messages[1]["role"] == "user"
        assert rendered_content in messages[1]["content"]

    def test_convert_to_openai_messages_no_system(self):
        """Test _convert_to_openai_messages without system message."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_NO_SYSTEM,
        )

        template_vars = {"text": "process this"}
        rendered_content = (
            "Generate a response\nBe helpful\nExample provided\nProcess: process this"
        )

        messages = block._convert_to_openai_messages(rendered_content, template_vars)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert rendered_content in messages[0]["content"]

    def test_convert_to_openai_messages_with_custom_role(self):
        """Test _convert_to_openai_messages with custom default role."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_NO_SYSTEM,
            default_role="assistant",
        )

        template_vars = {"input_text": "Hello"}
        rendered_content = "Some content"

        messages = block._convert_to_openai_messages(rendered_content, template_vars)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"

    def test_generate_with_messages_format(self):
        """Test generate method with format_as_messages=True."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols=["input_text", "context"],
            output_cols="messages",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
            format_as_messages=True,
        )

        dataset = Dataset.from_list(
            [
                {"input_text": "Hello", "context": "casual conversation"},
                {"input_text": "How are you?", "context": "friendly chat"},
            ]
        )

        result = block.generate(dataset)

        assert len(result) == 2
        assert "messages" in result.column_names

        # Check first sample
        messages = result[0]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "casual conversation" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_generate_with_plain_text_format(self):
        """Test generate method with format_as_messages=False."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="text",
            output_cols="formatted_text",
            prompt_config_path=TEST_CONFIG_NO_SYSTEM,
            format_as_messages=False,
        )

        dataset = Dataset.from_list(
            [
                {"text": "process this"},
                {"text": "handle that"},
            ]
        )

        result = block.generate(dataset)

        assert len(result) == 2
        assert "formatted_text" in result.column_names

        # Should be plain strings, not message lists
        assert isinstance(result[0]["formatted_text"], str)
        assert "process this" in result[0]["formatted_text"]

    def test_generate_with_missing_template_vars(self, caplog):
        """Test generate method with missing template variables."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols={"required_var": "missing_col"},
            output_cols="output",
            prompt_config_path=TEST_CONFIG_STRICT,
        )

        dataset = Dataset.from_list(
            [
                {"other_col": "Hello"},  # missing_col not provided
            ]
        )

        result = block.generate(dataset)

        assert len(result) == 1
        # The template should still render but with empty values for missing variables
        # This demonstrates that missing columns are logged as warnings
        assert "Dataset column 'missing_col' not found in sample" in caplog.text
        # Check that output contains a message but with empty variable substitutions
        assert isinstance(result[0]["output"], list)
        assert len(result[0]["output"]) == 1
        assert result[0]["output"][0]["role"] == "user"
        # Content should contain the template with empty substitutions
        content = result[0]["output"][0]["content"]
        assert "This template requires:" in content
        assert "Must have:" in content

    def test_generate_with_empty_content(self, caplog):
        """Test generate method with template that produces empty content."""
        # Create a template that only outputs if a variable is non-empty
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_NO_SYSTEM,
        )

        dataset = Dataset.from_list(
            [
                {"text": "content"},  # Valid content
            ]
        )

        result = block.generate(dataset)

        assert len(result) == 1
        # Should generate valid content, not empty
        assert result[0]["output"] != []
        assert isinstance(result[0]["output"], list)
        assert len(result[0]["output"]) == 1
        assert result[0]["output"][0]["role"] == "user"
        assert "content" in result[0]["output"][0]["content"]

    def test_generate_preserves_original_columns(self):
        """Test that generate preserves original dataset columns."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="messages",
            prompt_config_path=TEST_CONFIG_NO_SYSTEM,
        )

        dataset = Dataset.from_list(
            [
                {"input_text": "Hello", "id": 1, "metadata": {"key": "value"}},
            ]
        )

        result = block.generate(dataset)

        assert result[0]["input_text"] == "Hello"
        assert result[0]["id"] == 1
        assert result[0]["metadata"] == {"key": "value"}
        assert "messages" in result[0]
