# Standard
import os

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.llm import PromptBuilderBlock
from sdg_hub.core.blocks.llm.prompt_builder_block import ChatMessage
import pytest

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
TEST_CONFIG_NO_USER_MESSAGES = os.path.join(
    os.path.dirname(__file__), "..", "testdata", "test_prompt_no_user_messages.yaml"
)
TEST_CONFIG_INVALID_FINAL_ROLE = os.path.join(
    os.path.dirname(__file__), "..", "testdata", "test_prompt_invalid_final_role.yaml"
)


class TestChatMessage:
    """Test cases for ChatMessage Pydantic model."""

    def test_valid_chat_message(self):
        """Test creating a valid ChatMessage."""
        message = ChatMessage(role="user", content="Hello world")
        assert message.role == "user"
        assert message.content == "Hello world"

    def test_chat_message_strips_whitespace(self):
        """Test that ChatMessage strips whitespace from content."""
        message = ChatMessage(role="system", content="  Hello world  ")
        assert message.content == "Hello world"

    def test_chat_message_invalid_role(self):
        """Test that ChatMessage rejects invalid roles."""
        with pytest.raises(ValueError, match="Input should be 'system'"):
            ChatMessage(role="invalid", content="Hello")

    def test_chat_message_empty_content(self):
        """Test that ChatMessage rejects empty content."""
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            ChatMessage(role="user", content="")

    def test_chat_message_whitespace_only_content(self):
        """Test that ChatMessage rejects whitespace-only content."""
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            ChatMessage(role="user", content="   ")

    def test_chat_message_serialization(self):
        """Test that ChatMessage can be serialized to dict."""
        message = ChatMessage(role="assistant", content="Hello!")
        serialized = message.model_dump()
        assert serialized == {"role": "assistant", "content": "Hello!"}


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
        assert block.input_cols == ["input_text"]
        assert block.output_cols == ["output"]
        assert block.format_as_messages is True

    def test_init_with_list_input_cols(self):
        """Test initialization with list input_cols."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols=["input_text", "context"],
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        assert block.input_cols == ["input_text", "context"]

    def test_init_with_dict_input_cols(self):
        """Test initialization with dictionary input_cols for column mapping."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols={"input_text": "user_input", "context": "background_info"},
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        assert block.input_cols == {
            "input_text": "user_input",
            "context": "background_info",
        }

    def test_init_with_invalid_input_cols(self):
        """Test initialization with invalid input_cols type."""
        with pytest.raises(ValueError, match="Invalid column specification"):
            PromptBuilderBlock(
                block_name="test_block",
                input_cols=123,  # Invalid type
                output_cols="output",
                prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
            )

    def test_init_with_multiple_output_cols(self):
        """Test initialization with multiple output columns raises error."""
        with pytest.raises(
            ValueError, match="PromptBuilderBlock expects exactly one output column"
        ):
            PromptBuilderBlock(
                block_name="test_block",
                input_cols="input_text",
                output_cols=["output1", "output2"],
                prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
            )

    def test_init_with_invalid_config_path(self):
        """Test initialization with invalid config path raises error."""
        with pytest.raises(FileNotFoundError):
            PromptBuilderBlock(
                block_name="test_block",
                input_cols="input_text",
                output_cols="output",
                prompt_config_path="/nonexistent/path.yaml",
            )

    def test_init_with_invalid_yaml_config(self, tmp_path):
        """Test initialization with invalid YAML config raises error."""
        # Create a file with invalid YAML
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [unclosed")

        with pytest.raises(Exception):  # Could be ValueError or yaml.YAMLError
            PromptBuilderBlock(
                block_name="test_block",
                input_cols="input_text",
                output_cols="output",
                prompt_config_path=str(invalid_config),
            )

    def test_init_with_non_list_config(self, tmp_path):
        """Test initialization with non-list config raises error."""
        # Create a file with non-list YAML
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("role: user\ncontent: hello")

        with pytest.raises(ValueError):
            PromptBuilderBlock(
                block_name="test_block",
                input_cols="input_text",
                output_cols="output",
                prompt_config_path=str(invalid_config),
            )

    def test_init_with_no_user_messages(self):
        """Test initialization with no user messages raises error."""
        with pytest.raises(
            ValueError,
            match="Template must contain at least one message with role='user'",
        ):
            PromptBuilderBlock(
                block_name="test_block",
                input_cols="input_text",
                output_cols="output",
                prompt_config_path=TEST_CONFIG_NO_USER_MESSAGES,
            )

    def test_init_with_invalid_final_role(self):
        """Test initialization with non-user final message raises error."""
        with pytest.raises(ValueError, match="The final message must have role='user'"):
            PromptBuilderBlock(
                block_name="test_block",
                input_cols="input_text",
                output_cols="output",
                prompt_config_path=TEST_CONFIG_INVALID_FINAL_ROLE,
            )

    def test_init_with_invalid_message_structure(self, tmp_path):
        """Test initialization with invalid message structure raises error."""
        # Create a file with missing required fields
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("- role: user\n- content: hello")

        with pytest.raises(
            ValueError, match="Message 0 must have 'role' and 'content' fields"
        ):
            PromptBuilderBlock(
                block_name="test_block",
                input_cols="input_text",
                output_cols="output",
                prompt_config_path=str(invalid_config),
            )

    def test_init_with_invalid_role(self, tmp_path):
        """Test initialization with invalid role raises error."""
        # Create a file with invalid role
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text(
            "- role: invalid_role\n  content: hello\n- role: user\n  content: world"
        )

        with pytest.raises(ValueError):
            PromptBuilderBlock(
                block_name="test_block",
                input_cols="input_text",
                output_cols="output",
                prompt_config_path=str(invalid_config),
            )

    def test_resolve_template_vars_with_string_cols(self):
        """Test resolve_template_vars with string input_cols."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        sample = {"input_text": "Hello world", "other_col": "ignored"}
        template_vars = block.prompt_renderer.resolve_template_vars(
            sample, block.input_cols
        )

        assert template_vars == {"input_text": "Hello world"}

    def test_resolve_template_vars_with_list_cols(self):
        """Test resolve_template_vars with list input_cols."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols=["input_text", "context"],
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        sample = {"input_text": "Hello", "context": "friendly", "other_col": "ignored"}
        template_vars = block.prompt_renderer.resolve_template_vars(
            sample, block.input_cols
        )

        assert template_vars == {"input_text": "Hello", "context": "friendly"}

    def test_resolve_template_vars_with_dict_mapping(self):
        """Test resolve_template_vars with dictionary mapping."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols={"user_message": "input_text", "background": "context"},
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        sample = {
            "user_message": "Hello",
            "background": "conversation context",
            "ignored_col": "not used",
        }
        template_vars = block.prompt_renderer.resolve_template_vars(
            sample, block.input_cols
        )

        assert template_vars == {
            "input_text": "Hello",
            "context": "conversation context",
        }

    def test_resolve_template_vars_missing_column(self, caplog):
        """Test resolve_template_vars with missing dataset column."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols={"missing_col": "input_text"},
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        sample = {"other_col": "exists"}
        template_vars = block.prompt_renderer.resolve_template_vars(
            sample, block.input_cols
        )

        assert template_vars == {}
        assert "Dataset column 'missing_col' not found in sample" in caplog.text

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
        assert "Hello" in messages[1]["content"]

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
        assert "user:" in result[0]["formatted_text"]  # Should include role prefix

    def test_generate_with_missing_template_vars(self, caplog):
        """Test generate method with missing template variables."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols={"missing_col": "required_var"},
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

    def test_generate_with_template_render_error(self, caplog):
        """Test generate method handles template rendering errors gracefully."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_NO_SYSTEM,
        )

        dataset = Dataset.from_list([{"text": "valid content"}])

        # Mock a template that will fail to render
        block.prompt_renderer.message_templates[0].content_template.render = (
            lambda x: ""
        )  # Empty render

        result = block.generate(dataset)

        assert len(result) == 1
        # Should have empty output when no messages are generated
        assert result[0]["output"] == []
        assert "No valid messages generated" in caplog.text

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

    def test_baseblock_integration(self):
        """Test that PromptBuilderBlock properly integrates with BaseBlock."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="input_text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_NO_SYSTEM,
        )

        # Test BaseBlock properties
        assert block.block_name == "test_block"
        assert block.input_cols == ["input_text"]
        assert block.output_cols == ["output"]

        # Test that it has BaseBlock methods
        assert hasattr(block, "_validate_custom")
        assert hasattr(block, "_log_input_data")
        assert hasattr(block, "_log_output_data")

        # Test get_info method from BaseBlock
        info = block.get_info()
        assert info["block_name"] == "test_block"
        assert info["block_type"] == "PromptBuilderBlock"
        assert info["input_cols"] == ["input_text"]
        assert info["output_cols"] == ["output"]

    def test_validate_custom_with_missing_variables(self):
        """Test _validate_custom raises error for missing template variables."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="wrong_col",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_STRICT,
        )

        dataset = Dataset.from_list([{"wrong_col": "value"}])

        with pytest.raises(Exception):  # Should raise TemplateValidationError
            block._validate_custom(dataset)

    def test_message_templates_compilation(self):
        """Test that message templates are properly compiled."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols=["input_text", "context"],
            output_cols="output",
            prompt_config_path=TEST_CONFIG_WITH_SYSTEM,
        )

        message_templates = block.prompt_renderer.message_templates
        assert len(message_templates) == 2
        assert message_templates[0].role == "system"
        assert message_templates[1].role == "user"
        assert hasattr(message_templates[0], "content_template")
        assert hasattr(message_templates[1], "content_template")

    def test_chat_message_validation_in_generate(self):
        """Test that ChatMessage validation works during generation."""
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="text",
            output_cols="output",
            prompt_config_path=TEST_CONFIG_NO_SYSTEM,
        )

        dataset = Dataset.from_list([{"text": "Hello world"}])
        result = block.generate(dataset)

        # Check that generated messages are valid ChatMessage objects when serialized
        messages = result[0]["output"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Generate a response" in messages[0]["content"]
        assert "Be helpful" in messages[0]["content"]
        assert "Example provided" in messages[0]["content"]
        assert "Process: Hello world" in messages[0]["content"]

    def test_environment_reuse_with_custom_filter(self, tmp_path):
        """Test that get_required_variables uses the template's original environment.

        This test creates a template with a custom filter, then checks that
        get_required_variables can properly parse it using the same environment.
        """
        # Create a config with a template that uses a custom filter
        config_content = """
- role: user
  content: "Process: {{ text | upper }}"
"""
        config_path = tmp_path / "custom_filter_config.yaml"
        config_path.write_text(config_content)

        # Create block - this should work fine since the template itself doesn't use custom filters during creation
        block = PromptBuilderBlock(
            block_name="test_block",
            input_cols="text",
            output_cols="output",
            prompt_config_path=str(config_path),
        )

        # Add a custom filter to the template's environment
        def custom_filter(text):
            return f"CUSTOM_{text}"

        # Add the filter to the template's environment
        block.prompt_renderer.message_templates[0].content_template.environment.filters[
            "custom"
        ] = custom_filter

        # Now modify the original source to use the custom filter
        block.prompt_renderer.message_templates[
            0
        ].original_source = "Process: {{ text | custom }}"

        # This should work if we use the template's environment, but fail if we create a new Environment
        # because the new environment won't have the custom filter
        try:
            required_vars = block.prompt_renderer.get_required_variables()
            # If we get here, the method used the template's environment correctly
            assert "text" in required_vars
        except Exception as e:
            # If we get an exception, it means a new Environment was created without the custom filter
            assert "custom" in str(e) or "filter" in str(e), f"Unexpected error: {e}"
            # This is the bug we want to expose
            pytest.fail(
                "get_required_variables should use the template's original environment"
            )
