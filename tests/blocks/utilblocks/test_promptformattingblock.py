# SPDX-License-Identifier: Apache-2.0
"""Tests for PromptFormattingBlock."""

# Standard
import tempfile
import yaml
from unittest.mock import patch, MagicMock

# Third Party
import pytest
from datasets import Dataset

# Local
from sdg_hub.blocks import PromptFormattingBlock


class TestPromptFormattingBlock:
    """Test cases for PromptFormattingBlock."""

    @pytest.fixture
    def basic_config(self):
        """Basic test configuration."""
        return {
            "system": "You are a helpful assistant.",
            "introduction": "Please help with: {{task}}",
            "principles": "Be helpful and accurate.",
            "examples": "Example: {{example}}",
            "generation": "Now help with: {{task}}",
        }

    @pytest.fixture
    def custom_role_mapping_config(self):
        """Configuration with custom role mapping."""
        return {
            "system": "You are a helpful assistant.",
            "introduction": "Please help with: {{task}}",
            "principles": "Be helpful and accurate.",
            "examples": "Example: {{example}}",
            "generation": "Now help with: {{task}}",
            "role_mapping": {
                "system": "system",
                "introduction": "user",
                "principles": "assistant",  # Custom mapping
                "examples": "user",
                "generation": "user",
            },
        }

    @pytest.fixture
    def test_data(self):
        """Test dataset."""
        return [{"task": "summarize this text", "example": "This is an example"}]

    def create_temp_config(self, config):
        """Helper to create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            return f.name

    def test_basic_messages_format(self, basic_config, test_data):
        """Test basic functionality with messages format."""
        config_path = self.create_temp_config(basic_config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_basic",
                input_cols=["task", "example"],
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages"
            )
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            # Check output structure
            assert len(result) == 1
            assert "messages" in result.column_names
            messages = result["messages"][0]
            
            # Should have 2 messages: system + user
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            
            # Check content
            assert "helpful assistant" in messages[0]["content"]
            assert "summarize this text" in messages[1]["content"]
            
        finally:
            import os
            os.unlink(config_path)

    def test_basic_string_format(self, basic_config, test_data):
        """Test basic functionality with string format (legacy)."""
        config_path = self.create_temp_config(basic_config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_string",
                input_cols=["task", "example"],
                output_cols=["prompt"],
                config_path=config_path,
                output_format="string"
            )
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            # Check output structure
            assert len(result) == 1
            assert "prompt" in result.column_names
            prompt = result["prompt"][0]
            
            # Should be a single string
            assert isinstance(prompt, str)
            assert "helpful assistant" in prompt
            assert "summarize this text" in prompt
            
        finally:
            import os
            os.unlink(config_path)

    def test_custom_role_mapping(self, custom_role_mapping_config, test_data):
        """Test custom role mapping."""
        config_path = self.create_temp_config(custom_role_mapping_config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_custom_mapping",
                input_cols=["task", "example"],
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages"
            )
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            messages = result["messages"][0]
            
            # Should have 3 messages: system + assistant + user
            assert len(messages) == 3
            
            # Check roles
            roles = [msg["role"] for msg in messages]
            assert "system" in roles
            assert "assistant" in roles  # principles mapped to assistant
            assert "user" in roles
            
        finally:
            import os
            os.unlink(config_path)

    def test_empty_examples(self, test_data):
        """Test with empty examples field."""
        config = {
            "system": "You are a helpful assistant.",
            "introduction": "Please help with: {{task}}",
            "principles": "Be helpful and accurate.",
            "examples": "",  # Empty
            "generation": "Now help with: {{task}}",
        }
        
        config_path = self.create_temp_config(config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_empty_examples",
                input_cols=["task"],
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages"
            )
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            messages = result["messages"][0]
            
            # Should still work with empty examples
            assert len(messages) == 2  # system + user
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            
        finally:
            import os
            os.unlink(config_path)

    def test_missing_template_variables(self, test_data):
        """Test with missing template variables in sample."""
        config = {
            "system": "You are a helpful assistant.",
            "introduction": "Please help with: {{task}}",
            "principles": "Be helpful and accurate.",
            "examples": "Example: {{example}}",
            "generation": "Now help with: {{task}}",
        }
        
        config_path = self.create_temp_config(config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_missing_vars",
                input_cols=["task"],  # Missing "example"
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages"
            )
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            messages = result["messages"][0]
            
            # Should still work, missing variables will be empty
            assert len(messages) == 2
            assert "{{example}}" not in messages[1]["content"]  # Should be rendered as empty
            
        finally:
            import os
            os.unlink(config_path)

    def test_custom_prompt_struct(self, test_data):
        """Test with custom prompt structure."""
        config = {
            "system": "You are a helpful assistant.",
            "introduction": "Please help with: {{task}}",
            "principles": "Be helpful and accurate.",
            "examples": "Example: {{example}}",
            "generation": "Now help with: {{task}}",
            "prompt_struct": "{system}\n{introduction}\n{generation}",  # Custom structure
        }
        
        config_path = self.create_temp_config(config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_custom_struct",
                input_cols=["task", "example"],
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages"
            )
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            messages = result["messages"][0]
            
            # Should only include variables in custom structure
            assert len(messages) == 2  # system + user (introduction + generation)
            
        finally:
            import os
            os.unlink(config_path)

    def test_model_prompt_override(self, basic_config, test_data):
        """Test with custom model_prompt parameter."""
        config_path = self.create_temp_config(basic_config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_model_prompt",
                input_cols=["task", "example"],
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages",
                model_prompt="custom_prompt"  # Custom model prompt
            )
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            # Should still work with custom model_prompt
            assert len(result) == 1
            assert "messages" in result.column_names
            
        finally:
            import os
            os.unlink(config_path)

    def test_multiple_output_columns(self, basic_config, test_data):
        """Test with multiple output columns."""
        config_path = self.create_temp_config(basic_config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_multiple_outputs",
                input_cols=["task", "example"],
                output_cols=["messages", "prompt_string"],
                config_path=config_path,
                output_format="messages"
            )
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            # Should have both output columns
            assert "messages" in result.column_names
            assert "prompt_string" in result.column_names
            
            # Both should contain the same data
            assert result["messages"][0] == result["prompt_string"][0]
            
        finally:
            import os
            os.unlink(config_path)

    def test_multiple_input_columns(self, basic_config):
        """Test with multiple input columns."""
        config_path = self.create_temp_config(basic_config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_multiple_inputs",
                input_cols=["task", "example", "context"],
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages"
            )
            
            # Test data with all columns
            test_data = [{
                "task": "summarize this text",
                "example": "This is an example",
                "context": "Additional context"
            }]
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            # Should work with multiple input columns
            assert len(result) == 1
            assert "messages" in result.column_names
            
        finally:
            import os
            os.unlink(config_path)

    def test_empty_dataset(self, basic_config):
        """Test with empty dataset."""
        config_path = self.create_temp_config(basic_config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_empty_dataset",
                input_cols=["task", "example"],
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages"
            )
            
            dataset = Dataset.from_list([])
            result = block.generate(dataset)
            
            # Should return empty dataset
            assert len(result) == 0
            
        finally:
            import os
            os.unlink(config_path)

    def test_invalid_output_format(self, basic_config):
        """Test with invalid output format."""
        config_path = self.create_temp_config(basic_config)
        
        try:
            with pytest.raises(ValueError, match="output_format must be either 'messages' or 'string'"):
                PromptFormattingBlock(
                    block_name="test_invalid_format",
                    input_cols=["task", "example"],
                    output_cols=["messages"],
                    config_path=config_path,
                    output_format="invalid_format"
                )
        finally:
            import os
            os.unlink(config_path)

    def test_missing_config_file(self):
        """Test with missing config file."""
        with pytest.raises(FileNotFoundError):
            PromptFormattingBlock(
                block_name="test_missing_config",
                input_cols=["task", "example"],
                output_cols=["messages"],
                config_path="nonexistent_file.yaml",
                output_format="messages"
            )

    def test_invalid_yaml_config(self, test_data):
        """Test with invalid YAML config."""
        # Create invalid YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            with pytest.raises(Exception):  # Should raise some exception for invalid YAML
                PromptFormattingBlock(
                    block_name="test_invalid_yaml",
                    input_cols=["task", "example"],
                    output_cols=["messages"],
                    config_path=config_path,
                    output_format="messages"
                )
        finally:
            import os
            os.unlink(config_path)

    def test_template_rendering_errors(self, test_data):
        """Test handling of template rendering errors."""
        config = {
            "system": "You are a helpful assistant.",
            "introduction": "Please help with: {{task}}",
            "principles": "Be helpful and accurate.",
            "examples": "Example: {{example}}",
            "generation": "Now help with: {{task}}",
        }
        
        config_path = self.create_temp_config(config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_template_errors",
                input_cols=["task", "example"],
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages"
            )
            
            # Test data with malformed template variables
            malformed_data = [{
                "task": "{{unclosed_variable",  # Malformed template
                "example": "This is an example"
            }]
            
            dataset = Dataset.from_list(malformed_data)
            result = block.generate(dataset)
            
            # Should handle template errors gracefully
            assert len(result) == 1
            assert "messages" in result.column_names
            
        finally:
            import os
            os.unlink(config_path)

    def test_batch_processing(self, basic_config):
        """Test batch processing with multiple samples."""
        config_path = self.create_temp_config(basic_config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_batch",
                input_cols=["task", "example"],
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages",
                num_procs=2  # Enable multiprocessing
            )
            
            # Multiple test samples
            test_data = [
                {"task": "summarize text 1", "example": "example 1"},
                {"task": "summarize text 2", "example": "example 2"},
                {"task": "summarize text 3", "example": "example 3"},
            ]
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            # Should process all samples
            assert len(result) == 3
            assert all("messages" in result.column_names for _ in range(3))
            
        finally:
            import os
            os.unlink(config_path)

    def test_backward_compatibility_llmblock_structure(self, test_data):
        """Test backward compatibility with LLMBlock structure."""
        config = {
            "system": "You are a helpful assistant.",
            "introduction": "Please help with: {{task}}",
            "principles": "Be helpful and accurate.",
            "examples": "Example: {{example}}",
            "generation": "Now help with: {{task}}",
            "start_tags": ["[START]"],  # LLMBlock-style tags
            "end_tags": ["[END]"],
        }
        
        config_path = self.create_temp_config(config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_llmblock_compat",
                input_cols=["task", "example"],
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages"
            )
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            # Should work with LLMBlock-style config
            assert len(result) == 1
            assert "messages" in result.column_names
            
        finally:
            import os
            os.unlink(config_path)

    def test_role_mapping_with_empty_content(self, test_data):
        """Test role mapping when some template variables are empty."""
        config = {
            "system": "You are a helpful assistant.",
            "introduction": "",  # Empty
            "principles": "Be helpful and accurate.",
            "examples": "",  # Empty
            "generation": "Now help with: {{task}}",
        }
        
        config_path = self.create_temp_config(config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_empty_content",
                input_cols=["task"],
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages"
            )
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            messages = result["messages"][0]
            
            # Should only include non-empty content
            assert len(messages) == 2  # system + user (principles + generation)
            
        finally:
            import os
            os.unlink(config_path)

    def test_all_empty_content_fallback(self, test_data):
        """Test fallback when all content is empty."""
        config = {
            "system": "",
            "introduction": "",
            "principles": "",
            "examples": "",
            "generation": "",
        }
        
        config_path = self.create_temp_config(config)
        
        try:
            block = PromptFormattingBlock(
                block_name="test_all_empty",
                input_cols=["task"],
                output_cols=["messages"],
                config_path=config_path,
                output_format="messages"
            )
            
            dataset = Dataset.from_list(test_data)
            result = block.generate(dataset)
            
            messages = result["messages"][0]
            
            # Should fall back to legacy string format
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            
        finally:
            import os
            os.unlink(config_path) 