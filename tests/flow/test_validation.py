# SPDX-License-Identifier: Apache-2.0
"""Tests for flow validation."""

# Standard

# Third Party

# First Party
from sdg_hub import FlowValidator


class TestFlowValidator:
    """Test FlowValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = FlowValidator()

    def test_validate_yaml_structure_valid(self):
        """Test validation of valid YAML structure."""
        flow_config = {
            "metadata": {"name": "Test Flow", "description": "A test flow"},
            "blocks": [
                {
                    "block_type": "LLMChatBlock",
                    "block_config": {
                        "block_name": "test_block",
                        "input_cols": "input",
                        "output_cols": "output",
                    },
                }
            ],
        }

        errors = self.validator.validate_yaml_structure(flow_config)
        assert errors == []

    def test_validate_yaml_structure_missing_blocks(self):
        """Test validation with missing blocks section."""
        flow_config = {"metadata": {"name": "Test Flow"}}

        errors = self.validator.validate_yaml_structure(flow_config)
        assert len(errors) == 1
        assert "must contain 'blocks' section" in errors[0]

    def test_validate_yaml_structure_empty_blocks(self):
        """Test validation with empty blocks list."""
        flow_config = {"metadata": {"name": "Test Flow"}, "blocks": []}

        errors = self.validator.validate_yaml_structure(flow_config)
        assert len(errors) == 1
        assert "must contain at least one block" in errors[0]

    def test_validate_yaml_structure_invalid_blocks_type(self):
        """Test validation with invalid blocks type."""
        flow_config = {"metadata": {"name": "Test Flow"}, "blocks": "not a list"}

        errors = self.validator.validate_yaml_structure(flow_config)
        assert len(errors) == 1
        assert "'blocks' must be a list" in errors[0]

    def test_validate_block_config_valid(self):
        """Test validation of valid block configuration."""
        block_config = {
            "block_type": "LLMChatBlock",
            "block_config": {
                "block_name": "test_block",
                "input_cols": "input",
                "output_cols": "output",
            },
        }

        errors = self.validator._validate_block_config(block_config, 0)
        assert errors == []

    def test_validate_block_config_missing_block_type(self):
        """Test validation with missing block_type."""
        block_config = {"block_config": {"block_name": "test_block"}}

        errors = self.validator._validate_block_config(block_config, 0)
        assert len(errors) == 1
        assert "Missing required field 'block_type'" in errors[0]

    def test_validate_block_config_missing_block_config(self):
        """Test validation with missing block_config."""
        block_config = {"block_type": "LLMChatBlock"}

        errors = self.validator._validate_block_config(block_config, 0)
        assert len(errors) == 1
        assert "Missing required field 'block_config'" in errors[0]

    def test_validate_block_config_invalid_block_config_type(self):
        """Test validation with invalid block_config type."""
        block_config = {"block_type": "LLMChatBlock", "block_config": "not a dict"}

        errors = self.validator._validate_block_config(block_config, 0)
        assert len(errors) == 1
        assert "'block_config' must be a dictionary" in errors[0]

    def test_validate_block_config_missing_block_name(self):
        """Test validation with missing block_name."""
        block_config = {
            "block_type": "LLMChatBlock",
            "block_config": {"input_cols": "input"},
        }

        errors = self.validator._validate_block_config(block_config, 0)
        assert len(errors) == 1
        assert "must contain 'block_name'" in errors[0]

    def test_validate_block_config_invalid_runtime_overrides(self):
        """Test validation with invalid runtime_overrides."""
        block_config = {
            "block_type": "LLMChatBlock",
            "block_config": {"block_name": "test_block"},
            "runtime_overrides": "not a list",
        }

        errors = self.validator._validate_block_config(block_config, 0)
        assert len(errors) == 1
        assert "'runtime_overrides' must be a list" in errors[0]

    def test_validate_block_config_invalid_runtime_overrides_items(self):
        """Test validation with invalid runtime_overrides items."""
        block_config = {
            "block_type": "LLMChatBlock",
            "block_config": {"block_name": "test_block"},
            "runtime_overrides": [123, "valid", None],
        }

        errors = self.validator._validate_block_config(block_config, 0)
        assert len(errors) == 1
        assert "All 'runtime_overrides' items must be strings" in errors[0]

    def test_validate_metadata_config_valid(self):
        """Test validation of valid metadata configuration."""
        metadata = {
            "name": "Test Flow",
            "description": "A test flow",
            "version": "1.0.0",
            "author": "Test Author",
            "tags": ["test", "example"],
        }

        errors = self.validator._validate_metadata_config(metadata)
        assert errors == []

    def test_validate_metadata_config_missing_name(self):
        """Test validation with missing name."""
        metadata = {"description": "A test flow"}

        errors = self.validator._validate_metadata_config(metadata)
        assert len(errors) == 1
        assert "must contain 'name' field" in errors[0]

    def test_validate_metadata_config_empty_name(self):
        """Test validation with empty name."""
        metadata = {"name": ""}

        errors = self.validator._validate_metadata_config(metadata)
        assert len(errors) == 1
        assert "'name' must be a non-empty string" in errors[0]

    def test_validate_metadata_config_invalid_types(self):
        """Test validation with invalid field types."""
        metadata = {
            "name": "Test Flow",
            "description": 123,  # Should be string
            "version": ["1.0.0"],  # Should be string
            "tags": "not a list",  # Should be list
        }

        errors = self.validator._validate_metadata_config(metadata)
        assert len(errors) == 3
        assert any("'description' must be a string" in error for error in errors)
        assert any("'version' must be a string" in error for error in errors)
        assert any("'tags' must be a list" in error for error in errors)

    def test_validate_metadata_config_invalid_tags(self):
        """Test validation with invalid tags."""
        metadata = {"name": "Test Flow", "tags": ["valid", 123, "also valid"]}

        errors = self.validator._validate_metadata_config(metadata)
        assert len(errors) == 1
        assert "All metadata 'tags' must be strings" in errors[0]

    def test_validate_parameters_config_valid(self):
        """Test validation of valid parameters configuration."""
        parameters = {
            "param1": {
                "default": "value1",
                "description": "First parameter",
                "required": False,
            },
            "param2": {
                "default": 42,
                "description": "Second parameter",
                "required": True,
            },
        }

        errors = self.validator._validate_parameters_config(parameters)
        assert errors == []

    def test_validate_parameters_config_missing_default(self):
        """Test validation with missing default value."""
        parameters = {"param1": {"description": "Parameter without default"}}

        errors = self.validator._validate_parameters_config(parameters)
        assert len(errors) == 1
        assert "must have 'default' value" in errors[0]

    def test_validate_parameters_config_invalid_types(self):
        """Test validation with invalid parameter types."""
        parameters = {
            "param1": {
                "default": "value",
                "description": 123,  # Should be string
                "required": "yes",  # Should be boolean
            }
        }

        errors = self.validator._validate_parameters_config(parameters)
        assert len(errors) == 2
        assert any("description must be a string" in error for error in errors)
        assert any("required field must be boolean" in error for error in errors)

    def test_validate_parameters_config_invalid_name(self):
        """Test validation with invalid parameter name."""
        parameters = {
            123: {  # Invalid name type
                "default": "value"
            }
        }

        errors = self.validator._validate_parameters_config(parameters)
        assert len(errors) == 1
        assert "Parameter names must be strings" in errors[0]

    def test_validate_block_chain_empty(self):
        """Test validation of empty block chain."""
        errors = self.validator.validate_block_chain([])
        assert len(errors) == 1
        assert "Block chain is empty" in errors[0]

    def test_validate_block_chain_valid(self):
        """Test validation of valid block chain."""

        class MockBlock:
            def __init__(self, name):
                self.block_name = name

        blocks = [MockBlock("block1"), MockBlock("block2")]
        errors = self.validator.validate_block_chain(blocks)
        assert errors == []

    def test_validate_block_chain_duplicate_names(self):
        """Test validation with duplicate block names."""

        class MockBlock:
            def __init__(self, name):
                self.block_name = name

        blocks = [MockBlock("block1"), MockBlock("block2"), MockBlock("block1")]
        errors = self.validator.validate_block_chain(blocks)
        assert len(errors) == 1
        assert "Duplicate block name 'block1'" in errors[0]

    def test_validate_block_chain_missing_name(self):
        """Test validation with block missing name."""

        class MockBlockWithoutName:
            pass

        class MockBlock:
            def __init__(self, name):
                self.block_name = name

        blocks = [MockBlock("block1"), MockBlockWithoutName()]
        errors = self.validator.validate_block_chain(blocks)
        assert len(errors) == 1
        assert "missing 'block_name' attribute" in errors[0]

    def test_complete_flow_validation(self):
        """Test complete flow validation with all sections."""
        flow_config = {
            "metadata": {
                "name": "Complete Flow",
                "description": "A complete test flow",
                "version": "1.0.0",
                "author": "Test Author",
                "tags": ["test", "complete"],
            },
            "parameters": {
                "param1": {
                    "default": "value1",
                    "description": "Test parameter",
                    "required": False,
                }
            },
            "blocks": [
                {
                    "block_type": "LLMChatBlock",
                    "block_config": {
                        "block_name": "test_block",
                        "input_cols": "input",
                        "output_cols": "output",
                    },
                }
            ],
        }

        errors = self.validator.validate_yaml_structure(flow_config)
        assert errors == []

    def test_multiple_validation_errors(self):
        """Test that multiple validation errors are collected."""
        flow_config = {
            "metadata": {
                # Missing name
                "description": 123,  # Invalid type
                "tags": "not a list",  # Invalid type
            },
            "parameters": {
                "param1": {
                    # Missing default
                    "description": 456,  # Invalid type
                    "required": "yes",  # Invalid type
                }
            },
            "blocks": [
                {
                    # Missing block_type
                    "block_config": "not a dict"  # Invalid type
                }
            ],
        }

        errors = self.validator.validate_yaml_structure(flow_config)

        # Should collect multiple errors
        assert len(errors) > 1

        # Check that different types of errors are present
        error_text = " ".join(errors)
        assert "name" in error_text
        assert "description" in error_text
        assert "block_type" in error_text
