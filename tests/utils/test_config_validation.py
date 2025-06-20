# Standard
import os
import tempfile

# First Party
from sdg_hub.utils.config_validation import (
    validate_prompt_config_schema,
)


class TestConfigValidation:
    """Test cases for configuration validation functions."""

    def test_valid_config(self):
        """Test validation with a valid configuration."""
        config = {
            "system": "Test system prompt",
            "generation": "Test generation prompt",
            "introduction": "Test introduction",
            "principles": "Test principles",
            "examples": "Test examples",
            "start_tags": ["<output>"],
            "end_tags": ["</output>"],
        }

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is True
        assert errors == []

    def test_minimal_valid_config(self):
        """Test validation with minimal valid configuration (only required fields)."""
        config = {
            "system": "Test system prompt",
            "generation": "Test generation prompt",
        }

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is True
        assert errors == []

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        config = {"introduction": "Test introduction"}

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is False
        assert len(errors) == 1
        assert "Missing required fields: ['system', 'generation']" in errors[0]

    def test_missing_one_required_field(self):
        """Test validation with one missing required field."""
        config = {"system": "Test system prompt"}

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is False
        assert len(errors) == 1
        assert "Missing required fields: ['generation']" in errors[0]

    def test_null_required_fields(self):
        """Test validation with null required fields."""
        config = {"system": None, "generation": "Test generation prompt"}

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is False
        assert "Required field 'system' is null" in errors

    def test_empty_required_fields(self):
        """Test validation with empty required fields."""
        config = {
            "system": "",
            "generation": "   ",  # only whitespace
        }

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is False
        assert "Required field 'system' is empty" in errors
        assert "Required field 'generation' is empty" in errors

    def test_non_string_required_fields(self):
        """Test validation with non-string required fields."""
        config = {
            "system": 123,  # number instead of string
            "generation": ["list", "instead", "of", "string"],  # list instead of string
        }

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is False
        assert "Required field 'system' must be a string, got int" in errors
        assert "Required field 'generation' must be a string, got list" in errors

    def test_non_string_optional_fields(self):
        """Test validation with non-string optional string fields."""
        config = {
            "system": "Test system prompt",
            "generation": "Test generation prompt",
            "introduction": 123,  # should be string
            "principles": {"key": "value"},  # should be string
            "examples": True,  # should be string
        }

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is False
        assert "Field 'introduction' must be a string, got int" in errors
        assert "Field 'principles' must be a string, got dict" in errors
        assert "Field 'examples' must be a string, got bool" in errors

    def test_non_list_tag_fields(self):
        """Test validation with non-list tag fields."""
        config = {
            "system": "Test system prompt",
            "generation": "Test generation prompt",
            "start_tags": "should be list",  # should be list
            "end_tags": 123,  # should be list
        }

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is False
        assert "Field 'start_tags' must be a list, got str" in errors
        assert "Field 'end_tags' must be a list, got int" in errors

    def test_non_string_elements_in_tag_lists(self):
        """Test validation with non-string elements in tag lists."""
        config = {
            "system": "Test system prompt",
            "generation": "Test generation prompt",
            "start_tags": ["<output>", 123, None],  # mixed types
            "end_tags": [True, "</output>"],  # mixed types
        }

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is False
        assert "Field 'start_tags[1]' must be a string, got int" in errors
        assert "Field 'start_tags[2]' must be a string, got NoneType" in errors
        assert "Field 'end_tags[0]' must be a string, got bool" in errors

    def test_valid_tags(self):
        """Test validation with valid tag fields."""
        config = {
            "system": "Test system prompt",
            "generation": "Test generation prompt",
            "start_tags": ["<output>", "<response>"],
            "end_tags": ["</output>", "</response>"],
        }

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is True
        assert errors == []

    def test_empty_tag_lists(self):
        """Test validation with empty tag lists."""
        config = {
            "system": "Test system prompt",
            "generation": "Test generation prompt",
            "start_tags": [],
            "end_tags": [],
        }

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is True
        assert errors == []

    def test_null_optional_fields(self):
        """Test validation with null optional fields (should be allowed)."""
        config = {
            "system": "Test system prompt",
            "generation": "Test generation prompt",
            "introduction": None,  # null optional fields should be OK
            "start_tags": None,
            "end_tags": None,
        }

        is_valid, errors = validate_prompt_config_schema(config, "test_config.yaml")
        assert is_valid is True
        assert errors == []
