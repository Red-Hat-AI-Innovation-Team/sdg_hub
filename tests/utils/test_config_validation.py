# Standard
import os
import tempfile

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.utils.config_validation import (
    validate_block_column_requirements,
    validate_jinja_template_variables,
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


class TestJinjaTemplateValidation:
    """Test cases for Jinja template variable validation."""

    def test_no_variables(self):
        """Test template with no Jinja variables."""
        config = {
            "system": "Simple prompt without variables",
            "generation": "Generate text please",
        }
        available_columns = {"col1", "col2", "col3"}

        missing_vars = validate_jinja_template_variables(config, available_columns)
        assert missing_vars == []

    def test_all_variables_available(self):
        """Test template where all variables are available in columns."""
        config = {
            "system": "Use this {{context}} for generation",
            "generation": "Generate {{task_type}} with {{style}}",
            "examples": "Example with {{context}}",
        }
        available_columns = {"context", "task_type", "style", "extra_col"}

        missing_vars = validate_jinja_template_variables(config, available_columns)
        assert missing_vars == []

    def test_some_variables_missing(self):
        """Test template with some missing variables."""
        config = {
            "system": "Use {{context}} and {{missing_var}}",
            "generation": "Generate {{task_type}} with {{another_missing}}",
            "principles": "Follow {{context}} principles",
        }
        available_columns = {"context", "task_type"}

        missing_vars = validate_jinja_template_variables(config, available_columns)
        assert set(missing_vars) == {"missing_var", "another_missing"}

    def test_complex_jinja_syntax(self):
        """Test template with complex Jinja syntax."""
        config = {
            "system": "{% if context %}Use {{context}}{% endif %}",
            "generation": "{{ data | upper }} and {{format_type}}",
            "examples": "Loop: {% for item in items %}{{item}}{% endfor %}",
        }
        available_columns = {"context", "data", "format_type"}

        missing_vars = validate_jinja_template_variables(config, available_columns)
        assert "items" in missing_vars  # items is referenced but not available

    def test_empty_config_fields(self):
        """Test with empty or None config fields."""
        config = {
            "system": "Valid {{var1}}",
            "generation": "",  # empty string
            "examples": None,  # None value
            "principles": "Uses {{var2}}",
        }
        available_columns = {"var1", "var2"}

        missing_vars = validate_jinja_template_variables(config, available_columns)
        assert missing_vars == []

    def test_non_string_fields_ignored(self):
        """Test that non-string fields are ignored."""
        config = {
            "system": "Valid {{var1}}",
            "generation": ["list", "of", "strings"],  # non-string
            "start_tags": ["<output>"],  # non-template field
            "examples": 123,  # non-string
        }
        available_columns = {"var1"}

        missing_vars = validate_jinja_template_variables(config, available_columns)
        assert missing_vars == []


class TestBlockColumnRequirements:
    """Test cases for block column requirements validation."""

    def test_filter_by_value_block_valid(self):
        """Test FilterByValueBlock with valid column."""
        config = {"filter_column": "category", "filter_value": "test"}
        available_columns = {"category", "text", "other"}

        errors = validate_block_column_requirements(
            "test_block", "FilterByValueBlock", config, available_columns
        )
        assert errors == []

    def test_filter_by_value_block_missing_column(self):
        """Test FilterByValueBlock with missing column."""
        config = {"filter_column": "missing_col", "filter_value": "test"}
        available_columns = {"category", "text"}

        errors = validate_block_column_requirements(
            "test_block", "FilterByValueBlock", config, available_columns
        )
        assert len(errors) == 1
        assert "[test_block] Missing filter_column: 'missing_col'" in errors[0]

    def test_selector_block_valid(self):
        """Test SelectorBlock with valid columns."""
        config = {
            "choice_col": "type",
            "choice_map": {"A": "col1", "B": "col2"},
            "output_col": "result",
        }
        available_columns = {"type", "col1", "col2", "other"}

        errors = validate_block_column_requirements(
            "selector", "SelectorBlock", config, available_columns
        )
        assert errors == []

    def test_selector_block_missing_choice_col(self):
        """Test SelectorBlock with missing choice column."""
        config = {
            "choice_col": "missing_type",
            "choice_map": {"A": "col1", "B": "col2"},
        }
        available_columns = {"col1", "col2"}

        errors = validate_block_column_requirements(
            "selector", "SelectorBlock", config, available_columns
        )
        assert "[selector] Missing choice_col: 'missing_type'" in errors[0]

    def test_selector_block_missing_choice_map_columns(self):
        """Test SelectorBlock with missing choice_map columns."""
        config = {"choice_col": "type", "choice_map": {"A": "col1", "B": "missing_col"}}
        available_columns = {"type", "col1"}

        errors = validate_block_column_requirements(
            "selector", "SelectorBlock", config, available_columns
        )
        assert (
            "[selector] choice_map['B'] references missing column: 'missing_col'"
            in errors[0]
        )

    def test_combine_columns_block_valid(self):
        """Test CombineColumnsBlock with valid columns."""
        config = {"columns": ["a", "b", "c"], "output_col": "combined"}
        available_columns = {"a", "b", "c", "other"}

        errors = validate_block_column_requirements(
            "combine", "CombineColumnsBlock", config, available_columns
        )
        assert errors == []

    def test_combine_columns_block_missing_columns(self):
        """Test CombineColumnsBlock with missing columns."""
        config = {"columns": ["a", "missing", "c"]}
        available_columns = {"a", "c"}

        errors = validate_block_column_requirements(
            "combine", "CombineColumnsBlock", config, available_columns
        )
        assert "[combine] Missing column in columns: 'missing'" in errors[0]

    def test_rename_columns_block_valid(self):
        """Test RenameColumns with valid source columns."""
        config = {"columns_map": {"old1": "new1", "old2": "new2"}}
        available_columns = {"old1", "old2", "other"}

        errors = validate_block_column_requirements(
            "rename", "RenameColumns", config, available_columns
        )
        assert errors == []

    def test_rename_columns_block_missing_source(self):
        """Test RenameColumns with missing source columns."""
        config = {"columns_map": {"existing": "new1", "missing": "new2"}}
        available_columns = {"existing", "other"}

        errors = validate_block_column_requirements(
            "rename", "RenameColumns", config, available_columns
        )
        assert "[rename] Missing source column in columns_map: 'missing'" in errors[0]

    def test_conditional_llm_block_valid(self):
        """Test ConditionalLLMBlock with valid selector column."""
        config = {"selector_column_name": "task_type", "config_paths": {}}
        available_columns = {"task_type", "content"}

        errors = validate_block_column_requirements(
            "conditional", "ConditionalLLMBlock", config, available_columns
        )
        assert errors == []

    def test_conditional_llm_block_missing_selector(self):
        """Test ConditionalLLMBlock with missing selector column."""
        config = {"selector_column_name": "missing_type"}
        available_columns = {"content", "other"}

        errors = validate_block_column_requirements(
            "conditional", "ConditionalLLMBlock", config, available_columns
        )
        assert "[conditional] Missing selector_column_name: 'missing_type'" in errors[0]

    def test_unknown_block_type(self):
        """Test unknown block type returns no errors."""
        config = {"some_field": "value"}
        available_columns = {"col1", "col2"}

        errors = validate_block_column_requirements(
            "unknown", "UnknownBlock", config, available_columns
        )
        assert errors == []

    def test_multiple_block_types(self):
        """Test various other block types."""
        test_cases = [
            ("SamplePopulatorBlock", {"column_name": "key_col"}, {"key_col"}, []),
            (
                "SamplePopulatorBlock",
                {"column_name": "missing"},
                {"other"},
                ["Missing column_name: 'missing'"],
            ),
            ("FlattenColumnsBlock", {"var_cols": ["a", "b"]}, {"a", "b"}, []),
            (
                "FlattenColumnsBlock",
                {"var_cols": ["a", "missing"]},
                {"a"},
                ["Missing column in var_cols: 'missing'"],
            ),
            ("SetToMajorityValue", {"col_name": "target"}, {"target"}, []),
            (
                "SetToMajorityValue",
                {"col_name": "missing"},
                {"other"},
                ["Missing col_name: 'missing'"],
            ),
        ]

        for block_type, config, available_cols, expected_error_substrings in test_cases:
            errors = validate_block_column_requirements(
                "test", block_type, config, available_cols
            )
            if expected_error_substrings:
                assert len(errors) >= len(expected_error_substrings)
                for expected_substr in expected_error_substrings:
                    assert any(expected_substr in error for error in errors), (
                        f"Expected '{expected_substr}' in {errors}"
                    )
            else:
                assert errors == [], (
                    f"Expected no errors for {block_type}, got {errors}"
                )
