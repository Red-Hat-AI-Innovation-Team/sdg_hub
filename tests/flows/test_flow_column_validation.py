# Third Party
from datasets import Dataset
import pytest

# First Party
from sdg_hub.flow import Flow
from sdg_hub.utils.validation_result import ValidationResult


@pytest.fixture
def flow_with_blocks():
    flow = Flow(llm_client=None)
    flow.chained_blocks = [
        {
            "block_type": type("LLMBlock", (), {})(),
            "block_config": {
                "block_name": "llm_1",
                "config_path": "tests/fixtures/prompts/test_prompt.yaml",
            },
        },
        {
            "block_type": type("FilterByValueBlock", (), {})(),
            "block_config": {"block_name": "filter_1", "filter_column": "category"},
        },
        {
            "block_type": type("CombineColumnsBlock", (), {})(),
            "block_config": {"block_name": "combine_1", "columns": ["a", "b"]},
        },
    ]
    return flow


def test_validate_flow_success(tmp_path, flow_with_blocks):
    # Create a YAML file with Jinja2 var {{title}} that exists in the dataset
    prompt_path = tmp_path / "test_prompt.yaml"
    prompt_path.write_text("""
Hello {{ title }} world!
""")
    flow_with_blocks.chained_blocks[0]["block_config"]["config_path"] = str(prompt_path)

    dataset = Dataset.from_dict(
        {
            "title": ["test"],
            "category": ["x"],
            "a": [1],
            "b": [2],
        }
    )

    result = flow_with_blocks.validate_dataset_compatibility(dataset)
    assert result.valid
    assert result.errors == []


def test_validate_dataset_compatibility_missing_columns(tmp_path, flow_with_blocks):
    # Create a YAML file with Jinja2 var {{title}} that does not exist in the dataset
    prompt_path = tmp_path / "test_prompt.yaml"
    prompt_path.write_text("""
system: Hello {{ title }} world!
generation: Generate based on {{ title }}
""")
    flow_with_blocks.chained_blocks[0]["block_config"]["config_path"] = str(prompt_path)

    dataset = Dataset.from_dict(
        {
            "category": ["x"],
            "a": [1],
            # "b" and "title" are missing
        }
    )

    result = flow_with_blocks.validate_dataset_compatibility(dataset)
    assert not result.valid
    assert "[llm_1] Missing column for prompt var: 'title'" in result.errors
    assert "[combine_1] Missing column in columns: 'b'" in result.errors


def test_validate_dataset_compatibility_with_yaml_config(tmp_path, flow_with_blocks):
    """Test that the method correctly loads YAML config files."""
    prompt_path = tmp_path / "test_prompt.yaml"
    prompt_path.write_text("""
system: "You are a helpful assistant"
introduction: "Use the {{context}} to answer"
generation: "Based on {{question}}, generate a response"
examples: "For example: {{context}} -> answer"
""")
    flow_with_blocks.chained_blocks[0]["block_config"]["config_path"] = str(prompt_path)

    # Dataset missing 'question' variable from template
    dataset = Dataset.from_dict(
        {
            "context": ["test context"],
            "category": ["x"],
            "a": [1],
            "b": [2],
        }
    )

    result = flow_with_blocks.validate_dataset_compatibility(dataset)
    assert not result.valid
    assert "[llm_1] Missing column for prompt var: 'question'" in result.errors


def test_validate_dataset_compatibility_conditional_llm_block(tmp_path):
    """Test validation for ConditionalLLMBlock."""
    flow = Flow(llm_client=None)

    # Create config files for conditional block
    config1_path = tmp_path / "config1.yaml"
    config1_path.write_text("""
system: "Type A assistant"
generation: "Generate based on {{input_text}}"
""")

    config2_path = tmp_path / "config2.yaml"
    config2_path.write_text("""
system: "Type B assistant"  
generation: "Process {{input_text}} and {{extra_data}}"
""")

    flow.chained_blocks = [
        {
            "block_type": type("ConditionalLLMBlock", (), {})(),
            "block_config": {
                "block_name": "conditional_llm",
                "selector_column_name": "task_type",
                "config_paths": {"A": str(config1_path), "B": str(config2_path)},
            },
        }
    ]

    # Dataset missing selector column and template variables
    dataset = Dataset.from_dict(
        {
            "input_text": ["test"],
            # missing: task_type, extra_data
        }
    )

    result = flow.validate_dataset_compatibility(dataset)
    assert not result.valid
    assert (
        "[conditional_llm] Missing selector_column_name: 'task_type'" in result.errors
    )
    assert (
        "[conditional_llm] Missing column for template var in config_paths['B']: 'extra_data'"
        in result.errors
    )


def test_generate_fails_fast_on_validation_error(tmp_path, flow_with_blocks):
    """Test that generate() fails fast when dataset validation fails."""
    prompt_path = tmp_path / "test_prompt.yaml"
    prompt_path.write_text("""
system: Hello {{ missing_var }} world!
generation: Generate text
""")
    flow_with_blocks.chained_blocks[0]["block_config"]["config_path"] = str(prompt_path)

    dataset = Dataset.from_dict(
        {
            "category": ["x"],
            "a": [1],
            "b": [2],
            # missing: missing_var
        }
    )

    # Should raise ValueError before any block execution
    with pytest.raises(
        ValueError, match="Dataset is not compatible with flow requirements"
    ):
        flow_with_blocks.generate(dataset)


def test_column_tracking_through_flow():
    """Test that columns are tracked correctly as they're added by blocks."""
    flow = Flow(llm_client=None)

    flow.chained_blocks = [
        # Block that adds output_cols
        {
            "block_type": type("LLMBlock", (), {})(),
            "block_config": {
                "block_name": "llm_1",
                "output_cols": ["generated_text", "score"],
            },
        },
        # Block that uses the newly added column
        {
            "block_type": type("FilterByValueBlock", (), {})(),
            "block_config": {
                "block_name": "filter_1",
                "filter_column": "score",  # This should be valid after llm_1 adds it
            },
        },
        # Block that creates single output column
        {
            "block_type": type("CombineColumnsBlock", (), {})(),
            "block_config": {
                "block_name": "combine_1",
                "columns": ["input_text", "generated_text"],  # Both should be available
                "output_col": "combined",
            },
        },
    ]

    dataset = Dataset.from_dict(
        {
            "input_text": ["test"],
        }
    )

    result = flow.validate_dataset_compatibility(dataset)
    assert result.valid
    assert result.errors == []


def test_rename_columns_tracking():
    """Test that RenameColumns updates available column names correctly."""
    flow = Flow(llm_client=None)

    flow.chained_blocks = [
        # Rename columns
        {
            "block_type": type("RenameColumns", (), {})(),
            "block_config": {
                "block_name": "rename_1",
                "columns_map": {"old_name": "new_name"},
            },
        },
        # Use renamed column
        {
            "block_type": type("FilterByValueBlock", (), {})(),
            "block_config": {
                "block_name": "filter_1",
                "filter_column": "new_name",  # Should work with renamed column
            },
        },
    ]

    dataset = Dataset.from_dict({"old_name": ["test"], "other_col": ["data"]})

    result = flow.validate_dataset_compatibility(dataset)
    assert result.valid
    assert result.errors == []


# Update the original test name for backward compatibility
def test_validate_flow_success(tmp_path, flow_with_blocks):
    """Backward compatibility test - redirects to new method."""
    prompt_path = tmp_path / "test_prompt.yaml"
    prompt_path.write_text("""
system: Hello {{ title }} world!
generation: Generate text
""")
    flow_with_blocks.chained_blocks[0]["block_config"]["config_path"] = str(prompt_path)

    dataset = Dataset.from_dict(
        {
            "title": ["test"],
            "category": ["x"],
            "a": [1],
            "b": [2],
        }
    )
