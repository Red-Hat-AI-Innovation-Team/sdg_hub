import pytest
from datasets import Dataset
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
                "config_path": "tests/fixtures/prompts/test_prompt.yaml"
            },
        },
        {
            "block_type": type("FilterByValueBlock", (), {})(),
            "block_config": {
                "block_name": "filter_1",
                "filter_column": "category"
            },
        },
        {
            "block_type": type("CombineColumnsBlock", (), {})(),
            "block_config": {
                "block_name": "combine_1",
                "columns": ["a", "b"]
            },
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

    dataset = Dataset.from_dict({
        "title": ["test"],
        "category": ["x"],
        "a": [1],
        "b": [2],
    })

    result = flow_with_blocks.validate_flow(dataset)
    assert result.valid
    assert result.errors == []


def test_validate_flow_missing_columns(tmp_path, flow_with_blocks):
    # Create a YAML file with Jinja2 var {{title}} that does not exist in the dataset
    prompt_path = tmp_path / "test_prompt.yaml"
    prompt_path.write_text("""
Hello {{ title }} world!
""")
    flow_with_blocks.chained_blocks[0]["block_config"]["config_path"] = str(prompt_path)

    dataset = Dataset.from_dict({
        "category": ["x"],
        "a": [1],
        # "b" is missing
    })

    result = flow_with_blocks.validate_flow(dataset)
    assert not result.valid
    assert "[llm_1] Missing column for prompt var: 'title'" in result.errors
    assert "[combine_1] CombineColumnsBlock requires column: 'b'" in result.errors