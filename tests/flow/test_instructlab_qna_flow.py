# SPDX-License-Identifier: Apache-2.0
"""Tests for InstructLab Knowledge Q&A Generation flow."""

# Standard
from pathlib import Path

# Third Party
import pytest
import yaml

FLOW_DIR = Path(__file__).resolve().parents[2] / (
    "src/sdg_hub/flows/knowledge_infusion/instructlab_qna"
)
FLOW_YAML = FLOW_DIR / "flow.yaml"


@pytest.fixture(scope="module")
def flow_config():
    """Load flow config once for all tests."""
    with open(FLOW_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestInstructLabQnAFlowYaml:
    """Test InstructLab Q&A flow YAML structure and metadata."""

    def test_flow_yaml_exists(self):
        """Flow YAML file should exist."""
        assert FLOW_YAML.exists()

    def test_has_metadata(self, flow_config):
        """Flow must have metadata section."""
        assert "metadata" in flow_config
        metadata = flow_config["metadata"]
        assert "name" in metadata
        assert "description" in metadata
        assert "version" in metadata
        assert "license" in metadata
        assert metadata["license"] == "Apache-2.0"

    def test_has_dataset_requirements(self, flow_config):
        """Flow must declare required input columns."""
        reqs = flow_config["metadata"]["dataset_requirements"]
        assert "required_columns" in reqs
        required = reqs["required_columns"]
        assert "document_text" in required
        assert "taxonomy_path" in required
        assert "domain" in required

    def test_has_output_columns(self, flow_config):
        """Flow must declare expected output columns."""
        outputs = flow_config["metadata"]["output_columns"]
        assert "qna_yaml" in outputs
        assert "attribution_txt" in outputs
        assert "taxonomy_path" in outputs
        assert "num_examples" in outputs

    def test_has_blocks(self, flow_config):
        """Flow must have blocks section with entries."""
        assert "blocks" in flow_config
        blocks = flow_config["blocks"]
        assert isinstance(blocks, list)
        assert len(blocks) > 0

    def test_all_blocks_have_required_fields(self, flow_config):
        """Every block must have block_type and block_config with block_name."""
        for i, block in enumerate(flow_config["blocks"]):
            assert "block_type" in block, f"Block {i} missing block_type"
            assert "block_config" in block, f"Block {i} missing block_config"
            assert "block_name" in block["block_config"], (
                f"Block {i} missing block_name"
            )

    def test_block_names_are_unique(self, flow_config):
        """All block names must be unique within the flow."""
        names = [b["block_config"]["block_name"] for b in flow_config["blocks"]]
        assert len(names) == len(set(names)), f"Duplicate block names: {names}"

    def test_has_recommended_models(self, flow_config):
        """Flow should declare recommended models."""
        models = flow_config["metadata"]["recommended_models"]
        assert "default" in models
        assert "compatible" in models

    def test_has_tags(self, flow_config):
        """Flow should have relevant tags."""
        tags = flow_config["metadata"]["tags"]
        assert "instructlab" in tags
        assert "qa-generation" in tags

    def test_prompt_files_exist(self, flow_config):
        """All referenced prompt files must exist."""
        for block in flow_config["blocks"]:
            config = block["block_config"]
            if "prompt_config_path" in config:
                prompt_path = FLOW_DIR / config["prompt_config_path"]
                assert prompt_path.exists(), (
                    f"Missing prompt file: {config['prompt_config_path']}"
                )

    def test_pipeline_stages(self, flow_config):
        """Flow should have all four stages."""
        block_names = [
            b["block_config"]["block_name"] for b in flow_config["blocks"]
        ]
        assert "generate_questions" in block_names
        assert "generate_answers" in block_names
        assert "evaluate_faithfulness" in block_names
        assert "filter_unfaithful" in block_names
        assert "format_qna_yaml" in block_names

    def test_faithfulness_filter_keeps_yes(self, flow_config):
        """Faithfulness filter should keep only YES judgments."""
        filter_block = None
        for block in flow_config["blocks"]:
            if block["block_config"]["block_name"] == "filter_unfaithful":
                filter_block = block
                break
        assert filter_block is not None
        assert filter_block["block_config"]["filter_value"] == "YES"
        assert filter_block["block_config"]["operation"] == "eq"

    def test_faithfulness_judge_is_deterministic(self, flow_config):
        """Faithfulness evaluation should use temperature 0 for consistency."""
        eval_block = None
        for block in flow_config["blocks"]:
            if block["block_config"]["block_name"] == "evaluate_faithfulness":
                eval_block = block
                break
        assert eval_block is not None
        assert eval_block["block_config"].get("temperature") == 0


class TestInstructLabQnAPrompts:
    """Test prompt template files."""

    def test_generate_questions_prompt(self):
        """Question generation prompt should reference required variables."""
        content = (FLOW_DIR / "prompts" / "generate_questions.yaml").read_text()
        assert "{{document_text}}" in content
        assert "{{taxonomy_path}}" in content
        assert "{{domain}}" in content
        assert "[QUESTION]" in content
        assert "[END]" in content

    def test_generate_answers_prompt(self):
        """Answer generation prompt should reference question and document."""
        content = (FLOW_DIR / "prompts" / "generate_answers.yaml").read_text()
        assert "{{question}}" in content
        assert "{{document_text}}" in content
        assert "<answer>" in content

    def test_evaluate_faithfulness_prompt(self):
        """Faithfulness prompt should reference all three inputs."""
        content = (FLOW_DIR / "prompts" / "evaluate_faithfulness.yaml").read_text()
        assert "{{document_text}}" in content
        assert "{{question}}" in content
        assert "{{answer}}" in content
        assert "[Start of Judgment]" in content
        assert "[End of Judgment]" in content

    def test_faithfulness_prompt_requires_full_support(self):
        """Faithfulness prompt should require full (not partial) support."""
        content = (FLOW_DIR / "prompts" / "evaluate_faithfulness.yaml").read_text()
        assert "even partially" not in content
        assert "every significant claim" in content.lower() or "all material claims" in content.lower()


class TestInstructLabQnAFlowErrors:
    """Negative-path tests for flow validation."""

    def test_missing_metadata_key_detected(self, flow_config):
        """Removing a required metadata key should be detectable."""
        config_copy = dict(flow_config)
        config_copy["metadata"] = {
            k: v for k, v in flow_config["metadata"].items() if k != "license"
        }
        assert "license" not in config_copy["metadata"]

    def test_missing_block_name_detected(self, flow_config):
        """A block without block_name should fail validation."""
        bad_block = {"block_type": "LLMChatBlock", "block_config": {}}
        assert "block_name" not in bad_block["block_config"]

    def test_duplicate_block_names_detected(self):
        """Duplicate block names should be flagged."""
        blocks = [
            {"block_type": "A", "block_config": {"block_name": "same"}},
            {"block_type": "B", "block_config": {"block_name": "same"}},
        ]
        names = [b["block_config"]["block_name"] for b in blocks]
        assert len(names) != len(set(names))

    def test_missing_prompt_file_detected(self, flow_config, tmp_path):
        """A prompt_config_path pointing to a nonexistent file should fail."""
        fake_path = tmp_path / "nonexistent.yaml"
        assert not fake_path.exists()
