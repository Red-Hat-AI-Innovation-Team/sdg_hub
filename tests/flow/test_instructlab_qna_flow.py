# SPDX-License-Identifier: Apache-2.0
"""Tests for InstructLab Knowledge Q&A Generation flow."""

# Standard
from pathlib import Path

# Third Party
import yaml

FLOW_DIR = Path(__file__).resolve().parents[2] / (
    "src/sdg_hub/flows/knowledge_infusion/instructlab_qna"
)
FLOW_YAML = FLOW_DIR / "flow.yaml"


class TestInstructLabQnAFlowYaml:
    """Test InstructLab Q&A flow YAML structure and metadata."""

    def setup_method(self):
        """Load the flow config once per test."""
        with open(FLOW_YAML, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def test_flow_yaml_exists(self):
        """Flow YAML file should exist."""
        assert FLOW_YAML.exists()

    def test_has_metadata(self):
        """Flow must have metadata section."""
        assert "metadata" in self.config
        metadata = self.config["metadata"]
        assert "name" in metadata
        assert "description" in metadata
        assert "version" in metadata
        assert "license" in metadata
        assert metadata["license"] == "Apache-2.0"

    def test_has_dataset_requirements(self):
        """Flow must declare required input columns."""
        reqs = self.config["metadata"]["dataset_requirements"]
        assert "required_columns" in reqs
        required = reqs["required_columns"]
        assert "document_text" in required
        assert "taxonomy_path" in required
        assert "domain" in required

    def test_has_output_columns(self):
        """Flow must declare expected output columns."""
        outputs = self.config["metadata"]["output_columns"]
        assert "qna_yaml" in outputs
        assert "attribution_txt" in outputs
        assert "taxonomy_path" in outputs
        assert "num_examples" in outputs

    def test_has_blocks(self):
        """Flow must have blocks section with entries."""
        assert "blocks" in self.config
        blocks = self.config["blocks"]
        assert isinstance(blocks, list)
        assert len(blocks) > 0

    def test_all_blocks_have_required_fields(self):
        """Every block must have block_type and block_config with block_name."""
        for i, block in enumerate(self.config["blocks"]):
            assert "block_type" in block, f"Block {i} missing block_type"
            assert "block_config" in block, f"Block {i} missing block_config"
            assert "block_name" in block["block_config"], (
                f"Block {i} missing block_name"
            )

    def test_block_names_are_unique(self):
        """All block names must be unique within the flow."""
        names = [b["block_config"]["block_name"] for b in self.config["blocks"]]
        assert len(names) == len(set(names)), f"Duplicate block names: {names}"

    def test_has_recommended_models(self):
        """Flow should declare recommended models."""
        models = self.config["metadata"]["recommended_models"]
        assert "default" in models
        assert "compatible" in models

    def test_has_tags(self):
        """Flow should have relevant tags."""
        tags = self.config["metadata"]["tags"]
        assert "instructlab" in tags
        assert "qa-generation" in tags

    def test_prompt_files_exist(self):
        """All referenced prompt files must exist."""
        prompts_dir = FLOW_DIR / "prompts"
        for block in self.config["blocks"]:
            config = block["block_config"]
            if "prompt_config_path" in config:
                prompt_path = prompts_dir.parent / config["prompt_config_path"]
                assert prompt_path.exists(), (
                    f"Missing prompt file: {config['prompt_config_path']}"
                )

    def test_pipeline_stages(self):
        """Flow should have all four stages: question gen, answer gen, evaluation, formatting."""
        block_names = [b["block_config"]["block_name"] for b in self.config["blocks"]]
        # Stage 1: question generation
        assert "generate_questions" in block_names
        # Stage 2: answer generation
        assert "generate_answers" in block_names
        # Stage 3: faithfulness evaluation
        assert "evaluate_faithfulness" in block_names
        # Stage 3: filter
        assert "filter_unfaithful" in block_names
        # Stage 4: format to qna.yaml
        assert "format_qna_yaml" in block_names

    def test_faithfulness_filter_keeps_yes(self):
        """Faithfulness filter should keep only YES judgments."""
        filter_block = None
        for block in self.config["blocks"]:
            if block["block_config"]["block_name"] == "filter_unfaithful":
                filter_block = block
                break
        assert filter_block is not None
        assert filter_block["block_config"]["filter_value"] == "YES"
        assert filter_block["block_config"]["operation"] == "eq"


class TestInstructLabQnAPrompts:
    """Test prompt template files."""

    def test_generate_questions_prompt(self):
        """Question generation prompt should reference required variables."""
        prompt_path = FLOW_DIR / "prompts" / "generate_questions.yaml"
        content = prompt_path.read_text()
        assert "{{document_text}}" in content
        assert "{{taxonomy_path}}" in content
        assert "{{domain}}" in content
        assert "[QUESTION]" in content
        assert "[END]" in content

    def test_generate_answers_prompt(self):
        """Answer generation prompt should reference question and document."""
        prompt_path = FLOW_DIR / "prompts" / "generate_answers.yaml"
        content = prompt_path.read_text()
        assert "{{question}}" in content
        assert "{{document_text}}" in content
        assert "<answer>" in content

    def test_evaluate_faithfulness_prompt(self):
        """Faithfulness prompt should reference all three inputs."""
        prompt_path = FLOW_DIR / "prompts" / "evaluate_faithfulness.yaml"
        content = prompt_path.read_text()
        assert "{{document_text}}" in content
        assert "{{question}}" in content
        assert "{{answer}}" in content
        assert "[Start of Judgment]" in content
        assert "[End of Judgment]" in content
