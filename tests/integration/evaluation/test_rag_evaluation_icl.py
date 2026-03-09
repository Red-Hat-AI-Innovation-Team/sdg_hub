# SPDX-License-Identifier: Apache-2.0
"""Tests for the RAG Evaluation ICL flow."""

# Standard
from pathlib import Path

# Third Party
import pytest
import yaml

# First Party
from sdg_hub import Flow, FlowRegistry, FlowValidator


FLOW_DIR = Path(__file__).resolve().parents[3] / "src" / "sdg_hub" / "flows" / "evaluation" / "rag_evaluation_icl"
FLOW_YAML = FLOW_DIR / "flow.yaml"


class TestRagEvaluationIclFlowStructure:
    """Test the RAG Evaluation ICL flow YAML structure and configuration."""

    def test_flow_yaml_exists(self):
        """Test that the flow YAML file exists."""
        assert FLOW_YAML.exists(), f"Flow YAML not found at {FLOW_YAML}"

    def test_flow_loads_successfully(self):
        """Test that the flow loads from YAML without errors."""
        flow = Flow.from_yaml(str(FLOW_YAML))
        assert flow is not None
        assert flow.metadata.name == "RAG Evaluation ICL Dataset Flow"

    def test_flow_metadata(self):
        """Test that flow metadata is complete."""
        flow = Flow.from_yaml(str(FLOW_YAML))
        assert flow.metadata.version == "1.0.0"
        assert flow.metadata.author == "Red Hat AI RAG Contributors"
        assert flow.metadata.license == "Apache-2.0"
        assert "rag-evaluation" in flow.metadata.tags
        assert "icl" in flow.metadata.tags

    def test_flow_block_count(self):
        """Test that the flow has the expected number of blocks."""
        flow = Flow.from_yaml(str(FLOW_YAML))
        assert len(flow.blocks) == 16

    def test_flow_block_names_unique(self):
        """Test that all block names are unique."""
        flow = Flow.from_yaml(str(FLOW_YAML))
        block_names = [b.block_name for b in flow.blocks]
        assert len(block_names) == len(set(block_names)), (
            f"Duplicate block names found: {[n for n in block_names if block_names.count(n) > 1]}"
        )

    def test_flow_block_names(self):
        """Test that the flow contains the expected blocks in order."""
        flow = Flow.from_yaml(str(FLOW_YAML))
        expected_names = [
            "duplicate_to_context",
            "icl_question_prompt",
            "gen_icl_questions",
            "parse_icl_questions",
            "parse_question_tags",
            "answer_prompt",
            "gen_answer",
            "parse_answer",
            "critic_prompt",
            "gen_critic_score",
            "parse_critic_score",
            "filter_ungrounded",
            "extraction_prompt",
            "extract_context",
            "parse_extracted_context",
            "rename_final_columns",
        ]
        actual_names = [b.block_name for b in flow.blocks]
        assert actual_names == expected_names

    def test_dataset_requirements(self):
        """Test that dataset requirements specify all required ICL columns."""
        flow = Flow.from_yaml(str(FLOW_YAML))
        reqs = flow.get_dataset_requirements()
        assert reqs is not None

        expected_columns = [
            "document",
            "document_outline",
            "icl_document",
            "icl_query_1",
            "icl_query_2",
            "icl_query_3",
        ]
        assert reqs.required_columns == expected_columns

    def test_flow_yaml_validates(self):
        """Test that the flow YAML passes structural validation."""
        with open(FLOW_YAML, encoding="utf-8") as f:
            flow_config = yaml.safe_load(f)

        validator = FlowValidator()
        errors = validator.validate_yaml_structure(flow_config)
        assert errors == [], f"Validation errors: {errors}"


class TestRagEvaluationIclPrompts:
    """Test that all prompt YAML files are valid."""

    PROMPTS_DIR = FLOW_DIR / "prompts"
    EXPECTED_PROMPTS = [
        "icl_question_generation.yaml",
        "answer_generation.yaml",
        "groundedness_critic.yaml",
        "context_extraction.yaml",
    ]

    def test_prompts_directory_exists(self):
        """Test that the prompts directory exists."""
        assert self.PROMPTS_DIR.exists()

    @pytest.mark.parametrize("prompt_file", EXPECTED_PROMPTS)
    def test_prompt_file_exists(self, prompt_file):
        """Test that each expected prompt file exists."""
        path = self.PROMPTS_DIR / prompt_file
        assert path.exists(), f"Prompt file not found: {path}"

    @pytest.mark.parametrize("prompt_file", EXPECTED_PROMPTS)
    def test_prompt_file_valid_yaml(self, prompt_file):
        """Test that each prompt file is valid YAML."""
        path = self.PROMPTS_DIR / prompt_file
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, list), f"{prompt_file} should be a list of messages"

    @pytest.mark.parametrize("prompt_file", EXPECTED_PROMPTS)
    def test_prompt_messages_have_role_and_content(self, prompt_file):
        """Test that each message in a prompt has role and content fields."""
        path = self.PROMPTS_DIR / prompt_file
        with open(path, encoding="utf-8") as f:
            messages = yaml.safe_load(f)

        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and "role" in msg:
                assert "content" in msg, (
                    f"{prompt_file} message {i} has 'role' but missing 'content'"
                )

    def test_prompt_last_message_is_user(self):
        """Test that the last message in each prompt has user role."""
        for prompt_file in self.EXPECTED_PROMPTS:
            path = self.PROMPTS_DIR / prompt_file
            with open(path, encoding="utf-8") as f:
                messages = yaml.safe_load(f)

            # Filter to actual message dicts (skip comments)
            actual_messages = [m for m in messages if isinstance(m, dict) and "role" in m]
            assert actual_messages[-1]["role"] == "user", (
                f"{prompt_file}: last message should have role 'user', got '{actual_messages[-1]['role']}'"
            )

    def test_icl_prompt_contains_expected_variables(self):
        """Test that the ICL question generation prompt references all ICL variables."""
        path = self.PROMPTS_DIR / "icl_question_generation.yaml"
        with open(path, encoding="utf-8") as f:
            content = f.read()

        expected_vars = [
            "{{icl_document}}",
            "{{icl_query_1}}",
            "{{icl_query_2}}",
            "{{icl_query_3}}",
            "{{document}}",
            "{{document_outline}}",
        ]
        for var in expected_vars:
            assert var in content, f"ICL prompt missing template variable: {var}"

    def test_icl_prompt_uses_question_tags(self):
        """Test that the ICL prompt instructs the use of [QUESTION]...[END] tags."""
        path = self.PROMPTS_DIR / "icl_question_generation.yaml"
        with open(path, encoding="utf-8") as f:
            content = f.read()

        assert "[QUESTION]" in content
        assert "[END]" in content


class TestRagEvaluationIclFlowDiscovery:
    """Test that the flow is discoverable by FlowRegistry."""

    def test_flow_discoverable(self):
        """Test that the flow is found by FlowRegistry."""
        FlowRegistry._entries.clear()
        FlowRegistry._search_paths.clear()
        FlowRegistry._initialized = False

        flows_dir = str(FLOW_DIR.parent)  # evaluation/
        FlowRegistry.register_search_path(flows_dir)
        FlowRegistry._discover_flows(force_refresh=True)

        flows = FlowRegistry.list_flows()
        flow_names = [f["name"] for f in flows]
        assert "RAG Evaluation ICL Dataset Flow" in flow_names

    def test_flow_path_retrievable(self):
        """Test that the flow path can be retrieved by name."""
        FlowRegistry._entries.clear()
        FlowRegistry._search_paths.clear()
        FlowRegistry._initialized = False

        flows_dir = str(FLOW_DIR.parent)
        FlowRegistry.register_search_path(flows_dir)
        FlowRegistry._discover_flows(force_refresh=True)

        path = FlowRegistry.get_flow_path("RAG Evaluation ICL Dataset Flow")
        assert path is not None
        assert "rag_evaluation_icl" in path
