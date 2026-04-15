# SPDX-License-Identifier: Apache-2.0
"""Tests for the RAG Evaluation ICL flow."""

# Standard
from pathlib import Path

# Third Party
import pytest
import yaml

# First Party
from sdg_hub import Flow, FlowRegistry, FlowValidator


def _find_repo_root() -> Path:
    """Find repository root by locating pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root")


FLOW_DIR = (
    _find_repo_root()
    / "src"
    / "sdg_hub"
    / "flows"
    / "evaluation"
    / "rag_evaluation_icl"
)
FLOW_YAML = FLOW_DIR / "flow.yaml"

EXPECTED_BLOCK_NAMES = [
    "duplicate_to_context",
    "topic_prompt",
    "gen_topic",
    "parse_topic",
    "rename_topic",
    "conceptual_prompt",
    "gen_conceptual_question",
    "parse_question",
    "evolution_prompt",
    "evolve_question",
    "parse_evolved_question",
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


@pytest.fixture()
def clean_registry():
    """Reset FlowRegistry to clean state for test isolation."""
    original_entries = FlowRegistry._entries.copy()
    original_paths = FlowRegistry._search_paths.copy()
    original_init = FlowRegistry._initialized

    FlowRegistry._entries.clear()
    FlowRegistry._search_paths.clear()
    FlowRegistry._initialized = False

    flows_dir = str(FLOW_DIR.parents[1])
    FlowRegistry.register_search_path(flows_dir)
    FlowRegistry._discover_flows(force_refresh=True)

    yield

    FlowRegistry._entries = original_entries
    FlowRegistry._search_paths = original_paths
    FlowRegistry._initialized = original_init


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
        assert len(flow.blocks) == len(EXPECTED_BLOCK_NAMES)

    def test_flow_block_names_unique(self):
        """Test that all block names are unique."""
        flow = Flow.from_yaml(str(FLOW_YAML))
        block_names = [b.block_name for b in flow.blocks]
        assert len(block_names) == len(set(block_names)), (
            f"Duplicate block names found: "
            f"{[n for n in block_names if block_names.count(n) > 1]}"
        )

    def test_flow_block_names(self):
        """Test that the flow contains the expected blocks in order."""
        flow = Flow.from_yaml(str(FLOW_YAML))
        actual_names = [b.block_name for b in flow.blocks]
        assert actual_names == EXPECTED_BLOCK_NAMES

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
        "topic_generation.yaml",
        "conceptual_qa_generation_icl.yaml",
        "question_evolution_icl.yaml",
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
        """Test that each dict message in a prompt has both role and content."""
        path = self.PROMPTS_DIR / prompt_file
        with open(path, encoding="utf-8") as f:
            messages = yaml.safe_load(f)

        for i, msg in enumerate(messages):
            assert isinstance(msg, dict), (
                f"{prompt_file} message {i} must be a dict, got {type(msg).__name__}"
            )
            assert "role" in msg, (
                f"{prompt_file} message {i} is a dict but missing 'role'"
            )
            assert "content" in msg, (
                f"{prompt_file} message {i} has 'role' but missing 'content'"
            )

    @pytest.mark.parametrize("prompt_file", EXPECTED_PROMPTS)
    def test_prompt_last_message_is_user(self, prompt_file):
        """Test that the last message in each prompt has user role."""
        path = self.PROMPTS_DIR / prompt_file
        with open(path, encoding="utf-8") as f:
            messages = yaml.safe_load(f)

        last_msg = messages[-1]
        assert isinstance(last_msg, dict), (
            f"{prompt_file}: last entry must be a dict, got {type(last_msg).__name__}"
        )
        assert last_msg.get("role") == "user", (
            f"{prompt_file}: last message should have role 'user', "
            f"got '{last_msg.get('role')}'"
        )

    def test_conceptual_prompt_contains_icl_variables(self):
        """Test that the conceptual QA prompt references ICL variables."""
        path = self.PROMPTS_DIR / "conceptual_qa_generation_icl.yaml"
        with open(path, encoding="utf-8") as f:
            content = f.read()

        expected_vars = [
            "{{icl_document}}",
            "{{icl_query_1}}",
            "{{icl_query_2}}",
            "{{icl_query_3}}",
            "{{document}}",
            "{{document_outline}}",
            "{{topic}}",
        ]
        for var in expected_vars:
            assert var in content, (
                f"Conceptual QA prompt missing template variable: {var}"
            )


class TestRagEvaluationIclFlowDiscovery:
    """Test that the flow is discoverable by FlowRegistry."""

    def test_flow_discoverable(self, clean_registry):
        """Test that the flow is found by FlowRegistry."""
        flows = FlowRegistry.list_flows()
        flow_names = [f["name"] for f in flows]
        assert "RAG Evaluation ICL Dataset Flow" in flow_names

    def test_flow_path_retrievable(self, clean_registry):
        """Test that the flow path can be retrieved by name."""
        path = FlowRegistry.get_flow_path("RAG Evaluation ICL Dataset Flow")
        assert path is not None
        assert "rag_evaluation_icl" in path
