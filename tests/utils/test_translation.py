# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the flow translation module.

Tests the discovery, extraction, and validation functions.
LLM-dependent functions (translate_text, verify_translation) are tested
via mocking.
"""

from unittest.mock import MagicMock, patch

from sdg_hub.core.utils.translation import (
    _build_translation_system_prompt,
    _compute_output_paths,
    _is_flow_yaml,
    adapt_flow_yaml,
    discover_prompt_yamls,
    extract_structural_tags,
    validate_translation,
)
import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_flow_yaml(tmp_path):
    """Create a minimal flow YAML with one PromptBuilderBlock."""
    flow = {
        "metadata": {
            "name": "Test Flow",
            "id": "test-flow-1",
            "version": "1.0.0",
            "description": "A test flow.",
            "tags": ["test"],
        },
        "blocks": [
            {
                "block_type": "PromptBuilderBlock",
                "block_config": {
                    "block_name": "prompt1",
                    "prompt_config_path": "my_prompt.yaml",
                    "input_cols": ["document"],
                    "output_cols": "prompt",
                },
            },
            {
                "block_type": "LLMChatBlock",
                "block_config": {
                    "block_name": "gen",
                    "input_cols": "prompt",
                    "output_cols": "response",
                },
            },
        ],
    }
    flow_path = tmp_path / "flow.yaml"
    prompt_path = tmp_path / "my_prompt.yaml"
    flow_path.write_text(yaml.dump(flow))
    prompt_path.write_text(yaml.dump([{"role": "system", "content": "Hello"}]))
    return flow_path


@pytest.fixture()
def flow_with_tags(tmp_path):
    """Create a flow YAML with TagParserBlock."""
    flow = {
        "metadata": {
            "name": "Tag Flow",
            "id": "tag-flow-1",
            "version": "1.0.0",
            "tags": ["test"],
        },
        "blocks": [
            {
                "block_type": "PromptBuilderBlock",
                "block_config": {
                    "block_name": "p",
                    "prompt_config_path": "prompt.yaml",
                },
            },
            {
                "block_type": "TagParserBlock",
                "block_config": {
                    "block_name": "parser",
                    "start_tags": ["[QUESTION]", "[ANSWER]"],
                    "end_tags": ["[END]", "[END]"],
                },
            },
        ],
    }
    flow_path = tmp_path / "flow.yaml"
    flow_path.write_text(yaml.dump(flow))
    (tmp_path / "prompt.yaml").write_text(
        yaml.dump([{"role": "user", "content": "test"}])
    )
    return flow_path


@pytest.fixture()
def flow_with_parent_prompts(tmp_path):
    """Create a flow that references prompts via ../ (pre-flat-layout)."""
    parent_prompt = tmp_path / "shared_prompt.yaml"
    parent_prompt.write_text(
        yaml.dump([{"role": "system", "content": "shared prompt"}])
    )

    sub = tmp_path / "my_flow"
    sub.mkdir()
    local_prompt = sub / "local_prompt.yaml"
    local_prompt.write_text(yaml.dump([{"role": "user", "content": "local prompt"}]))
    flow = {
        "metadata": {
            "name": "Parent Ref Flow",
            "id": "parent-ref-1",
            "version": "1.0.0",
            "tags": ["test"],
        },
        "blocks": [
            {
                "block_type": "PromptBuilderBlock",
                "block_config": {
                    "block_name": "shared",
                    "prompt_config_path": "../shared_prompt.yaml",
                },
            },
            {
                "block_type": "PromptBuilderBlock",
                "block_config": {
                    "block_name": "local",
                    "prompt_config_path": "local_prompt.yaml",
                },
            },
        ],
    }
    (sub / "flow.yaml").write_text(yaml.dump(flow))
    return sub / "flow.yaml"


# ---------------------------------------------------------------------------
# _is_flow_yaml
# ---------------------------------------------------------------------------


class TestIsFlowYaml:
    def test_valid_flow(self, simple_flow_yaml):
        assert _is_flow_yaml(simple_flow_yaml) is True

    def test_non_flow_yaml(self, tmp_path):
        p = tmp_path / "not_a_flow.yaml"
        p.write_text(yaml.dump([{"role": "user", "content": "hi"}]))
        assert _is_flow_yaml(p) is False

    def test_missing_file(self, tmp_path):
        assert _is_flow_yaml(tmp_path / "missing.yaml") is False

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("{{{{not yaml")
        assert _is_flow_yaml(p) is False


# ---------------------------------------------------------------------------
# discover_prompt_yamls
# ---------------------------------------------------------------------------


class TestDiscoverPromptYamls:
    def test_finds_prompt_config_path(self, simple_flow_yaml):
        prompts = discover_prompt_yamls(simple_flow_yaml)
        assert len(prompts) == 1
        abs_path = list(prompts.keys())[0]
        assert abs_path.name == "my_prompt.yaml"
        assert prompts[abs_path] == "my_prompt.yaml"

    def test_no_prompt_blocks(self, tmp_path):
        flow = {
            "metadata": {"name": "No prompts", "id": "x", "version": "1"},
            "blocks": [
                {
                    "block_type": "LLMChatBlock",
                    "block_config": {"block_name": "gen"},
                }
            ],
        }
        p = tmp_path / "flow.yaml"
        p.write_text(yaml.dump(flow))
        assert discover_prompt_yamls(p) == {}

    def test_relative_parent_path_resolves(self, flow_with_parent_prompts):
        """Prompts referenced via ../ still resolve to correct abs path."""
        prompts = discover_prompt_yamls(flow_with_parent_prompts)
        assert len(prompts) == 2
        names = {p.name for p in prompts}
        assert "shared_prompt.yaml" in names
        assert "local_prompt.yaml" in names
        # All values should be basenames (flat layout)
        for basename in prompts.values():
            assert "/" not in basename


# ---------------------------------------------------------------------------
# extract_structural_tags
# ---------------------------------------------------------------------------


class TestExtractStructuralTags:
    def test_extracts_tags_from_parser(self, flow_with_tags):
        tags = extract_structural_tags(flow_with_tags)
        assert tags == frozenset({"[QUESTION]", "[ANSWER]", "[END]"})

    def test_no_parser_blocks(self, simple_flow_yaml):
        tags = extract_structural_tags(simple_flow_yaml)
        assert tags == frozenset()

    def test_skips_empty_tags(self, tmp_path):
        flow = {
            "metadata": {"name": "X", "id": "x", "version": "1"},
            "blocks": [
                {
                    "block_type": "TagParserBlock",
                    "block_config": {
                        "block_name": "p",
                        "start_tags": ["", "[REAL]"],
                        "end_tags": ["", ""],
                    },
                }
            ],
        }
        p = tmp_path / "flow.yaml"
        p.write_text(yaml.dump(flow))
        assert extract_structural_tags(p) == frozenset({"[REAL]"})


# ---------------------------------------------------------------------------
# _build_translation_system_prompt
# ---------------------------------------------------------------------------


class TestBuildTranslationSystemPrompt:
    def test_with_tags(self):
        prompt = _build_translation_system_prompt(
            "Spanish", frozenset({"[Q]", "[END]"})
        )
        assert "Spanish" in prompt
        assert "[END]" in prompt
        assert "[Q]" in prompt
        assert "DO NOT translate parsing/structural tags" in prompt

    def test_without_tags(self):
        prompt = _build_translation_system_prompt("French", frozenset())
        assert "French" in prompt
        assert "no structural parsing tags" in prompt


# ---------------------------------------------------------------------------
# validate_translation
# ---------------------------------------------------------------------------


class TestValidateTranslation:
    def test_all_preserved(self):
        source = "Translate {{document}} into [QUESTION] format [END]"
        translated = "Traduzca {{document}} al formato [QUESTION] [END]"
        issues = validate_translation(
            source, translated, frozenset({"[QUESTION]", "[END]"})
        )
        assert issues == []

    def test_missing_jinja_var(self):
        source = "Use {{document}} and {{query}}"
        translated = "Usa {{document}} y"
        issues = validate_translation(source, translated, frozenset())
        assert any("Missing Jinja2 variables" in i for i in issues)

    def test_extra_jinja_var(self):
        source = "Use {{document}}"
        translated = "Usa {{document}} {{extra}}"
        issues = validate_translation(source, translated, frozenset())
        assert any("Unexpected Jinja2 variables" in i for i in issues)

    def test_missing_structural_tag(self):
        source = "Format as [QUESTION] ... [END]"
        translated = "Formatea como ... "
        issues = validate_translation(
            source, translated, frozenset({"[QUESTION]", "[END]"})
        )
        assert any("Missing structural tags" in i for i in issues)

    def test_tag_not_in_source_ignored(self):
        """Tags defined in the flow but not present in this prompt are OK."""
        source = "No tags here"
        translated = "Sin etiquetas aqui"
        issues = validate_translation(
            source, translated, frozenset({"[QUESTION]", "[END]"})
        )
        assert issues == []


# ---------------------------------------------------------------------------
# adapt_flow_yaml
# ---------------------------------------------------------------------------


class TestAdaptFlowYaml:
    def test_metadata_adapted(self, simple_flow_yaml, tmp_path):
        out = tmp_path / "out" / "flow.yaml"
        adapt_flow_yaml(simple_flow_yaml, out, "Spanish", "es")

        with open(out) as f:
            result = yaml.safe_load(f)

        meta = result["metadata"]
        assert meta["name"] == "Test Flow (Spanish)"
        assert meta["id"] == "test-flow-1-es"
        assert meta["description"] == "A test flow in Spanish."
        assert "spanish" in meta["tags"]

    def test_prompt_config_path_updated(self, simple_flow_yaml, tmp_path):
        out = tmp_path / "out" / "flow.yaml"
        adapt_flow_yaml(simple_flow_yaml, out, "French", "fr")

        with open(out) as f:
            result = yaml.safe_load(f)

        prompt_block = result["blocks"][0]
        assert (
            prompt_block["block_config"]["prompt_config_path"]
            == "prompts/my_prompt_fr.yaml"
        )

    def test_parent_path_flattened(self, flow_with_parent_prompts, tmp_path):
        """../shared_prompt.yaml is rewritten to prompts/shared_prompt_es.yaml."""
        out = tmp_path / "out" / "flow.yaml"
        adapt_flow_yaml(flow_with_parent_prompts, out, "Spanish", "es")

        with open(out) as f:
            result = yaml.safe_load(f)

        paths = [
            b["block_config"]["prompt_config_path"]
            for b in result["blocks"]
            if "prompt_config_path" in b.get("block_config", {})
        ]
        assert "prompts/shared_prompt_es.yaml" in paths
        assert "prompts/local_prompt_es.yaml" in paths
        # All paths should point into prompts/ subdir, no ../ prefixes
        for p in paths:
            assert p.startswith("prompts/")
            assert "../" not in p

    def test_dataset_requirements_updated(self, tmp_path):
        flow = {
            "metadata": {
                "name": "Flow",
                "id": "f1",
                "version": "1",
                "tags": [],
                "dataset_requirements": {
                    "description": "Input dataset should contain documents"
                },
            },
            "blocks": [],
        }
        src = tmp_path / "src" / "flow.yaml"
        src.parent.mkdir()
        src.write_text(yaml.dump(flow))

        out = tmp_path / "out" / "flow.yaml"
        adapt_flow_yaml(src, out, "Japanese", "ja")

        with open(out) as f:
            result = yaml.safe_load(f)

        desc = result["metadata"]["dataset_requirements"]["description"]
        assert "Japanese documents" in desc


# ---------------------------------------------------------------------------
# _compute_output_paths
# ---------------------------------------------------------------------------


class TestComputeOutputPaths:
    def test_single_flow(self, simple_flow_yaml, tmp_path):
        prompts = discover_prompt_yamls(simple_flow_yaml)
        out = tmp_path / "output"
        flow_out, prompt_map = _compute_output_paths(
            simple_flow_yaml.resolve(), prompts, out, "es"
        )
        assert flow_out == out / "flow.yaml"
        assert len(prompt_map) == 1
        out_prompt = list(prompt_map.values())[0]
        assert out_prompt.name == "my_prompt_es.yaml"
        assert out_prompt.parent == out / "prompts"


# ---------------------------------------------------------------------------
# translate_flow (integration test with mocked LLM)
# ---------------------------------------------------------------------------


class TestTranslateFlowMocked:
    def _mock_llm(self, mock_litellm):
        """Configure mock_litellm to return translator + verifier responses."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Translated content"

        def side_effect(**kwargs):
            # Verifier calls have max_tokens=256
            if kwargs.get("max_tokens") == 256:
                resp = MagicMock()
                resp.choices = [MagicMock()]
                resp.choices[0].message.content = "PASS"
                return resp
            return mock_response

        mock_litellm.completion.side_effect = side_effect

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_end_to_end(self, mock_litellm, simple_flow_yaml, tmp_path):
        """Full translate_flow with mocked LLM returns no issues."""
        self._mock_llm(mock_litellm)

        from sdg_hub.core.utils.translation import translate_flow

        out = tmp_path / "output"
        # Mock FlowRegistry: get_flow_path returns None (not found) so it
        # falls through to filesystem resolution. Then mock the register
        # calls at the end.
        with patch("sdg_hub.core.flow.registry.FlowRegistry") as mock_registry:
            mock_registry.get_flow_path.return_value = None
            issues = translate_flow(
                flow=str(simple_flow_yaml),
                lang="Spanish",
                lang_code="es",
                translator_model="test/model",
                verifier_model="test/verifier",
                output_dir=str(out),
            )

        assert issues == []
        assert (out / "flow.yaml").exists()
        assert (out / "prompts" / "my_prompt_es.yaml").exists()

        # Verify adapted flow has updated metadata
        with open(out / "flow.yaml") as f:
            flow = yaml.safe_load(f)
        assert flow["metadata"]["name"] == "Test Flow (Spanish)"
        assert flow["metadata"]["id"] == "test-flow-1-es"

        # Verify FlowRegistry was called (register=True by default)
        mock_registry.register_search_path.assert_called_once_with(str(out.resolve()))
        mock_registry.discover_flows.assert_called_once()

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_register_false_skips_registry(
        self, mock_litellm, simple_flow_yaml, tmp_path
    ):
        """translate_flow(register=False) does not touch FlowRegistry."""
        self._mock_llm(mock_litellm)

        from sdg_hub.core.utils.translation import translate_flow

        out = tmp_path / "output"
        with patch("sdg_hub.core.flow.registry.FlowRegistry") as mock_registry:
            mock_registry.get_flow_path.return_value = None
            issues = translate_flow(
                flow=str(simple_flow_yaml),
                lang="Spanish",
                lang_code="es",
                translator_model="test/model",
                verifier_model="test/verifier",
                output_dir=str(out),
                register=False,
            )

        assert issues == []
        mock_registry.register_search_path.assert_not_called()
