# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the flow translation module."""

from unittest.mock import MagicMock, patch

from sdg_hub.core.utils.error_handling import APIConnectionError
from sdg_hub.core.utils.translation import (
    _adapt_flow_yaml,
    _parse_flow_yaml,
    _validate_translation,
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
        ],
    }
    flow_path = tmp_path / "flow.yaml"
    flow_path.write_text(yaml.dump(flow))
    (tmp_path / "my_prompt.yaml").write_text(
        yaml.dump([{"role": "system", "content": "Hello"}])
    )
    return flow_path


@pytest.fixture()
def flow_with_tags(tmp_path):
    """Create a flow YAML with TagParserBlock."""
    flow = {
        "metadata": {"name": "Tag Flow", "id": "tag-flow-1", "version": "1.0.0"},
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


# ---------------------------------------------------------------------------
# _parse_flow_yaml
# ---------------------------------------------------------------------------


class TestParseFlowYaml:
    def test_discovers_prompts_and_tags(self, flow_with_tags):
        prompts, tags = _parse_flow_yaml(flow_with_tags)
        assert len(prompts) == 1
        assert list(prompts.keys())[0].name == "prompt.yaml"
        assert tags == frozenset({"[QUESTION]", "[ANSWER]", "[END]"})

    def test_parent_path_rejected(self, tmp_path):
        """Prompts referenced via ../ are rejected (path traversal guard)."""
        sub = tmp_path / "my_flow"
        sub.mkdir()
        (tmp_path / "shared.yaml").write_text(
            yaml.dump([{"role": "system", "content": "x"}])
        )
        (sub / "local.yaml").write_text(yaml.dump([{"role": "user", "content": "x"}]))
        flow = {
            "metadata": {"name": "X", "id": "x", "version": "1"},
            "blocks": [
                {
                    "block_type": "PromptBuilderBlock",
                    "block_config": {
                        "block_name": "a",
                        "prompt_config_path": "../shared.yaml",
                    },
                },
                {
                    "block_type": "PromptBuilderBlock",
                    "block_config": {
                        "block_name": "b",
                        "prompt_config_path": "local.yaml",
                    },
                },
            ],
        }
        (sub / "flow.yaml").write_text(yaml.dump(flow))
        prompts, _ = _parse_flow_yaml(sub / "flow.yaml")
        assert len(prompts) == 1
        assert "shared.yaml" not in {p.name for p in prompts}


# ---------------------------------------------------------------------------
# _validate_translation
# ---------------------------------------------------------------------------


class TestValidateTranslation:
    def test_valid_translation(self):
        issues = _validate_translation(
            "Translate {{document}} into [Q] format [END]",
            "Traduzca {{document}} al formato [Q] [END]",
            frozenset({"[Q]", "[END]"}),
        )
        assert issues == []

    def test_missing_jinja_var(self):
        issues = _validate_translation(
            "Use {{document}} and {{query}}",
            "Usa {{document}} y",
            frozenset(),
        )
        assert any("Missing Jinja2 variables" in i for i in issues)

    def test_missing_structural_tag(self):
        issues = _validate_translation(
            "Format as [Q] ... [END]",
            "Formatea como ...",
            frozenset({"[Q]", "[END]"}),
        )
        assert any("Missing structural tags" in i for i in issues)


# ---------------------------------------------------------------------------
# _adapt_flow_yaml
# ---------------------------------------------------------------------------


class TestAdaptFlowYaml:
    def test_adapts_metadata_and_prompt_paths(self, simple_flow_yaml, tmp_path):
        out = tmp_path / "out" / "flow.yaml"
        _adapt_flow_yaml(simple_flow_yaml, out, "Spanish", "es", {"my_prompt.yaml"})

        with open(out) as f:
            result = yaml.safe_load(f)

        assert result["metadata"]["name"] == "Test Flow (Spanish)"
        assert result["metadata"]["id"] == "test-flow-1-es"
        assert (
            result["blocks"][0]["block_config"]["prompt_config_path"]
            == "prompts/my_prompt_es.yaml"
        )


# ---------------------------------------------------------------------------
# _llm_call
# ---------------------------------------------------------------------------


class TestLlmCall:
    @patch("sdg_hub.core.utils.translation.litellm")
    def test_returns_response(self, mock_litellm):
        from sdg_hub.core.utils.translation import _llm_call

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "response"
        mock_litellm.completion.return_value = mock_resp

        result = _llm_call(
            [{"role": "user", "content": "hi"}],
            "test/model",
            None,
            None,
            max_tokens=100,
            temperature=0.0,
        )
        assert result == "response"

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_auth_error_raises_api_connection_error(self, mock_litellm):
        from sdg_hub.core.utils.translation import _llm_call
        import litellm

        mock_litellm.AuthenticationError = litellm.AuthenticationError
        mock_litellm.completion.side_effect = litellm.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4"
        )

        with pytest.raises(APIConnectionError, match="Authentication failed"):
            _llm_call(
                [{"role": "user", "content": "hi"}],
                "test/model",
                None,
                None,
                max_tokens=100,
                temperature=0.0,
            )


# ---------------------------------------------------------------------------
# _translate_and_verify
# ---------------------------------------------------------------------------


class TestTranslateAndVerify:
    @patch("sdg_hub.core.utils.translation._verify_translation")
    @patch("sdg_hub.core.utils.translation._translate_text")
    def test_passes_first_attempt(self, mock_translate, mock_verify):
        from sdg_hub.core.utils.translation import _translate_and_verify

        mock_translate.return_value = "Hola mundo"
        mock_verify.return_value = "PASS"

        translated, issues = _translate_and_verify(
            "Hello world",
            "Spanish",
            "test/model",
            None,
            None,
            "test/verifier",
            None,
            None,
            3,
            "test-label",
            structural_tags=frozenset(),
            tag_rule="- No tags",
        )
        assert translated == "Hola mundo"
        assert issues == []
        assert mock_translate.call_count == 1

    @patch("sdg_hub.core.utils.translation._verify_translation")
    @patch("sdg_hub.core.utils.translation._translate_text")
    def test_retries_on_failure(self, mock_translate, mock_verify):
        from sdg_hub.core.utils.translation import _translate_and_verify

        mock_translate.return_value = "Hola mundo"
        mock_verify.side_effect = ["FAIL: bad", "PASS"]

        translated, issues = _translate_and_verify(
            "Hello world",
            "Spanish",
            "test/model",
            None,
            None,
            "test/verifier",
            None,
            None,
            3,
            "test-label",
            structural_tags=frozenset(),
            tag_rule="- No tags",
        )
        assert issues == []
        assert mock_translate.call_count == 2


# ---------------------------------------------------------------------------
# translate_flow (integration test with mocked LLM)
# ---------------------------------------------------------------------------


class TestTranslateFlow:
    @patch("sdg_hub.core.utils.translation.litellm")
    def test_end_to_end(self, mock_litellm, simple_flow_yaml, tmp_path):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Translated content"

        def side_effect(**kwargs):
            if kwargs.get("max_tokens") == 256:
                resp = MagicMock()
                resp.choices = [MagicMock()]
                resp.choices[0].message.content = "PASS"
                return resp
            return mock_response

        mock_litellm.completion.side_effect = side_effect

        from sdg_hub.core.utils.translation import translate_flow

        out = tmp_path / "output"
        sentinel = MagicMock()
        with (
            patch("sdg_hub.core.utils.translation.FlowRegistry") as mock_registry,
            patch("sdg_hub.core.utils.translation.Flow") as mock_flow_cls,
        ):
            mock_registry.get_flow_path_safe.return_value = str(simple_flow_yaml)
            mock_registry.get_flow_path.return_value = None
            mock_flow_cls.from_yaml.return_value = sentinel
            result = translate_flow(
                flow="test-flow-1",
                lang="Spanish",
                lang_code="es",
                translator_model="test/model",
                verifier_model="test/verifier",
                output_dir=str(out),
            )

        assert result is sentinel
        assert (out / "flow.yaml").exists()
        assert (out / "prompts" / "my_prompt_es.yaml").exists()

        with open(out / "flow.yaml") as f:
            flow = yaml.safe_load(f)
        assert flow["metadata"]["id"] == "test-flow-1-es"
