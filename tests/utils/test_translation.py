# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the flow translation module.

Tests the discovery, extraction, and validation functions.
LLM-dependent functions (_translate_text, _verify_translation) are tested
via mocking.
"""

from unittest.mock import MagicMock, patch

from sdg_hub.core.utils.translation import (
    _adapt_flow_yaml,
    _build_tag_rule,
    _parse_flow_yaml,
    _validate_translation,
)
import litellm
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
# _parse_flow_yaml
# ---------------------------------------------------------------------------


class TestParseFlowYaml:
    def test_discovers_prompts(self, simple_flow_yaml):
        prompts, tags = _parse_flow_yaml(simple_flow_yaml)
        assert len(prompts) == 1
        abs_path = list(prompts.keys())[0]
        assert abs_path.name == "my_prompt.yaml"
        assert prompts[abs_path] == "my_prompt.yaml"
        assert tags == frozenset()

    def test_extracts_tags(self, flow_with_tags):
        prompts, tags = _parse_flow_yaml(flow_with_tags)
        assert len(prompts) == 1
        assert tags == frozenset({"[QUESTION]", "[ANSWER]", "[END]"})

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
        prompts, tags = _parse_flow_yaml(p)
        assert prompts == {}
        assert tags == frozenset()

    def test_parent_path_rejected(self, flow_with_parent_prompts):
        """Prompts referenced via ../ are rejected (path traversal guard)."""
        prompts, _ = _parse_flow_yaml(flow_with_parent_prompts)
        assert len(prompts) == 1
        names = {p.name for p in prompts}
        assert "local_prompt.yaml" in names
        assert "shared_prompt.yaml" not in names

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
        _, tags = _parse_flow_yaml(p)
        assert tags == frozenset({"[REAL]"})


# ---------------------------------------------------------------------------
# _build_tag_rule
# ---------------------------------------------------------------------------


class TestBuildTagRule:
    def test_with_tags(self):
        rule = _build_tag_rule(frozenset({"[Q]", "[END]"}))
        assert "[END]" in rule
        assert "[Q]" in rule
        assert "DO NOT translate parsing/structural tags" in rule

    def test_without_tags(self):
        rule = _build_tag_rule(frozenset())
        assert "no structural parsing tags" in rule


# ---------------------------------------------------------------------------
# _validate_translation
# ---------------------------------------------------------------------------


class TestValidateTranslation:
    def test_all_preserved(self):
        source = "Translate {{document}} into [QUESTION] format [END]"
        translated = "Traduzca {{document}} al formato [QUESTION] [END]"
        issues = _validate_translation(
            source, translated, frozenset({"[QUESTION]", "[END]"})
        )
        assert issues == []

    def test_missing_jinja_var(self):
        source = "Use {{document}} and {{query}}"
        translated = "Usa {{document}} y"
        issues = _validate_translation(source, translated, frozenset())
        assert any("Missing Jinja2 variables" in i for i in issues)

    def test_extra_jinja_var(self):
        source = "Use {{document}}"
        translated = "Usa {{document}} {{extra}}"
        issues = _validate_translation(source, translated, frozenset())
        assert any("Unexpected Jinja2 variables" in i for i in issues)

    def test_missing_structural_tag(self):
        source = "Format as [QUESTION] ... [END]"
        translated = "Formatea como ... "
        issues = _validate_translation(
            source, translated, frozenset({"[QUESTION]", "[END]"})
        )
        assert any("Missing structural tags" in i for i in issues)

    def test_tag_not_in_source_ignored(self):
        """Tags defined in the flow but not present in this prompt are OK."""
        source = "No tags here"
        translated = "Sin etiquetas aqui"
        issues = _validate_translation(
            source, translated, frozenset({"[QUESTION]", "[END]"})
        )
        assert issues == []


# ---------------------------------------------------------------------------
# _adapt_flow_yaml
# ---------------------------------------------------------------------------


class TestAdaptFlowYaml:
    def test_metadata_adapted(self, simple_flow_yaml, tmp_path):
        out = tmp_path / "out" / "flow.yaml"
        _adapt_flow_yaml(simple_flow_yaml, out, "Spanish", "es")

        with open(out) as f:
            result = yaml.safe_load(f)

        meta = result["metadata"]
        assert meta["name"] == "Test Flow (Spanish)"
        assert meta["id"] == "test-flow-1-es"

    def test_prompt_config_path_updated(self, simple_flow_yaml, tmp_path):
        out = tmp_path / "out" / "flow.yaml"
        _adapt_flow_yaml(simple_flow_yaml, out, "French", "fr")

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
        _adapt_flow_yaml(flow_with_parent_prompts, out, "Spanish", "es")

        with open(out) as f:
            result = yaml.safe_load(f)

        paths = [
            b["block_config"]["prompt_config_path"]
            for b in result["blocks"]
            if "prompt_config_path" in b.get("block_config", {})
        ]
        assert "prompts/shared_prompt_es.yaml" in paths
        assert "prompts/local_prompt_es.yaml" in paths
        for p in paths:
            assert p.startswith("prompts/")
            assert "../" not in p


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
        """Full translate_flow with mocked LLM returns a Flow object."""
        self._mock_llm(mock_litellm)

        from sdg_hub.core.utils.translation import translate_flow

        out = tmp_path / "output"
        sentinel = MagicMock()
        with (
            patch("sdg_hub.core.utils.translation.FlowRegistry") as mock_registry,
            patch("sdg_hub.core.utils.translation.Flow") as mock_flow_cls,
        ):
            mock_registry.get_flow_path_safe.return_value = str(simple_flow_yaml)
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
        mock_flow_cls.from_yaml.assert_called_once_with(
            str(out.resolve() / "flow.yaml")
        )
        assert (out / "flow.yaml").exists()
        assert (out / "prompts" / "my_prompt_es.yaml").exists()

        # Verify adapted flow has updated metadata
        with open(out / "flow.yaml") as f:
            flow = yaml.safe_load(f)
        assert flow["metadata"]["name"] == "Test Flow (Spanish)"
        assert flow["metadata"]["id"] == "test-flow-1-es"

        # Verify FlowRegistry was called (register=True by default)
        mock_registry.register_search_path.assert_called_once_with(str(out.resolve()))
        mock_registry._discover_flows.assert_called_once_with(force_refresh=True)

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_register_false_skips_registry(
        self, mock_litellm, simple_flow_yaml, tmp_path
    ):
        """translate_flow(register=False) does not touch FlowRegistry."""
        self._mock_llm(mock_litellm)

        from sdg_hub.core.utils.translation import translate_flow

        out = tmp_path / "output"
        with (
            patch("sdg_hub.core.utils.translation.FlowRegistry") as mock_registry,
            patch("sdg_hub.core.utils.translation.Flow") as mock_flow_cls,
        ):
            mock_registry.get_flow_path_safe.return_value = str(simple_flow_yaml)
            mock_flow_cls.from_yaml.return_value = MagicMock()
            result = translate_flow(
                flow="test-flow-1",
                lang="Spanish",
                lang_code="es",
                translator_model="test/model",
                verifier_model="test/verifier",
                output_dir=str(out),
                register=False,
            )

        assert result is mock_flow_cls.from_yaml.return_value
        mock_registry.register_search_path.assert_not_called()

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_default_output_dir(
        self, mock_litellm, simple_flow_yaml, tmp_path, monkeypatch
    ):
        """When output_dir is None, it defaults to CWD/<parent_name>_<lang_code>."""
        self._mock_llm(mock_litellm)

        from pathlib import Path

        from sdg_hub.core.utils.translation import translate_flow

        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))

        with (
            patch("sdg_hub.core.utils.translation.FlowRegistry") as mock_registry,
            patch("sdg_hub.core.utils.translation.Flow") as mock_flow_cls,
        ):
            mock_registry.get_flow_path_safe.return_value = str(simple_flow_yaml)
            mock_flow_cls.from_yaml.return_value = MagicMock()
            result = translate_flow(
                flow="test-flow-1",
                lang="Spanish",
                lang_code="es",
                translator_model="test/model",
                verifier_model="test/verifier",
                register=False,
            )

        assert result is mock_flow_cls.from_yaml.return_value
        # The parent dir of simple_flow_yaml fixture is the tmp_path itself
        expected_dir = tmp_path / f"{simple_flow_yaml.parent.name}_es"
        assert (expected_dir / "flow.yaml").exists()

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_flow_with_structural_tags(self, mock_litellm, flow_with_tags, tmp_path):
        """translate_flow correctly discovers and uses structural tags."""
        self._mock_llm(mock_litellm)

        from sdg_hub.core.utils.translation import translate_flow

        out = tmp_path / "output"
        with (
            patch("sdg_hub.core.utils.translation.FlowRegistry") as mock_registry,
            patch("sdg_hub.core.utils.translation.Flow") as mock_flow_cls,
        ):
            mock_registry.get_flow_path_safe.return_value = str(flow_with_tags)
            mock_flow_cls.from_yaml.return_value = MagicMock()
            result = translate_flow(
                flow="tag-flow-1",
                lang="Spanish",
                lang_code="es",
                translator_model="test/model",
                verifier_model="test/verifier",
                output_dir=str(out),
                register=False,
            )

        assert result is mock_flow_cls.from_yaml.return_value
        # The system prompt (first message in translation call) should mention tags
        calls = mock_litellm.completion.call_args_list
        sys_msg = calls[0].kwargs["messages"][0]["content"]
        assert "[QUESTION]" in sys_msg
        assert "[ANSWER]" in sys_msg
        assert "[END]" in sys_msg

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_verifier_failure_still_returns_flow(
        self, mock_litellm, simple_flow_yaml, tmp_path
    ):
        """When verifier returns FAIL, a Flow is still returned."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Translated content"

        def side_effect(**kwargs):
            if kwargs.get("max_tokens") == 256:
                resp = MagicMock()
                resp.choices = [MagicMock()]
                resp.choices[0].message.content = "FAIL: missing context"
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
            mock_flow_cls.from_yaml.return_value = sentinel
            result = translate_flow(
                flow="test-flow-1",
                lang="Spanish",
                lang_code="es",
                translator_model="test/model",
                verifier_model="test/verifier",
                output_dir=str(out),
                max_retries=1,
                register=False,
            )

        assert result is sentinel


# ---------------------------------------------------------------------------
# _llm_call
# ---------------------------------------------------------------------------


class TestLlmCall:
    @patch("sdg_hub.core.utils.translation.litellm")
    def test_basic_call(self, mock_litellm):
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
        mock_litellm.completion.assert_called_once()

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_passes_api_credentials(self, mock_litellm):
        from sdg_hub.core.utils.translation import _llm_call

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        mock_litellm.completion.return_value = mock_resp

        _llm_call(
            [{"role": "user", "content": "hi"}],
            "test/model",
            "my-key",
            "https://example.com",
            max_tokens=100,
            temperature=0.0,
        )
        call_kwargs = mock_litellm.completion.call_args.kwargs
        assert call_kwargs["api_key"] == "my-key"
        assert call_kwargs["api_base"] == "https://example.com"

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_auth_error_raises_system_exit(self, mock_litellm):
        from sdg_hub.core.utils.translation import _llm_call

        mock_litellm.AuthenticationError = litellm.AuthenticationError
        mock_litellm.completion.side_effect = litellm.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4"
        )

        with pytest.raises(SystemExit, match="Authentication failed"):
            _llm_call(
                [{"role": "user", "content": "hi"}],
                "test/model",
                None,
                None,
                max_tokens=100,
                temperature=0.0,
            )

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_empty_response(self, mock_litellm):
        from sdg_hub.core.utils.translation import _llm_call

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = ""
        mock_litellm.completion.return_value = mock_resp

        result = _llm_call(
            [{"role": "user", "content": "hi"}],
            "test/model",
            None,
            None,
            max_tokens=100,
            temperature=0.0,
        )
        assert result == ""


# ---------------------------------------------------------------------------
# _translate_text
# ---------------------------------------------------------------------------


class TestTranslateText:
    @patch("sdg_hub.core.utils.translation.litellm")
    def test_basic_translation(self, mock_litellm):
        from sdg_hub.core.utils.translation import _translate_text

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Hola mundo"
        mock_litellm.completion.return_value = mock_resp

        result = _translate_text("Hello world", "Spanish", "test/model")
        assert result == "Hola mundo"

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_custom_tag_rule(self, mock_litellm):
        from sdg_hub.core.utils.translation import _translate_text

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Result"
        mock_litellm.completion.return_value = mock_resp

        _translate_text(
            "Hello",
            "Spanish",
            "test/model",
            tag_rule="- DO NOT translate [TAG]",
        )
        messages = mock_litellm.completion.call_args.kwargs["messages"]
        assert "DO NOT translate [TAG]" in messages[0]["content"]


# ---------------------------------------------------------------------------
# _verify_translation
# ---------------------------------------------------------------------------


class TestVerifyTranslation:
    @patch("sdg_hub.core.utils.translation.litellm")
    def test_pass_verdict(self, mock_litellm):
        from sdg_hub.core.utils.translation import _verify_translation

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "PASS"
        mock_litellm.completion.return_value = mock_resp

        result = _verify_translation(
            "Hello", "Hola", "Spanish", "test/model", frozenset()
        )
        assert result == "PASS"

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_fail_verdict(self, mock_litellm):
        from sdg_hub.core.utils.translation import _verify_translation

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "FAIL: missing instruction"
        mock_litellm.completion.return_value = mock_resp

        result = _verify_translation(
            "Hello", "Hola", "Spanish", "test/model", frozenset()
        )
        assert result == "FAIL: missing instruction"

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_with_structural_tags(self, mock_litellm):
        from sdg_hub.core.utils.translation import _verify_translation

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "PASS"
        mock_litellm.completion.return_value = mock_resp

        _verify_translation(
            "Hello [Q] [END]",
            "Hola [Q] [END]",
            "Spanish",
            "test/model",
            frozenset({"[Q]", "[END]"}),
        )
        sys_msg = mock_litellm.completion.call_args.kwargs["messages"][0]["content"]
        assert "[END]" in sys_msg
        assert "[Q]" in sys_msg

    @patch("sdg_hub.core.utils.translation.litellm")
    def test_empty_response_returns_fail(self, mock_litellm):
        from sdg_hub.core.utils.translation import _verify_translation

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = ""
        mock_litellm.completion.return_value = mock_resp

        result = _verify_translation(
            "Hello", "Hola", "Spanish", "test/model", frozenset()
        )
        assert result.startswith("FAIL")


# ---------------------------------------------------------------------------
# _clean_content
# ---------------------------------------------------------------------------


class TestCleanContent:
    def test_strips_trailing_whitespace(self):
        from sdg_hub.core.utils.translation import _clean_content

        assert _clean_content("hello   \nworld  ") == "hello\nworld"

    def test_preserves_leading_whitespace(self):
        from sdg_hub.core.utils.translation import _clean_content

        assert _clean_content("  hello\n  world") == "  hello\n  world"


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
        assert translated == "Hola mundo"
        assert issues == []
        assert mock_translate.call_count == 2

    @patch("sdg_hub.core.utils.translation._verify_translation")
    @patch("sdg_hub.core.utils.translation._translate_text")
    def test_max_retries_exhausted(self, mock_translate, mock_verify):
        from sdg_hub.core.utils.translation import _translate_and_verify

        mock_translate.return_value = "Bad translation"
        mock_verify.return_value = "FAIL: still bad"

        translated, issues = _translate_and_verify(
            "Hello world",
            "Spanish",
            "test/model",
            None,
            None,
            "test/verifier",
            None,
            None,
            2,
            "test-label",
            structural_tags=frozenset(),
            tag_rule="- No tags",
        )
        assert translated == "Bad translation"
        assert len(issues) > 0
        assert mock_translate.call_count == 2

    @patch("sdg_hub.core.utils.translation._verify_translation")
    @patch("sdg_hub.core.utils.translation._translate_text")
    def test_programmatic_issues_cause_retry(self, mock_translate, mock_verify):
        from sdg_hub.core.utils.translation import _translate_and_verify

        # First attempt: missing Jinja2 variable; second attempt: fixed
        mock_translate.side_effect = ["Missing var", "Fixed {{name}}"]
        mock_verify.side_effect = ["PASS", "PASS"]

        translated, issues = _translate_and_verify(
            "Hello {{name}}",
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
        # First attempt passes verifier but fails programmatic (missing {{name}})
        # Second attempt should pass both
        assert mock_translate.call_count == 2
        assert issues == []


# ---------------------------------------------------------------------------
# _str_representer (YAML block style)
# ---------------------------------------------------------------------------


class TestBlockStyleDumper:
    def test_multiline_string_uses_block_scalar(self):
        from sdg_hub.core.utils.translation import _BlockStyleDumper

        data = [{"role": "system", "content": "Line 1\nLine 2\nLine 3"}]
        output = yaml.dump(data, Dumper=_BlockStyleDumper, default_flow_style=False)
        assert "|" in output

    def test_single_line_string_no_block_scalar(self):
        from sdg_hub.core.utils.translation import _BlockStyleDumper

        data = [{"role": "system", "content": "Just one line"}]
        output = yaml.dump(data, Dumper=_BlockStyleDumper, default_flow_style=False)
        assert "Just one line" in output


# ---------------------------------------------------------------------------
# __init__.py lazy import
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_translate_flow_importable(self):
        from sdg_hub.core.utils import translate_flow

        assert callable(translate_flow)

    def test_unknown_attr_raises(self):
        with pytest.raises(AttributeError, match="no attribute"):
            from sdg_hub.core import utils

            utils.__getattr__("nonexistent_function")
