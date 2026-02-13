# SPDX-License-Identifier: Apache-2.0
"""Tests for RegexParserBlock."""

from sdg_hub.core.blocks.parsing import RegexParserBlock
import pandas as pd
import pytest


@pytest.fixture
def regex_parser():
    return RegexParserBlock(
        block_name="test",
        input_cols="text",
        output_cols=["answer"],
        parsing_pattern=r"Answer: (.*?)(?:\n|$)",
    )


@pytest.fixture
def multi_group_parser():
    return RegexParserBlock(
        block_name="test",
        input_cols="text",
        output_cols=["question", "answer"],
        parsing_pattern=r"Q: (.*?)\nA: (.*?)(?:\n|$)",
    )


class TestExtraction:
    def test_single_match(self, regex_parser):
        df = pd.DataFrame([{"text": "Question\nAnswer: hello world"}])
        result = regex_parser.generate(df)
        assert len(result) == 1
        assert result.iloc[0]["answer"] == "hello world"

    def test_multiple_matches(self, regex_parser):
        df = pd.DataFrame([{"text": "Answer: first\nAnswer: second\n"}])
        result = regex_parser.generate(df)
        assert len(result) == 2
        assert result.iloc[0]["answer"] == "first"
        assert result.iloc[1]["answer"] == "second"

    def test_multi_capture_groups(self, multi_group_parser):
        df = pd.DataFrame([{"text": "Q: What?\nA: Yes\n"}])
        result = multi_group_parser.generate(df)
        assert len(result) == 1
        assert result.iloc[0]["question"] == "What?"
        assert result.iloc[0]["answer"] == "Yes"

    def test_no_match(self, regex_parser):
        df = pd.DataFrame([{"text": "no pattern here"}])
        result = regex_parser.generate(df)
        assert len(result) == 0

    def test_empty_input(self, regex_parser):
        df = pd.DataFrame([{"text": ""}])
        result = regex_parser.generate(df)
        assert len(result) == 0

    def test_empty_dataset(self, regex_parser):
        df = pd.DataFrame(columns=["text"])
        result = regex_parser.generate(df)
        assert len(result) == 0

    def test_cleanup_tags(self):
        parser = RegexParserBlock(
            block_name="test",
            input_cols="text",
            output_cols=["out"],
            parsing_pattern=r"<out>(.*?)</out>",
            parser_cleanup_tags=["<br>"],
        )
        df = pd.DataFrame([{"text": "<out>hello<br>world</out>"}])
        result = parser.generate(df)
        assert result.iloc[0]["out"] == "helloworld"

    def test_multiline_dotall(self):
        parser = RegexParserBlock(
            block_name="test",
            input_cols="text",
            output_cols=["content"],
            parsing_pattern=r"<content>(.*?)</content>",
        )
        df = pd.DataFrame([{"text": "<content>line1\nline2</content>"}])
        result = parser.generate(df)
        assert "line1\nline2" in result.iloc[0]["content"]


class TestEdgeCases:
    def test_list_input(self, regex_parser):
        """List of strings should be parsed and collected as lists."""
        df = pd.DataFrame([{"text": ["Answer: first\n", "Answer: second\n"]}])
        result = regex_parser.generate(df)
        assert len(result) == 1
        assert result.iloc[0]["answer"] == ["first", "second"]

    def test_list_input_with_unparseable_items(self, regex_parser):
        """Unparseable items in list should be skipped."""
        df = pd.DataFrame(
            [{"text": ["Answer: good\n", "no match", "Answer: also good\n"]}]
        )
        result = regex_parser.generate(df)
        assert len(result) == 1
        assert result.iloc[0]["answer"] == ["good", "also good"]

    def test_empty_list_input(self, regex_parser):
        """Empty list should return empty result."""
        df = pd.DataFrame([{"text": []}])
        result = regex_parser.generate(df)
        assert len(result) == 0

    def test_non_string_input(self, regex_parser):
        """Non-string, non-list input should return empty result."""
        df = pd.DataFrame([{"text": {"key": "val"}}])
        result = regex_parser.generate(df)
        assert len(result) == 0


class TestValidation:
    def test_multiple_input_cols_rejected(self):
        parser = RegexParserBlock(
            block_name="test",
            input_cols=["a", "b"],
            output_cols=["out"],
            parsing_pattern=r"(.*)",
        )
        df = pd.DataFrame([{"a": "1", "b": "2"}])
        with pytest.raises(ValueError, match="exactly one"):
            parser(df)
