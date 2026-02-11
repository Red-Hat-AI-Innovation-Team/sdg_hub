# SPDX-License-Identifier: Apache-2.0
"""Tests for TagParserBlock."""

from sdg_hub.core.blocks.parsing import TagParserBlock
import pandas as pd
import pytest


@pytest.fixture
def tag_parser():
    return TagParserBlock(
        block_name="test",
        input_cols="text",
        output_cols=["output"],
        start_tags=["<out>"],
        end_tags=["</out>"],
    )


@pytest.fixture
def multi_col_parser():
    return TagParserBlock(
        block_name="test",
        input_cols="text",
        output_cols=["title", "content"],
        start_tags=["<title>", "<content>"],
        end_tags=["</title>", "</content>"],
    )


class TestExtraction:
    def test_single_match(self, tag_parser):
        df = pd.DataFrame([{"text": "Hello <out>world</out> bye"}])
        result = tag_parser.generate(df)
        assert len(result) == 1
        assert result.iloc[0]["output"] == "world"

    def test_multiple_matches(self, tag_parser):
        df = pd.DataFrame([{"text": "<out>first</out> and <out>second</out>"}])
        result = tag_parser.generate(df)
        assert len(result) == 2
        assert result.iloc[0]["output"] == "first"
        assert result.iloc[1]["output"] == "second"

    def test_multi_column(self, multi_col_parser):
        df = pd.DataFrame([{"text": "<title>T1</title><content>C1</content>"}])
        result = multi_col_parser.generate(df)
        assert len(result) == 1
        assert result.iloc[0]["title"] == "T1"
        assert result.iloc[0]["content"] == "C1"

    def test_no_match(self, tag_parser):
        df = pd.DataFrame([{"text": "no tags here"}])
        result = tag_parser.generate(df)
        assert len(result) == 0

    def test_empty_input(self, tag_parser):
        df = pd.DataFrame([{"text": ""}])
        result = tag_parser.generate(df)
        assert len(result) == 0

    def test_empty_dataset(self, tag_parser):
        df = pd.DataFrame(columns=["text"])
        result = tag_parser.generate(df)
        assert len(result) == 0

    def test_cleanup_tags(self):
        parser = TagParserBlock(
            block_name="test",
            input_cols="text",
            output_cols=["output"],
            start_tags=["<out>"],
            end_tags=["</out>"],
            parser_cleanup_tags=["<br>"],
        )
        df = pd.DataFrame([{"text": "<out>hello<br>world</out>"}])
        result = parser.generate(df)
        assert result.iloc[0]["output"] == "helloworld"

    def test_whitespace_handling(self, tag_parser):
        df = pd.DataFrame([{"text": "<out>  trimmed  </out>"}])
        result = tag_parser.generate(df)
        assert result.iloc[0]["output"] == "trimmed"

    def test_multiline_content(self, tag_parser):
        df = pd.DataFrame([{"text": "<out>line1\nline2</out>"}])
        result = tag_parser.generate(df)
        assert "line1\nline2" in result.iloc[0]["output"]


class TestValidation:
    def test_mismatched_tag_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            TagParserBlock(
                block_name="test",
                input_cols="text",
                output_cols=["out"],
                start_tags=["<a>", "<b>"],
                end_tags=["</a>"],
            )

    def test_tag_count_output_mismatch(self):
        parser = TagParserBlock(
            block_name="test",
            input_cols="text",
            output_cols=["a", "b"],
            start_tags=["<a>"],
            end_tags=["</a>"],
        )
        df = pd.DataFrame([{"text": "test"}])
        with pytest.raises(ValueError, match="must match"):
            parser(df)

    def test_multiple_input_cols_rejected(self):
        parser = TagParserBlock(
            block_name="test",
            input_cols=["a", "b"],
            output_cols=["out"],
            start_tags=["<x>"],
            end_tags=["</x>"],
        )
        df = pd.DataFrame([{"a": "1", "b": "2"}])
        with pytest.raises(ValueError, match="exactly one"):
            parser(df)

    def test_string_tags_normalized(self):
        parser = TagParserBlock(
            block_name="test",
            input_cols="text",
            output_cols=["out"],
            start_tags="<x>",
            end_tags="</x>",
        )
        assert parser.start_tags == ["<x>"]
        assert parser.end_tags == ["</x>"]
