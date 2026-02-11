# SPDX-License-Identifier: Apache-2.0
"""Tests for deprecated TextParserBlock."""

import warnings

from sdg_hub.core.blocks.parsing import TextParserBlock
import pandas as pd
import pytest


class TestDeprecationWarning:
    def test_deprecation_warning_emitted(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            TextParserBlock(
                block_name="test",
                input_cols="text",
                output_cols=["out"],
                start_tags=["<out>"],
                end_tags=["</out>"],
            )


class TestBackwardsCompatibility:
    @pytest.fixture
    def tag_parser(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return TextParserBlock(
                block_name="test",
                input_cols="text",
                output_cols=["output"],
                start_tags=["<out>"],
                end_tags=["</out>"],
            )

    @pytest.fixture
    def regex_parser(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return TextParserBlock(
                block_name="test",
                input_cols="text",
                output_cols=["answer"],
                parsing_pattern=r"Answer: (.*?)(?:\n|$)",
            )

    def test_tag_parsing(self, tag_parser):
        df = pd.DataFrame([{"text": "<out>hello</out>"}])
        result = tag_parser.generate(df)
        assert len(result) == 1
        assert result.iloc[0]["output"] == "hello"

    def test_regex_parsing(self, regex_parser):
        df = pd.DataFrame([{"text": "Answer: hello\n"}])
        result = regex_parser.generate(df)
        assert len(result) == 1
        assert result.iloc[0]["answer"] == "hello"

    def test_cleanup_tags(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            parser = TextParserBlock(
                block_name="test",
                input_cols="text",
                output_cols=["out"],
                parsing_pattern=r"<out>(.*?)</out>",
                parser_cleanup_tags=["<br>"],
            )
        df = pd.DataFrame([{"text": "<out>a<br>b</out>"}])
        result = parser.generate(df)
        assert result.iloc[0]["out"] == "ab"


class TestValidation:
    def test_no_parsing_method(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Requires"):
                TextParserBlock(
                    block_name="test",
                    input_cols="text",
                    output_cols=["out"],
                )

    def test_mismatched_tag_lengths(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="same length"):
                TextParserBlock(
                    block_name="test",
                    input_cols="text",
                    output_cols=["out"],
                    start_tags=["<a>", "<b>"],
                    end_tags=["</a>"],
                )


class TestImportPaths:
    def test_import_from_parsing(self):
        from sdg_hub.core.blocks.parsing import TextParserBlock as ParsingBlock

        assert ParsingBlock is TextParserBlock

    def test_import_from_blocks(self):
        from sdg_hub.core.blocks import TextParserBlock as BlocksParser

        assert BlocksParser is TextParserBlock
