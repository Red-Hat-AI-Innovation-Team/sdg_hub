# Standard
from unittest.mock import patch

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.llm import TextParserBlock
import pytest


@pytest.fixture
def postprocessing_block():
    """Create a basic TextParserBlock instance for testing."""
    return TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<output>"],
        end_tags=["</output>"],
    )


@pytest.fixture
def postprocessing_block_with_custom_parser():
    """Create a TextParserBlock instance with custom parser configuration."""
    return TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        parsing_pattern=r"Answer: (.*?)(?:\n|$)",
        parser_cleanup_tags=["<br>", "</br>"],
    )


@pytest.fixture
def postprocessing_block_with_tags():
    """Create a TextParserBlock instance with tag-based parsing configuration."""
    return TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<output>"],
        end_tags=["</output>"],
    )


@pytest.fixture
def postprocessing_block_multi_column():
    """Create a TextParserBlock instance with multiple output columns."""
    return TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["title", "content"],
        start_tags=["<title>", "<content>"],
        end_tags=["</title>", "</content>"],
    )


def test_extract_matches_no_tags(postprocessing_block):
    """Test extraction when no tags are provided."""
    text = "This is a test text"
    result = postprocessing_block._extract_matches(text, None, None)
    assert result == ["This is a test text"]


def test_extract_matches_with_start_tag(postprocessing_block):
    """Test extraction with only start tag."""
    text = "START This is a test text"
    result = postprocessing_block._extract_matches(text, "START", None)
    assert result == ["This is a test text"]


def test_extract_matches_with_end_tag(postprocessing_block):
    """Test extraction with only end tag."""
    text = "This is a test text END"
    result = postprocessing_block._extract_matches(text, None, "END")
    assert result == ["This is a test text"]


def test_extract_matches_with_both_tags(postprocessing_block):
    """Test extraction with both start and end tags."""
    text = "START This is a test text END"
    result = postprocessing_block._extract_matches(text, "START", "END")
    assert result == ["This is a test text"]


def test_extract_matches_multiple_matches(postprocessing_block):
    """Test extraction with multiple matches."""
    text = "START First text END START Second text END"
    result = postprocessing_block._extract_matches(text, "START", "END")
    assert result == ["First text", "Second text"]


def test_extract_matches_empty_text(postprocessing_block):
    """Test extraction with empty text."""
    result = postprocessing_block._extract_matches("", "START", "END")
    assert result == []


def test_parse_tag_based_single_column(postprocessing_block_with_tags):
    """Test tag-based parsing with single output column."""
    text = "Some text <output>This is the output</output> more text"
    result = postprocessing_block_with_tags._parse(text)
    assert result == {"output": ["This is the output"]}


def test_parse_tag_based_multiple_columns(postprocessing_block_multi_column):
    """Test tag-based parsing with multiple output columns."""
    text = """
    <title>First Title</title>
    <content>First Content</content>
    <title>Second Title</title>
    <content>Second Content</content>
    """
    result = postprocessing_block_multi_column._parse(text)
    assert result == {
        "title": ["First Title", "Second Title"],
        "content": ["First Content", "Second Content"],
    }


def test_parse_custom_regex_single_match(postprocessing_block_with_custom_parser):
    """Test custom regex parser with a single match."""
    text = "Question: What is the answer?\nAnswer: This is the answer"
    result = postprocessing_block_with_custom_parser._parse(text)
    assert result == {"output": ["This is the answer"]}


def test_parse_custom_regex_multiple_matches(postprocessing_block_with_custom_parser):
    """Test custom regex parser with multiple matches."""
    text = "Question 1: What is the answer?\nAnswer: First answer\nQuestion 2: Another question?\nAnswer: Second answer"
    result = postprocessing_block_with_custom_parser._parse(text)
    assert result == {"output": ["First answer", "Second answer"]}


def test_parse_custom_regex_with_cleanup_tags(postprocessing_block_with_custom_parser):
    """Test custom regex parser with cleanup tags."""
    text = "Answer: This is the <br>answer</br>"
    result = postprocessing_block_with_custom_parser._parse(text)
    assert result == {"output": ["This is the answer"]}


def test_parse_empty_input(postprocessing_block):
    """Test parsing with empty input."""
    result = postprocessing_block._parse("")
    assert result == {"output": []}


def test_parse_no_matches(postprocessing_block_with_custom_parser):
    """Test parsing when no matches are found."""
    text = "This text has no matches for the pattern"
    result = postprocessing_block_with_custom_parser._parse(text)
    assert result == {"output": []}


def test_generate_basic_functionality(postprocessing_block):
    """Test basic generate functionality with tag-based parsing."""
    postprocessing_block.start_tags = ["<output>"]
    postprocessing_block.end_tags = ["</output>"]

    data = [
        {"raw_output": "Text <output>Result 1</output> more text"},
        {"raw_output": "Text <output>Result 2</output> more text"},
    ]
    dataset = Dataset.from_list(data)

    result = postprocessing_block.generate(dataset)

    assert len(result) == 2
    assert result[0]["output"] == "Result 1"
    assert result[1]["output"] == "Result 2"


def test_generate_custom_regex(postprocessing_block_with_custom_parser):
    """Test generate functionality with custom regex parsing."""
    data = [
        {"raw_output": "Question: Q1\nAnswer: A1"},
        {"raw_output": "Question: Q2\nAnswer: A2"},
    ]
    dataset = Dataset.from_list(data)

    result = postprocessing_block_with_custom_parser.generate(dataset)

    assert len(result) == 2
    assert result[0]["output"] == "A1"
    assert result[1]["output"] == "A2"


def test_generate_multiple_matches_per_input(postprocessing_block_multi_column):
    """Test generate functionality with multiple matches per input."""
    data = [
        {
            "raw_output": """
            <title>Title 1</title>
            <content>Content 1</content>
            <title>Title 2</title>
            <content>Content 2</content>
            """
        }
    ]
    dataset = Dataset.from_list(data)

    result = postprocessing_block_multi_column.generate(dataset)

    assert len(result) == 2
    assert result[0]["title"] == "Title 1"
    assert result[0]["content"] == "Content 1"
    assert result[1]["title"] == "Title 2"
    assert result[1]["content"] == "Content 2"


def test_generate_missing_input_column(postprocessing_block):
    """Test that missing input column is handled by BaseBlock validation."""
    # First Party
    from sdg_hub.core.utils.error_handling import MissingColumnError

    data = [{"other_column": "some text"}]
    dataset = Dataset.from_list(data)

    # BaseBlock should handle validation and raise MissingColumnError
    with pytest.raises(MissingColumnError):
        postprocessing_block(dataset)  # Use __call__ to trigger validation


def test_generate_empty_dataset(postprocessing_block):
    """Test generate functionality with empty dataset."""
    dataset = Dataset.from_list([])

    result = postprocessing_block.generate(dataset)

    assert len(result) == 0


def test_generate_all_empty_parsed_outputs(postprocessing_block):
    """Test generate functionality when all parsed outputs are empty lists."""
    postprocessing_block.start_tags = ["<output>"]
    postprocessing_block.end_tags = ["</output>"]

    data = [
        {"raw_output": "Text without any tags"},
        {"raw_output": "More text without tags"},
    ]
    dataset = Dataset.from_list(data)

    result = postprocessing_block.generate(dataset)

    # Should not raise ValueError and should return empty dataset
    assert len(result) == 0


def test_generate_all_empty_parsed_outputs_custom_parser(
    postprocessing_block_with_custom_parser,
):
    """Test generate functionality with custom parser when all parsed outputs are empty."""
    data = [
        {"raw_output": "Question: What is the answer?\nNo answer provided"},
        {"raw_output": "Another question without answer"},
    ]
    dataset = Dataset.from_list(data)

    result = postprocessing_block_with_custom_parser.generate(dataset)

    # Should not raise ValueError and should return empty dataset
    assert len(result) == 0


def test_constructor_validation_no_input_cols():
    """Test validation with no input columns during execution."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols=[],
        output_cols=["output"],
        start_tags=["<output>"],
        end_tags=["</output>"],
    )

    # Create test dataset
    test_data = Dataset.from_list([{"test": "value"}])

    # Validation should fail during execution
    with pytest.raises(
        ValueError, match="TextParserBlock expects at least one input column"
    ):
        block(test_data)


def test_constructor_validation_multiple_input_cols():
    """Test constructor validation with multiple input columns (should warn but not error)."""
    # This should not raise an error, just log a warning
    block = TextParserBlock(
        block_name="test_block",
        input_cols=["col1", "col2"],
        output_cols=["output"],
        start_tags=["<output>"],
        end_tags=["</output>"],
    )

    assert len(block.input_cols) == 2
    assert block.input_cols[0] == "col1"


def test_constructor_string_input_cols():
    """Test constructor with string input_cols (should be converted to list)."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<output>"],
        end_tags=["</output>"],
    )

    assert block.input_cols == ["raw_output"]


def test_constructor_string_output_cols():
    """Test constructor with string output_cols (should be converted to list)."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols="output",
        start_tags=["<output>"],
        end_tags=["</output>"],
    )

    assert block.output_cols == ["output"]


def test_parse_uneven_tags():
    """Test parsing with uneven start and end tags."""
    # Creating a block with more start tags than end tags should raise ValidationError
    with pytest.raises(
        Exception, match="start_tags and end_tags must have the same length"
    ):
        TextParserBlock(
            block_name="test_block",
            input_cols="raw_output",
            output_cols=["title", "content", "footer"],
            start_tags=["<title>", "<content>", "<footer>"],
            end_tags=["</title>", "</content>"],
        )


def test_parse_more_output_cols_than_tags():
    """Test parsing when there are more output columns than tag pairs."""
    # Create a block with 3 output columns but only 2 tag pairs
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["title", "content", "footer"],
        start_tags=["<title>", "<content>"],
        end_tags=["</title>", "</content>"],
    )

    text = """
    <title>Header content</title>
    <content>Main content</content>
    """

    result = block._parse(text)
    # All output columns should be present, with footer having empty list
    assert result == {
        "title": ["Header content"],
        "content": ["Main content"],
        "footer": [],
    }


def test_parse_with_whitespace(postprocessing_block_with_tags):
    """Test parsing with various whitespace patterns."""
    text = """
    <output>  Leading and trailing spaces  </output>
    <output>
    Multiple
    Lines
    </output>
    """

    result = postprocessing_block_with_tags._parse(text)
    assert result == {"output": ["Leading and trailing spaces", "Multiple\n    Lines"]}


def test_extract_matches_incomplete_tags(postprocessing_block):
    """Test extraction with incomplete tag pairs."""
    text = "START First text END START Second text"
    result = postprocessing_block._extract_matches(text, "START", "END")
    assert result == ["First text"]


def test_extract_matches_cascading_tags(postprocessing_block):
    """Test extraction with cascading start and end tags."""
    text = "START1 START2 Nested text END2 END1"
    result = postprocessing_block._extract_matches(text, "START1", "END1")
    assert result == ["START2 Nested text END2"]

    result = postprocessing_block._extract_matches(text, "START2", "END2")
    assert result == ["Nested text"]


def test_parse_mixed_tag_types(postprocessing_block_multi_column):
    """Test parsing with mixed tag types (XML-style and custom markers)."""
    # Create a new block with the desired configuration
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["header", "body"],
        start_tags=["<header>", "START"],
        end_tags=["</header>", "END"],
    )

    text = """
    <header>XML Style Header</header>
    START Custom Style Body END
    <header>Another XML Header</header>
    START Another Custom Body END
    """

    result = block._parse(text)
    assert result == {
        "header": ["XML Style Header", "Another XML Header"],
        "body": ["Custom Style Body", "Another Custom Body"],
    }


def test_parse_with_special_characters(postprocessing_block_with_tags):
    """Test parsing with special characters in tags and content."""
    # Create a new block with the desired configuration
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["special"],
        start_tags=["<special>"],
        end_tags=["</special>"],
    )

    text = """
    <special>Content with &amp; entities</special>
    <special>Content with <nested> tags</special>
    <special>Content with "quotes" and 'apostrophes'</special>
    """

    result = block._parse(text)
    assert result == {
        "special": [
            "Content with &amp; entities",
            "Content with <nested> tags",
            "Content with \"quotes\" and 'apostrophes'",
        ]
    }


def test_parse_mismatched_config_tags(postprocessing_block_multi_column):
    """Test parsing with mismatched numbers of start and end tags in configuration."""
    # Test case 1: More start tags than end tags should raise ValidationError
    with pytest.raises(
        Exception, match="start_tags and end_tags must have the same length"
    ):
        TextParserBlock(
            block_name="test_block",
            input_cols="raw_output",
            output_cols=["header", "content", "footer"],
            start_tags=["<header>", "<content>", "<footer>"],
            end_tags=["</header>", "</content>"],
        )

    # Test case 2: More end tags than start tags should also raise ValidationError
    with pytest.raises(
        Exception, match="start_tags and end_tags must have the same length"
    ):
        TextParserBlock(
            block_name="test_block",
            input_cols="raw_output",
            output_cols=["header", "content", "footer"],
            start_tags=["<header>"],
            end_tags=["</header>", "</content>", "</footer>"],
        )


def test_parse_uneven_tags_comprehensive(postprocessing_block_multi_column):
    """Test parsing with uneven or mismatched start and end tags - comprehensive test cases."""
    # Create a new block with the desired configuration
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["section", "subsection"],
        start_tags=["<section>", "<subsection>"],
        end_tags=["</section>", "</subsection>"],
    )

    # Test cases with various uneven tag scenarios
    test_cases = [
        # Missing end tag - parser should not capture content without proper end tag
        (
            """
        <section>First section
        <subsection>First subsection</subsection>
        """,
            {
                "section": [],  # No valid section content due to missing end tag
                "subsection": ["First subsection"],
            },
        ),
        # Extra end tag - parser should ignore extra end tag
        (
            """
        <section>First section</section>
        </section>
        """,
            {"section": ["First section"], "subsection": []},
        ),
        # Nested tags with missing outer end tag
        (
            """
        <section>Outer content
        <subsection>Inner content</subsection>
        """,
            {
                "section": [],  # No valid section content due to missing end tag
                "subsection": ["Inner content"],
            },
        ),
        # Multiple start tags without end tags
        (
            """
        <section>First section
        <section>Second section
        <subsection>First subsection</subsection>
        """,
            {
                "section": [],  # No valid section content due to missing end tags
                "subsection": ["First subsection"],
            },
        ),
    ]

    for text, expected in test_cases:
        result = block._parse(text)
        assert result == expected, f"Failed for text: {text}"


def test_parse_with_whitespace_comprehensive(postprocessing_block_with_tags):
    """Test parsing with various whitespace patterns - comprehensive test."""
    # Create a new block with the desired configuration
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["text"],
        start_tags=["<text>"],
        end_tags=["</text>"],
    )

    text = """
    <text>  Leading and trailing spaces  </text>
    <text>
    Multiple
    Lines
    </text>
    <text>\tTabbed content\t</text>
    """

    result = block._parse(text)
    assert result == {
        "text": ["Leading and trailing spaces", "Multiple\n    Lines", "Tabbed content"]
    }


# New validation tests
def test_validation_no_parsing_method_configured():
    """Test validation failure when no parsing method is configured."""
    with pytest.raises(ValueError, match="at least one parsing method"):
        block = TextParserBlock(
            block_name="test_block",
            input_cols="raw_output",
            output_cols=["output"],
            # No parsing_pattern, start_tags, or end_tags
        )
        test_data = Dataset.from_list([{"raw_output": "test"}])
        block(test_data)


def test_validation_mismatched_tag_lengths():
    """Test validation failure when start_tags and end_tags have different lengths."""
    with pytest.raises(
        ValueError, match="start_tags and end_tags must have the same length"
    ):
        block = TextParserBlock(
            block_name="test_block",
            input_cols="raw_output",
            output_cols=["output"],
            start_tags=["<start1>", "<start2>"],
            end_tags=["<end1>"],  # Missing second end tag
        )
        test_data = Dataset.from_list([{"raw_output": "test"}])
        block(test_data)


def test_validation_tag_pairs_output_cols_mismatch():
    """Test validation failure when tag pairs don't match output columns."""
    with pytest.raises(ValueError, match="number of tag pairs must match output_cols"):
        block = TextParserBlock(
            block_name="test_block",
            input_cols="raw_output",
            output_cols=["col1", "col2", "col3"],  # 3 output columns
            start_tags=["<start1>", "<start2>"],  # Only 2 tag pairs
            end_tags=["<end1>", "<end2>"],
        )
        test_data = Dataset.from_list([{"raw_output": "test"}])
        block(test_data)


def test_validation_regex_only_configuration():
    """Test that regex-only configuration is valid."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        parsing_pattern=r"Answer: (.*)",
        # No tags specified - this should be valid
    )

    data = [{"raw_output": "Answer: test response"}]
    dataset = Dataset.from_list(data)

    # Should not raise validation errors
    result = block(dataset)
    assert len(result) == 1
    assert result[0]["output"] == "test response"


def test_validation_tags_only_configuration():
    """Test that tags-only configuration is valid."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
        # No parsing_pattern - this should be valid
    )

    data = [{"raw_output": "<answer>test response</answer>"}]
    dataset = Dataset.from_list(data)

    # Should not raise validation errors
    result = block(dataset)
    assert len(result) == 1
    assert result[0]["output"] == "test response"


def test_enhanced_error_handling_invalid_input_data():
    """Test enhanced error handling for invalid input data types."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        parsing_pattern=r"Answer: (.*)",
    )

    # Test with non-string input (separate datasets to avoid PyArrow issues)
    test_cases = [
        [{"raw_output": None}],
        [{"raw_output": 123}],
        [{"raw_output": ""}],  # Empty string instead of list to avoid PyArrow issues
    ]

    warning_count = 0
    for data in test_cases:
        dataset = Dataset.from_list(data)

        with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
            result = block.generate(dataset)

            # Should log warnings for invalid data
            if mock_logger.warning.called:
                warning_count += 1
            assert len(result) == 0  # Should return empty dataset

    assert warning_count >= 2  # At least None and 123 should trigger warnings


def test_enhanced_logging_for_parsing_failures():
    """Test enhanced logging when parsing fails."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    # Test with input that won't match the pattern
    data = [{"raw_output": "No tags in this text"}]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        block.generate(dataset)

        # Should log warning about parsing failure
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Failed to parse any content" in warning_call
        assert "parsing method: tags" in warning_call


def test_enhanced_logging_missing_input_column():
    """Test that BaseBlock handles missing input columns with proper validation."""
    # BaseBlock should handle missing column validation, so this should raise an error
    # during validation, not during generate()
    # First Party
    from sdg_hub.core.utils.error_handling import MissingColumnError

    block = TextParserBlock(
        block_name="test_block",
        input_cols="missing_column",
        output_cols=["output"],
        parsing_pattern=r"Answer: (.*)",
    )

    data = [{"other_column": "test"}]
    dataset = Dataset.from_list(data)

    # BaseBlock should validate and raise MissingColumnError
    with pytest.raises(MissingColumnError):
        block(dataset)  # Use __call__ to trigger validation


def test_enhanced_logging_regex_parsing():
    """Test enhanced debug logging for regex parsing."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        parsing_pattern=r"Answer: (.*)",
    )

    data = [{"raw_output": "Answer: test response"}]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        block.generate(dataset)

        # Should log debug info about parsing
        mock_logger.debug.assert_called()
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        # Check for the general parsing message
        assert any("Parsing outputs for" in call for call in debug_calls)


def test_enhanced_logging_tag_parsing():
    """Test enhanced debug logging for tag parsing."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    data = [{"raw_output": "<answer>test response</answer>"}]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        block.generate(dataset)

        # Should log debug info about parsing
        mock_logger.debug.assert_called()
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        # Check for the general parsing message
        assert any("Parsing outputs for" in call for call in debug_calls)


def test_generate_with_string_input():
    """Test generate functionality with string input."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    # String input
    data = [{"raw_output": "<answer>String response</answer>"}]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    assert len(result) == 1
    assert result[0]["output"] == "String response"


def test_generate_with_empty_string_input():
    """Test generate functionality with empty string input."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    # Empty string should trigger warning and return empty result
    data = [{"raw_output": ""}]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        result = block.generate(dataset)

        # Should log warning about empty string
        mock_logger.warning.assert_called_with(
            "Input column 'raw_output' contains empty string"
        )
        assert len(result) == 0


def test_generate_with_list_input_basic():
    """Test generate functionality with list of strings input."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    # List of strings input
    data = [
        {
            "raw_output": [
                "<answer>First response</answer>",
                "<answer>Second response</answer>",
                "<answer>Third response</answer>",
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should return single row with all parsed results collected as lists
    assert len(result) == 1
    assert result[0]["output"] == [
        "First response",
        "Second response",
        "Third response",
    ]


def test_generate_with_list_input_parsing_failures():
    """Test list input where some items fail to parse."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    # Some items have parseable content, others don't
    data = [
        {
            "raw_output": [
                "<answer>Parseable response 1</answer>",
                "No tags in this response",  # Won't parse
                "<answer>Parseable response 2</answer>",
                "Another response without tags",  # Won't parse
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        result = block.generate(dataset)

        # Should process only the 2 parseable responses
        assert len(result) == 1
        assert result[0]["output"] == ["Parseable response 1", "Parseable response 2"]

        # Should log warnings for parsing failures
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any(
            "Failed to parse content from list item 1" in call for call in warning_calls
        )
        assert any(
            "Failed to parse content from list item 3" in call for call in warning_calls
        )


def test_generate_with_invalid_list_items():
    """Test list input with invalid (non-string) items."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    # Test with separate datasets since PyArrow doesn't handle mixed types well
    # Test with valid string and empty string
    data = [{"raw_output": ["<answer>Valid</answer>", ""]}]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        result = block.generate(dataset)

        # Should process only the valid string
        assert len(result) == 1
        assert result[0]["output"] == ["Valid"]

        # Should log warning for empty string
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("is empty" in call for call in warning_calls)


def test_generate_with_empty_list_input():
    """Test generate functionality with empty list input."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    data = [{"raw_output": []}]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        result = block.generate(dataset)

        # Should log warning about empty list
        mock_logger.warning.assert_called_with(
            "Input column 'raw_output' contains empty list"
        )
        assert len(result) == 0


def test_generate_with_invalid_input_type():
    """Test handling of completely invalid input types (not string or list)."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    # Test with dict input (invalid type - should be string or list of strings)
    data = [{"raw_output": {"content": "test"}}]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        result = block.generate(dataset)

        # Should log warning about invalid data type
        mock_logger.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("invalid data type" in call for call in warning_calls)

        # Should return empty result
        assert len(result) == 0
