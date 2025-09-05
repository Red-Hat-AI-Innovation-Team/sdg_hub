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
        {"raw_output": {"content": "Text <output>Result 1</output> more text"}},
        {"raw_output": {"content": "Text <output>Result 2</output> more text"}},
    ]
    dataset = Dataset.from_list(data)

    result = postprocessing_block.generate(dataset)

    assert len(result) == 2
    assert result[0]["output"] == "Result 1"
    assert result[1]["output"] == "Result 2"


def test_generate_custom_regex(postprocessing_block_with_custom_parser):
    """Test generate functionality with custom regex parsing."""
    data = [
        {"raw_output": {"content": "Question: Q1\nAnswer: A1"}},
        {"raw_output": {"content": "Question: Q2\nAnswer: A2"}},
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
            "raw_output": {
                "content": """
            <title>Title 1</title>
            <content>Content 1</content>
            <title>Title 2</title>
            <content>Content 2</content>
            """
            }
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
        {"raw_output": {"content": "Text without any tags"}},
        {"raw_output": {"content": "More text without tags"}},
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
        {
            "raw_output": {
                "content": "Question: What is the answer?\nNo answer provided"
            }
        },
        {"raw_output": {"content": "Another question without answer"}},
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

    data = [{"raw_output": {"content": "Answer: test response"}}]
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

    data = [{"raw_output": {"content": "<answer>test response</answer>"}}]
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
    data = [{"raw_output": {"content": "No tags in this text"}}]
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

    data = [{"raw_output": {"content": "Answer: test response"}}]
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

    data = [{"raw_output": {"content": "<answer>test response</answer>"}}]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        block.generate(dataset)

        # Should log debug info about parsing
        mock_logger.debug.assert_called()
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        # Check for the general parsing message
        assert any("Parsing outputs for" in call for call in debug_calls)


def test_generate_with_list_input_tag_parsing():
    """Test generate functionality with list input from LLMChatBlock (n > 1) using tag parsing."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    # Simulate output from LLMChatBlock with n=3
    data = [
        {
            "raw_output": [
                {"content": "<answer>First response</answer>"},
                {"content": "<answer>Second response</answer>"},
                {"content": "<answer>Third response</answer>"},
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should create 3 output rows, one for each response in the list
    assert len(result) == 3
    assert result[0]["output"] == "First response"
    assert result[1]["output"] == "Second response"
    assert result[2]["output"] == "Third response"


def test_generate_with_list_input_regex_parsing():
    """Test generate functionality with list input using regex parsing."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        parsing_pattern=r"Answer: (.*?)(?:\n|$)",
    )

    # Simulate output from LLMChatBlock with n=2
    data = [
        {
            "raw_output": [
                {"content": "Question: What is 2+2?\nAnswer: Four"},
                {"content": "Question: What is 3+3?\nAnswer: Six"},
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should create 2 output rows
    assert len(result) == 2
    assert result[0]["output"] == "Four"
    assert result[1]["output"] == "Six"


def test_generate_with_list_input_multiple_matches_per_response():
    """Test list input where each response contains multiple matches."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["title", "content"],
        start_tags=["<title>", "<content>"],
        end_tags=["</title>", "</content>"],
    )

    # Each response has multiple matches
    data = [
        {
            "raw_output": [
                {
                    "content": "<title>Title 1</title><content>Content 1</content><title>Title 2</title><content>Content 2</content>"
                },
                {"content": "<title>Title 3</title><content>Content 3</content>"},
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # First response creates 2 rows, second response creates 1 row = 3 total
    assert len(result) == 3
    assert result[0]["title"] == "Title 1"
    assert result[0]["content"] == "Content 1"
    assert result[1]["title"] == "Title 2"
    assert result[1]["content"] == "Content 2"
    assert result[2]["title"] == "Title 3"
    assert result[2]["content"] == "Content 3"


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


def test_generate_with_mixed_valid_invalid_list_items():
    """Test list input with some valid and some invalid items."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    # Test separate cases since PyArrow doesn't handle mixed types well
    # Test with empty strings
    data_with_empty = [
        {
            "raw_output": [
                {"content": "<answer>Valid response 1</answer>"},
                {"content": ""},  # Empty string
                {"content": "<answer>Valid response 2</answer>"},
            ]
        }
    ]
    dataset = Dataset.from_list(data_with_empty)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        result = block.generate(dataset)

        # Should process only the 2 valid responses
        assert len(result) == 2
        assert result[0]["output"] == "Valid response 1"
        assert result[1]["output"] == "Valid response 2"

        # Should log warning for failed parsing (empty content)
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any(
            "Failed to parse content from list item 1" in call for call in warning_calls
        )


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
                {"content": "<answer>Parseable response 1</answer>"},
                {"content": "No tags in this response"},  # Won't parse
                {"content": "<answer>Parseable response 2</answer>"},
                {"content": "Another response without tags"},  # Won't parse
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        result = block.generate(dataset)

        # Should process only the 2 parseable responses
        assert len(result) == 2
        assert result[0]["output"] == "Parseable response 1"
        assert result[1]["output"] == "Parseable response 2"

        # Should log warnings for parsing failures
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any(
            "Failed to parse content from list item 1" in call for call in warning_calls
        )
        assert any(
            "Failed to parse content from list item 3" in call for call in warning_calls
        )


def test_generate_with_list_input_all_invalid():
    """Test list input where all items are invalid or fail parsing."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    data = [
        {
            "raw_output": [
                {"content": "No tags here"},
                {"content": ""},
                None,
                {"content": "Also no tags"},
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should return empty result
    assert len(result) == 0


def test_backwards_compatibility_string_input():
    """Test that dict inputs work as expected."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    # Dict input (new behavior)
    data = [{"raw_output": {"content": "<answer>Single string response</answer>"}}]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should work as expected
    assert len(result) == 1
    assert result[0]["output"] == "Single string response"


def test_generate_with_invalid_input_type():
    """Test handling of completely invalid input types (not dict or list)."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
    )

    # Test with list of strings (invalid type - should be list of dicts)
    data = [{"raw_output": ["test"]}]  # list of strings instead of list of dicts
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        result = block.generate(dataset)

        # Should log warnings about content not found and parsing failure
        mock_logger.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("Content not found in sample" in call for call in warning_calls)
        assert any(
            "Failed to parse content from list item" in call for call in warning_calls
        )

        # Should return empty result
        assert len(result) == 0


# Tests for expand_lists functionality
def test_expand_lists_false_basic_functionality():
    """Test basic functionality with expand_lists=False."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["entities"],
        start_tags=["<entity>"],
        end_tags=["</entity>"],
        expand_lists=False,
    )

    # List input with multiple entity responses
    data = [
        {
            "raw_output": [
                {"content": "<entity>A</entity><entity>B</entity>"},
                {"content": "<entity>C</entity>"},
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should return single row with entities as a list
    assert len(result) == 1
    assert result[0]["entities"] == ["A", "B", "C"]


def test_expand_lists_false_multiple_output_columns():
    """Test expand_lists=False with multiple output columns."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["title", "content"],
        start_tags=["<title>", "<content>"],
        end_tags=["</title>", "</content>"],
        expand_lists=False,
    )

    # List input with title/content pairs
    data = [
        {
            "raw_output": [
                {"content": "<title>Title 1</title><content>Content 1</content>"},
                {"content": "<title>Title 2</title><content>Content 2</content>"},
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should return single row with lists for each column
    assert len(result) == 1
    assert result[0]["title"] == ["Title 1", "Title 2"]
    assert result[0]["content"] == ["Content 1", "Content 2"]


def test_expand_lists_false_with_regex_parsing():
    """Test expand_lists=False with regex parsing."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["answer"],
        parsing_pattern=r"Answer: (.*?)(?:\n|$)",
        expand_lists=False,
    )

    # List input with multiple answers
    data = [
        {
            "raw_output": [
                {"content": "Question 1\nAnswer: Response 1"},
                {"content": "Question 2\nAnswer: Response 2\nAnswer: Response 3"},
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should return single row with all answers as a list
    assert len(result) == 1
    assert result[0]["answer"] == ["Response 1", "Response 2", "Response 3"]


def test_expand_lists_false_with_parsing_failures():
    """Test expand_lists=False when some items fail to parse."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["entity"],
        start_tags=["<entity>"],
        end_tags=["</entity>"],
        expand_lists=False,
    )

    # Mix of valid and invalid responses
    data = [
        {
            "raw_output": [
                {"content": "<entity>Valid 1</entity>"},
                {"content": "No tags here"},  # Will fail to parse
                {"content": "<entity>Valid 2</entity>"},
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        result = block.generate(dataset)

        # Should return single row with only valid entities
        assert len(result) == 1
        assert result[0]["entity"] == ["Valid 1", "Valid 2"]

        # Should log warning for parsing failure
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any(
            "Failed to parse content from list item 1" in call for call in warning_calls
        )


def test_expand_lists_false_empty_list():
    """Test expand_lists=False with empty list input."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["entity"],
        start_tags=["<entity>"],
        end_tags=["</entity>"],
        expand_lists=False,
    )

    data = [{"raw_output": []}]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        result = block.generate(dataset)

        # Should return empty result and log warning
        assert len(result) == 0
        mock_logger.warning.assert_called_with(
            "Input column 'raw_output' contains empty list"
        )


def test_expand_lists_false_all_parsing_failures():
    """Test expand_lists=False when all items fail to parse."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["entity"],
        start_tags=["<entity>"],
        end_tags=["</entity>"],
        expand_lists=False,
    )

    # All responses fail to parse
    data = [
        {
            "raw_output": [
                {"content": "No tags here"},
                {"content": "Also no tags"},
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should return empty result
    assert len(result) == 0


def test_expand_lists_true_backward_compatibility():
    """Test that expand_lists=True (default) maintains backward compatibility."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["entity"],
        start_tags=["<entity>"],
        end_tags=["</entity>"],
        expand_lists=True,  # Explicit True (same as default)
    )

    # List input
    data = [
        {
            "raw_output": [
                {"content": "<entity>A</entity><entity>B</entity>"},
                {"content": "<entity>C</entity>"},
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should expand to individual rows (existing behavior)
    assert len(result) == 3
    assert result[0]["entity"] == "A"
    assert result[1]["entity"] == "B"
    assert result[2]["entity"] == "C"


def test_expand_lists_default_value():
    """Test that expand_lists defaults to True for backward compatibility."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["entity"],
        start_tags=["<entity>"],
        end_tags=["</entity>"],
        # expand_lists not specified - should default to True
    )

    # Verify default value
    assert block.expand_lists is True

    # Test behavior matches expanding behavior
    data = [
        {
            "raw_output": [
                {"content": "<entity>A</entity>"},
                {"content": "<entity>B</entity>"},
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should expand to individual rows (default behavior)
    assert len(result) == 2
    assert result[0]["entity"] == "A"
    assert result[1]["entity"] == "B"


# Tests for save_reasoning_content functionality
def test_save_reasoning_content_basic_dict_input():
    """Test basic save_reasoning_content functionality with dict input."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
        save_reasoning_content=True,
    )

    data = [
        {
            "raw_output": {
                "content": "<answer>Final answer</answer>",
                "reasoning_content": "This is my reasoning process",
            }
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    assert len(result) == 1
    assert result[0]["output"] == "Final answer"
    assert result[0]["test_block_reasoning_content"] == "This is my reasoning process"


def test_save_reasoning_content_custom_field_name():
    """Test save_reasoning_content with custom field name."""
    block = TextParserBlock(
        block_name="parser",
        input_cols="raw_output",
        output_cols=["answer"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
        save_reasoning_content=True,
        reasoning_content_field="custom_reasoning",
    )

    data = [
        {
            "raw_output": {
                "content": "<answer>My answer</answer>",
                "custom_reasoning": "Custom reasoning field content",
            }
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    assert len(result) == 1
    assert result[0]["answer"] == "My answer"
    assert result[0]["parser_custom_reasoning"] == "Custom reasoning field content"


def test_save_reasoning_content_disabled_by_default():
    """Test that save_reasoning_content is disabled by default."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
        # save_reasoning_content not specified - should default to False
    )

    # Verify default value
    assert block.save_reasoning_content is False

    data = [
        {
            "raw_output": {
                "content": "<answer>Final answer</answer>",
                "reasoning_content": "This reasoning should not be saved",
            }
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    assert len(result) == 1
    assert result[0]["output"] == "Final answer"
    # Should not have reasoning content field
    assert "test_block_reasoning_content" not in result[0]


def test_save_reasoning_content_missing_field():
    """Test save_reasoning_content when reasoning field is missing from input."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
        save_reasoning_content=True,
    )

    data = [
        {
            "raw_output": {
                "content": "<answer>Final answer</answer>",
                # No reasoning_content field
            }
        }
    ]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        result = block.generate(dataset)

        # Should log warning about missing field
        mock_logger.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any(
            "Reasoning content field 'reasoning_content' not found" in call
            for call in warning_calls
        )

        assert len(result) == 1
        assert result[0]["output"] == "Final answer"
        assert result[0]["test_block_reasoning_content"] == ""


def test_save_reasoning_content_with_regex_parsing():
    """Test save_reasoning_content with regex parsing."""
    block = TextParserBlock(
        block_name="regex_block",
        input_cols="raw_output",
        output_cols=["answer"],
        parsing_pattern=r"Answer: (.*?)(?:\n|$)",
        save_reasoning_content=True,
    )

    data = [
        {
            "raw_output": {
                "content": "Question: What is 2+2?\nAnswer: Four",
                "reasoning_content": "Simple arithmetic calculation",
            }
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    assert len(result) == 1
    assert result[0]["answer"] == "Four"
    assert result[0]["regex_block_reasoning_content"] == "Simple arithmetic calculation"


def test_save_reasoning_content_list_input_expand_true():
    """Test save_reasoning_content with list input and expand_lists=True."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
        save_reasoning_content=True,
        expand_lists=True,
    )

    data = [
        {
            "raw_output": [
                {
                    "content": "<answer>First answer</answer>",
                    "reasoning_content": "First reasoning",
                },
                {
                    "content": "<answer>Second answer</answer>",
                    "reasoning_content": "Second reasoning",
                },
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should create separate rows for each response
    assert len(result) == 2
    assert result[0]["output"] == "First answer"
    assert result[0]["test_block_reasoning_content"] == "First reasoning"
    assert result[1]["output"] == "Second answer"
    assert result[1]["test_block_reasoning_content"] == "Second reasoning"


def test_save_reasoning_content_list_input_expand_false():
    """Test save_reasoning_content with list input and expand_lists=False."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
        save_reasoning_content=True,
        expand_lists=False,
    )

    data = [
        {
            "raw_output": [
                {
                    "content": "<answer>First answer</answer>",
                    "reasoning_content": "First reasoning",
                },
                {
                    "content": "<answer>Second answer</answer>",
                    "reasoning_content": "Second reasoning",
                },
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)
    # Should create single row with lists
    assert len(result) == 1
    assert result[0]["output"] == ["First answer", "Second answer"]
    assert result[0]["test_block_reasoning_content"] == [
        "First reasoning",
        "Second reasoning",
    ]


def test_save_reasoning_content_list_input_multiple_matches():
    """Test save_reasoning_content with list input where each response has multiple matches."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["title", "content"],
        start_tags=["<title>", "<content>"],
        end_tags=["</title>", "</content>"],
        save_reasoning_content=True,
        expand_lists=True,
    )

    data = [
        {
            "raw_output": [
                {
                    "content": "<title>Title 1</title><content>Content 1</content><title>Title 2</title><content>Content 2</content>",
                    "reasoning_content": "First response reasoning",
                },
                {
                    "content": "<title>Title 3</title><content>Content 3</content>",
                    "reasoning_content": "Second response reasoning",
                },
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should create 3 rows total (2 from first response, 1 from second)
    assert len(result) == 3
    assert result[0]["title"] == "Title 1"
    assert result[0]["content"] == "Content 1"
    assert result[0]["test_block_reasoning_content"] == "First response reasoning"
    assert result[1]["title"] == "Title 2"
    assert result[1]["content"] == "Content 2"
    assert result[1]["test_block_reasoning_content"] == "First response reasoning"
    assert result[2]["title"] == "Title 3"
    assert result[2]["content"] == "Content 3"
    assert result[2]["test_block_reasoning_content"] == "Second response reasoning"


def test_save_reasoning_content_list_input_expand_false_multiple_matches():
    """Test save_reasoning_content with list input, expand_lists=False, and multiple matches per response."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["entity"],
        start_tags=["<entity>"],
        end_tags=["</entity>"],
        save_reasoning_content=True,
        expand_lists=False,
    )

    data = [
        {
            "raw_output": [
                {
                    "content": "<entity>A</entity><entity>B</entity>",
                    "reasoning_content": "First reasoning",
                },
                {
                    "content": "<entity>C</entity>",
                    "reasoning_content": "Second reasoning",
                },
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should create single row with lists
    assert len(result) == 1
    assert result[0]["entity"] == ["A", "B", "C"]
    assert result[0]["test_block_reasoning_content"] == [
        "First reasoning",
        "Second reasoning",
    ]


def test_save_reasoning_content_list_input_mixed_valid_invalid():
    """Test save_reasoning_content with list input containing valid and invalid responses."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
        save_reasoning_content=True,
        expand_lists=True,
    )

    data = [
        {
            "raw_output": [
                {
                    "content": "<answer>Valid answer 1</answer>",
                    "reasoning_content": "Valid reasoning 1",
                },
                {
                    "content": "No tags here",  # Will fail to parse
                    "reasoning_content": "This reasoning won't be used",
                },
                {
                    "content": "<answer>Valid answer 2</answer>",
                    "reasoning_content": "Valid reasoning 2",
                },
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    with patch("sdg_hub.core.blocks.llm.text_parser_block.logger") as mock_logger:
        result = block.generate(dataset)

        # Should process only the 2 valid responses
        assert len(result) == 2
        assert result[0]["output"] == "Valid answer 1"
        assert result[0]["test_block_reasoning_content"] == "Valid reasoning 1"
        assert result[1]["output"] == "Valid answer 2"
        assert result[1]["test_block_reasoning_content"] == "Valid reasoning 2"

        # Should log warning for parsing failure
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any(
            "Failed to parse content from list item 1" in call for call in warning_calls
        )


def test_save_reasoning_content_empty_reasoning_field():
    """Test save_reasoning_content when reasoning field is empty."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
        save_reasoning_content=True,
    )

    data = [
        {
            "raw_output": {
                "content": "<answer>Final answer</answer>",
                "reasoning_content": "",  # Empty reasoning content
            }
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    assert len(result) == 1
    assert result[0]["output"] == "Final answer"
    assert result[0]["test_block_reasoning_content"] == ""


def test_save_reasoning_content_default_field_name():
    """Test that reasoning_content_field defaults to 'reasoning_content'."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
        save_reasoning_content=True,
        # reasoning_content_field not specified - should default to 'reasoning_content'
    )

    # Verify default value
    assert block.reasoning_content_field == "reasoning_content"

    data = [
        {
            "raw_output": {
                "content": "<answer>Final answer</answer>",
                "reasoning_content": "Default field reasoning",
            }
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    assert len(result) == 1
    assert result[0]["output"] == "Final answer"
    assert result[0]["test_block_reasoning_content"] == "Default field reasoning"


def test_save_reasoning_content_multiple_responses_one_per_row():
    """Test that when n>1, each row gets its corresponding reasoning content."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
        save_reasoning_content=True,
        expand_lists=True,
    )

    # Simulate LLMChatBlock output with n=3 (3 different responses)
    data = [
        {
            "raw_output": [
                {
                    "content": "<answer>Response 1</answer>",
                    "reasoning_content": "Reasoning for response 1",
                },
                {
                    "content": "<answer>Response 2</answer>",
                    "reasoning_content": "Reasoning for response 2",
                },
                {
                    "content": "<answer>Response 3</answer>",
                    "reasoning_content": "Reasoning for response 3",
                },
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should create 3 separate rows, each with its own reasoning
    assert len(result) == 3

    # Each row should have the reasoning content from its corresponding response
    assert result[0]["output"] == "Response 1"
    assert result[0]["test_block_reasoning_content"] == "Reasoning for response 1"

    assert result[1]["output"] == "Response 2"
    assert result[1]["test_block_reasoning_content"] == "Reasoning for response 2"

    assert result[2]["output"] == "Response 3"
    assert result[2]["test_block_reasoning_content"] == "Reasoning for response 3"


def test_save_reasoning_content_multiple_responses_collected_as_list():
    """Test that when n>1 and expand_lists=False, reasoning contents are collected as a list."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
        start_tags=["<answer>"],
        end_tags=["</answer>"],
        save_reasoning_content=True,
        expand_lists=False,
    )

    # Simulate LLMChatBlock output with n=3 (3 different responses)
    data = [
        {
            "raw_output": [
                {
                    "content": "<answer>Response 1</answer>",
                    "reasoning_content": "Reasoning for response 1",
                },
                {
                    "content": "<answer>Response 2</answer>",
                    "reasoning_content": "Reasoning for response 2",
                },
                {
                    "content": "<answer>Response 3</answer>",
                    "reasoning_content": "Reasoning for response 3",
                },
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    result = block.generate(dataset)

    # Should create single row with lists containing all responses and reasoning
    assert len(result) == 1
    assert result[0]["output"] == ["Response 1", "Response 2", "Response 3"]
    assert result[0]["test_block_reasoning_content"] == [
        "Reasoning for response 1",
        "Reasoning for response 2",
        "Reasoning for response 3",
    ]
