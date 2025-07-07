# Third Party
from datasets import Dataset
import pytest

# First Party
from sdg_hub.blocks.llm import TextParserBlock


@pytest.fixture
def postprocessing_block():
    """Create a basic TextParserBlock instance for testing."""
    return TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
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
    """Test generate functionality when input column is missing."""
    data = [{"other_column": "some text"}]
    dataset = Dataset.from_list(data)

    result = postprocessing_block.generate(dataset)

    assert len(result) == 0


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
    """Test constructor validation with no input columns."""
    with pytest.raises(
        ValueError, match="TextParserBlock expects at least one input column"
    ):
        TextParserBlock(
            block_name="test_block",
            input_cols=[],
            output_cols=["output"],
        )


def test_constructor_validation_multiple_input_cols():
    """Test constructor validation with multiple input columns (should warn but not error)."""
    # This should not raise an error, just log a warning
    block = TextParserBlock(
        block_name="test_block",
        input_cols=["col1", "col2"],
        output_cols=["output"],
    )

    assert len(block.input_cols) == 2
    assert block.input_cols[0] == "col1"


def test_constructor_string_input_cols():
    """Test constructor with string input_cols (should be converted to list)."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols=["output"],
    )

    assert block.input_cols == ["raw_output"]


def test_constructor_string_output_cols():
    """Test constructor with string output_cols (should be converted to list)."""
    block = TextParserBlock(
        block_name="test_block",
        input_cols="raw_output",
        output_cols="output",
    )

    assert block.output_cols == ["output"]


def test_parse_uneven_tags(postprocessing_block_multi_column):
    """Test parsing with uneven start and end tags."""
    # Test with more start tags than end tags
    postprocessing_block_multi_column.start_tags = ["<title>", "<content>", "<footer>"]
    postprocessing_block_multi_column.end_tags = ["</title>", "</content>"]
    postprocessing_block_multi_column.output_cols = ["title", "content", "footer"]

    text = """
    <title>Header content</title>
    <content>Main content</content>
    <footer>Footer content</footer>
    """

    result = postprocessing_block_multi_column._parse(text)
    assert result == {
        "title": ["Header content"],
        "content": ["Main content"],
        "footer": [],
    }


def test_parse_more_output_cols_than_tags(postprocessing_block_multi_column):
    """Test parsing when there are more output columns than tag pairs."""
    # Configure with 3 output columns but only 2 tag pairs
    postprocessing_block_multi_column.start_tags = ["<title>", "<content>"]
    postprocessing_block_multi_column.end_tags = ["</title>", "</content>"]
    postprocessing_block_multi_column.output_cols = ["title", "content", "footer"]

    text = """
    <title>Header content</title>
    <content>Main content</content>
    """

    result = postprocessing_block_multi_column._parse(text)
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
    postprocessing_block_multi_column.start_tags = ["<header>", "START"]
    postprocessing_block_multi_column.end_tags = ["</header>", "END"]
    postprocessing_block_multi_column.output_cols = ["header", "body"]

    text = """
    <header>XML Style Header</header>
    START Custom Style Body END
    <header>Another XML Header</header>
    START Another Custom Body END
    """

    result = postprocessing_block_multi_column._parse(text)
    assert result == {
        "header": ["XML Style Header", "Another XML Header"],
        "body": ["Custom Style Body", "Another Custom Body"],
    }


def test_parse_with_special_characters(postprocessing_block_with_tags):
    """Test parsing with special characters in tags and content."""
    postprocessing_block_with_tags.start_tags = ["<special>"]
    postprocessing_block_with_tags.end_tags = ["</special>"]
    postprocessing_block_with_tags.output_cols = ["special"]

    text = """
    <special>Content with &amp; entities</special>
    <special>Content with <nested> tags</special>
    <special>Content with "quotes" and 'apostrophes'</special>
    """

    result = postprocessing_block_with_tags._parse(text)
    assert result == {
        "special": [
            "Content with &amp; entities",
            "Content with <nested> tags",
            "Content with \"quotes\" and 'apostrophes'",
        ]
    }


def test_parse_mismatched_config_tags(postprocessing_block_multi_column):
    """Test parsing with mismatched numbers of start and end tags in configuration."""
    # Test case 1: More start tags than end tags
    postprocessing_block_multi_column.start_tags = ["<header>", "<content>", "<footer>"]
    postprocessing_block_multi_column.end_tags = ["</header>", "</content>"]
    postprocessing_block_multi_column.output_cols = ["header", "content", "footer"]

    text = """
    <header>Header content</header>
    <content>Main content</content>
    <footer>Footer content</footer>
    """

    result = postprocessing_block_multi_column._parse(text)
    assert result == {
        "header": ["Header content"],
        "content": ["Main content"],
        "footer": [],
    }

    # Test case 2: More end tags than start tags
    postprocessing_block_multi_column.start_tags = ["<header>"]
    postprocessing_block_multi_column.end_tags = [
        "</header>",
        "</content>",
        "</footer>",
    ]
    postprocessing_block_multi_column.output_cols = ["header", "content", "footer"]

    text = """
    <header>Header content</header>
    </content>
    </footer>
    """

    result = postprocessing_block_multi_column._parse(text)
    assert result == {"header": ["Header content"], "content": [], "footer": []}

    # Test case 3: Empty tags list
    postprocessing_block_multi_column.start_tags = []
    postprocessing_block_multi_column.end_tags = []
    postprocessing_block_multi_column.output_cols = ["text"]

    text = "Some text without tags"

    result = postprocessing_block_multi_column._parse(text)
    assert result == {"text": []}


def test_parse_uneven_tags_comprehensive(postprocessing_block_multi_column):
    """Test parsing with uneven or mismatched start and end tags - comprehensive test cases."""
    postprocessing_block_multi_column.start_tags = ["<section>", "<subsection>"]
    postprocessing_block_multi_column.end_tags = ["</section>", "</subsection>"]
    postprocessing_block_multi_column.output_cols = ["section", "subsection"]

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
        result = postprocessing_block_multi_column._parse(text)
        assert result == expected, f"Failed for text: {text}"


def test_parse_with_whitespace_comprehensive(postprocessing_block_with_tags):
    """Test parsing with various whitespace patterns - comprehensive test."""
    postprocessing_block_with_tags.start_tags = ["<text>"]
    postprocessing_block_with_tags.end_tags = ["</text>"]
    postprocessing_block_with_tags.output_cols = ["text"]

    text = """
    <text>  Leading and trailing spaces  </text>
    <text>
    Multiple
    Lines
    </text>
    <text>\tTabbed content\t</text>
    """

    result = postprocessing_block_with_tags._parse(text)
    assert result == {
        "text": ["Leading and trailing spaces", "Multiple\n    Lines", "Tabbed content"]
    }
