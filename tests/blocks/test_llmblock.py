import os
import pytest
from unittest.mock import MagicMock
from sdg_hub.blocks.llmblock import LLMBlock

# Get the absolute path to the test config file
TEST_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "testdata", "test_config.yaml"
)


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    client = MagicMock()
    client.models.list.return_value.data = [MagicMock(id="test-model")]
    return client


@pytest.fixture
def llm_block(mock_client):
    """Create a basic LLMBlock instance for testing."""
    return LLMBlock(
        block_name="test_block",
        config_path=TEST_CONFIG_PATH,
        client=mock_client,
        output_cols=["output"],
        parser_kwargs={},
        model_prompt="{prompt}",
    )


@pytest.fixture
def llm_block_with_custom_parser(mock_client):
    """Create an LLMBlock instance with custom parser configuration."""
    return LLMBlock(
        block_name="test_block",
        config_path=TEST_CONFIG_PATH,
        client=mock_client,
        output_cols=["output"],
        parser_kwargs={
            "parser_name": "custom",
            "parsing_pattern": r"Answer: (.*?)(?:\n|$)",
            "parser_cleanup_tags": ["<br>", "</br>"],
        },
        model_prompt="{prompt}",
    )


@pytest.fixture
def llm_block_with_tags(mock_client):
    """Create an LLMBlock instance with tag-based parsing configuration."""
    return LLMBlock(
        block_name="test_block",
        config_path=TEST_CONFIG_PATH,
        client=mock_client,
        output_cols=["output"],
        parser_kwargs={},
        model_prompt="{prompt}",
    )


def test_extract_matches_no_tags(llm_block):
    """Test extraction when no tags are provided."""
    text = "This is a test text"
    result = llm_block._extract_matches(text, None, None)
    assert result == ["This is a test text"]


def test_extract_matches_with_start_tag(llm_block):
    """Test extraction with only start tag."""
    text = "START This is a test text"
    result = llm_block._extract_matches(text, "START", None)
    assert result == ["This is a test text"]


def test_extract_matches_with_end_tag(llm_block):
    """Test extraction with only end tag."""
    text = "This is a test text END"
    result = llm_block._extract_matches(text, None, "END")
    assert result == ["This is a test text"]


def test_extract_matches_with_both_tags(llm_block):
    """Test extraction with both start and end tags."""
    text = "START This is a test text END"
    result = llm_block._extract_matches(text, "START", "END")
    assert result == ["This is a test text"]


def test_extract_matches_multiple_matches(llm_block):
    """Test extraction with multiple matches."""
    text = "START First text END START Second text END"
    result = llm_block._extract_matches(text, "START", "END")
    assert result == ["First text", "Second text"]


def test_extract_matches_incomplete_tags(llm_block):
    """Test extraction with incomplete tag pairs."""
    text = "START First text END START Second text"
    result = llm_block._extract_matches(text, "START", "END")
    assert result == ["First text"]


def test_extract_matches_cascading_tags(llm_block):
    """Test extraction with cascading start and end tags."""
    text = "START1 START2 Nested text END2 END1"
    result = llm_block._extract_matches(text, "START1", "END1")
    assert result == ["START2 Nested text END2"]

    result = llm_block._extract_matches(text, "START2", "END2")
    assert result == ["Nested text"]


def test_parse_multiple_tags(llm_block):
    """Test parsing with multiple start and end tags for different output columns."""
    # Configure the block with multiple tag pairs
    llm_block.block_config = {
        "start_tags": ["<title>", "<content>"],
        "end_tags": ["</title>", "</content>"],
    }
    llm_block.output_cols = ["title", "content"]

    # Test text with multiple tag pairs
    text = """
    <title>First Title</title>
    <content>First Content</content>
    <title>Second Title</title>
    <content>Second Content</content>
    """

    result = llm_block._parse(text)
    assert result == {
        "title": ["First Title", "Second Title"],
        "content": ["First Content", "Second Content"],
    }


def test_parse_mixed_tag_types(llm_block):
    """Test parsing with mixed tag types (XML-style and custom markers)."""
    llm_block.block_config = {
        "start_tags": ["<header>", "START"],
        "end_tags": ["</header>", "END"],
    }
    llm_block.output_cols = ["header", "body"]

    text = """
    <header>XML Style Header</header>
    START Custom Style Body END
    <header>Another XML Header</header>
    START Another Custom Body END
    """

    result = llm_block._parse(text)
    assert result == {
        "header": ["XML Style Header", "Another XML Header"],
        "body": ["Custom Style Body", "Another Custom Body"],
    }


def test_parse_with_whitespace(llm_block):
    """Test parsing with various whitespace patterns."""
    llm_block.block_config = {"start_tags": ["<text>"], "end_tags": ["</text>"]}
    llm_block.output_cols = ["text"]

    text = """
    <text>  Leading and trailing spaces  </text>
    <text>
    Multiple
    Lines
    </text>
    <text>\tTabbed content\t</text>
    """

    result = llm_block._parse(text)
    assert result == {
        "text": ["Leading and trailing spaces", "Multiple\n    Lines", "Tabbed content"]
    }


def test_parse_with_special_characters(llm_block):
    """Test parsing with special characters in tags and content."""
    llm_block.block_config = {"start_tags": ["<special>"], "end_tags": ["</special>"]}
    llm_block.output_cols = ["special"]

    text = """
    <special>Content with &amp; entities</special>
    <special>Content with <nested> tags</special>
    <special>Content with "quotes" and 'apostrophes'</special>
    """

    result = llm_block._parse(text)
    assert result == {
        "special": [
            "Content with &amp; entities",
            "Content with <nested> tags",
            "Content with \"quotes\" and 'apostrophes'",
        ]
    }


def test_parse_uneven_tags(llm_block):
    """Test parsing with uneven or mismatched start and end tags."""
    llm_block.block_config = {
        "start_tags": ["<section>", "<subsection>"],
        "end_tags": ["</section>", "</subsection>"],
    }
    llm_block.output_cols = ["section", "subsection"]

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
        result = llm_block._parse(text)
        assert result == expected, f"Failed for text: {text}"


def test_parse_mismatched_config_tags(llm_block):
    """Test parsing with mismatched numbers of start and end tags in configuration."""
    # Test case 1: More start tags than end tags
    llm_block.block_config = {
        "start_tags": ["<header>", "<content>", "<footer>"],
        "end_tags": ["</header>", "</content>"],
    }
    llm_block.output_cols = ["header", "content", "footer"]

    text = """
    <header>Header content</header>
    <content>Main content</content>
    <footer>Footer content</footer>
    """

    result = llm_block._parse(text)
    assert result == {
        "header": ["Header content"],
        "content": ["Main content"],
        # footer key is not present in result when no matching end tag
    }

    # Test case 2: More end tags than start tags
    llm_block.block_config = {
        "start_tags": ["<header>"],
        "end_tags": ["</header>", "</content>", "</footer>"],
    }
    llm_block.output_cols = ["header", "content", "footer"]

    text = """
    <header>Header content</header>
    </content>
    </footer>
    """

    result = llm_block._parse(text)
    assert result == {
        "header": ["Header content"]
        # content and footer keys are not present when no matching start tags
    }

    # Test case 3: Empty tags list
    llm_block.block_config = {"start_tags": [], "end_tags": []}
    llm_block.output_cols = ["text"]

    text = "Some text without tags"

    result = llm_block._parse(text)
    assert result == {}  # When no tags are configured, parser returns empty dict


def test_custom_parser_single_match(llm_block_with_custom_parser):
    """Test custom parser with a single match."""
    text = "Question: What is the answer?\nAnswer: This is the answer"
    result = llm_block_with_custom_parser._parse(text)
    assert result == {"output": ["This is the answer"]}


def test_custom_parser_multiple_matches(llm_block_with_custom_parser):
    """Test custom parser with multiple matches."""
    text = "Question 1: What is the answer?\nAnswer: First answer\nQuestion 2: Another question?\nAnswer: Second answer"
    result = llm_block_with_custom_parser._parse(text)
    assert result == {"output": ["First answer", "Second answer"]}


def test_tag_based_parsing(llm_block_with_tags):
    """Test tag-based parsing configuration."""
    text = "Some text <output>This is the output</output> more text"
    result = llm_block_with_tags._parse(text)
    assert result == {"output": ["This is the output"]}


def test_parse_empty_input(llm_block):
    """Test parsing with empty input."""
    result = llm_block._parse("")
    assert result == {"output": []}


def test_parse_no_matches(llm_block_with_custom_parser):
    """Test parsing when no matches are found."""
    text = "This text has no matches for the pattern"
    result = llm_block_with_custom_parser._parse(text)
    assert result == {"output": []}
