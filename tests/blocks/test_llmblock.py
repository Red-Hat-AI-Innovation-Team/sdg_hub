import os
import pytest
from unittest.mock import MagicMock
from sdg_hub.blocks.llmblock import LLMBlock

# Get the absolute path to the test config file
TEST_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "testdata", "test_config.yaml"
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
