# Standard
from unittest.mock import MagicMock
import os
import warnings

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.deprecated_blocks.llmblock import LLMBlock
import pytest

# Get the absolute path to the test config file
TEST_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "testdata", "test_config.yaml"
)


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    client = MagicMock()
    client.models.list.return_value.data = [MagicMock(id="test-model")]
    client.base_url = "http://localhost:8000"
    client.api_key = "test"
    return client


@pytest.fixture
def llm_block(mock_client):
    """Create a basic LLMBlock instance for testing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LLMBlock(
            block_name="test_block",
            config_path=TEST_CONFIG_PATH,
            client=mock_client,
            output_cols=["output"],
            model_id="test-model",
            parser_kwargs={},
            model_prompt="{prompt}",
        )


def test_llm_block_initialization(llm_block):
    """Test that LLMBlock initializes properly with deprecation warning."""
    # Test that the block was created successfully
    assert llm_block.block_name == "test_block"
    assert llm_block.output_cols == ["output"]
    assert llm_block.model == "test-model"

    # Test that internal blocks were created
    assert hasattr(llm_block, "prompt_builder")
    assert hasattr(llm_block, "llm_chat")
    assert hasattr(llm_block, "text_parser")


def test_llm_block_deprecation_warning():
    """Test that LLMBlock issues deprecation warning on initialization."""
    client = MagicMock()
    client.models.list.return_value.data = [MagicMock(id="test-model")]
    client.base_url = "http://localhost:8000"
    client.api_key = "test"

    with pytest.warns(DeprecationWarning, match="LLMBlock is deprecated"):
        LLMBlock(
            block_name="test_block",
            config_path=TEST_CONFIG_PATH,
            client=client,
            output_cols=["output"],
            model_id="test-model",
        )


def test_generate_method_exists(llm_block):
    """Test that the generate method exists and accepts proper parameters."""
    # Create a simple test dataset
    test_data = [{"input": "test input"}]
    Dataset.from_list(test_data)

    # Test that generate method exists and can be called
    # Note: We're not testing the actual generation since that would require
    # mocking the internal blocks' behavior, which is tested separately
    assert hasattr(llm_block, "generate")
    assert callable(llm_block.generate)


def test_model_name_conversion_defaults_to_vllm():
    """Test that model names default to hosted_vllm."""
    client = MagicMock()
    client.models.list.return_value.data = [MagicMock(id="gpt-4")]
    client.base_url = "https://api.openai.com/v1"
    client.api_key = "test"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        block = LLMBlock(
            block_name="test_block",
            config_path=TEST_CONFIG_PATH,
            client=client,
            output_cols=["output"],
            model_id="gpt-4",
        )

    # Should default to hosted_vllm
    assert block.llm_chat.model == "hosted_vllm/gpt-4"


def test_model_name_conversion_hosted_vllm():
    """Test that model names are properly converted for hosted vLLM."""
    client = MagicMock()
    client.models.list.return_value.data = [MagicMock(id="llama-7b")]
    client.base_url = "http://localhost:8000"
    client.api_key = "test"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        block = LLMBlock(
            block_name="test_block",
            config_path=TEST_CONFIG_PATH,
            client=client,
            output_cols=["output"],
            model_id="llama-7b",
        )

    # Should detect as hosted_vllm since it's localhost
    assert block.llm_chat.model == "hosted_vllm/llama-7b"


def test_model_name_with_existing_prefix():
    """Test that model names with existing prefixes are preserved."""
    client = MagicMock()
    client.models.list.return_value.data = [MagicMock(id="hosted_vllm/llama-7b")]
    client.base_url = "http://localhost:8000"
    client.api_key = "test"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        block = LLMBlock(
            block_name="test_block",
            config_path=TEST_CONFIG_PATH,
            client=client,
            output_cols=["output"],
            model_id="hosted_vllm/llama-7b",
        )

    # Should preserve existing prefix
    assert block.llm_chat.model == "hosted_vllm/llama-7b"
