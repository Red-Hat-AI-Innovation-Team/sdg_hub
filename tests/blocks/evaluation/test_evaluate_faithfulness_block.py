# SPDX-License-Identifier: Apache-2.0
"""Tests for EvaluateFaithfulnessBlock."""

# Standard
from unittest.mock import MagicMock, patch
import os
import tempfile

# Third Party
from datasets import Dataset

# First Party
from sdg_hub import BlockRegistry
from sdg_hub.core.blocks.evaluation.evaluate_faithfulness_block import (
    EvaluateFaithfulnessBlock,
)
import pytest


class TestEvaluateFaithfulnessBlock:
    """Test cases for EvaluateFaithfulnessBlock."""

    @pytest.fixture
    def test_yaml_config(self):
        """Create a temporary YAML config file for testing."""
        yaml_content = """- role: "user"
  content: |
    Please evaluate the faithfulness of the following response to the given document.
    
    Document: {{ document }}
    
    Response: {{ response }}
    
    Please provide your evaluation in the following format:
    
    [Start of Explanation]
    Provide a detailed explanation of why the response is or is not faithful to the document.
    [End of Explanation]
    
    [Start of Answer]
    YES or NO
    [End of Answer]"""

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        temp_file.write(yaml_content)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)

    def test_block_registry(self):
        """Test that EvaluateFaithfulnessBlock is properly registered."""
        block_class = BlockRegistry._get("EvaluateFaithfulnessBlock")
        assert block_class == EvaluateFaithfulnessBlock

        # Check category
        eval_blocks = BlockRegistry.list_blocks(category="evaluation")
        assert "EvaluateFaithfulnessBlock" in eval_blocks

    def test_init_with_valid_params(self, test_yaml_config):
        """Test initialization with valid parameters."""
        block = EvaluateFaithfulnessBlock(
            block_name="test_faithfulness",
            input_cols=["document", "response"],
            output_cols=[
                "faithfulness_explanation",
                "faithfulness_judgment",
            ],
            prompt_config_path=test_yaml_config,
            model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
        )

        # Test basic block properties
        assert block.block_name == "test_faithfulness"
        assert block.input_cols == ["document", "response"]
        assert block.output_cols == [
            "faithfulness_explanation",
            "faithfulness_judgment",
        ]

        # Test parameters are stored in appropriate dictionaries
        assert "model" in block.llm_params
        assert block.llm_params["model"] == "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"
        assert "prompt_config_path" in block.prompt_params
        assert block.prompt_params["prompt_config_path"] == test_yaml_config

        # Check internal blocks are created
        assert block.prompt_builder is not None
        assert block.llm_chat is not None
        assert block.text_parser is not None
        assert block.filter_block is not None

    def test_init_with_invalid_input_cols(self, test_yaml_config):
        """Test initialization with invalid input columns."""
        with pytest.raises(ValueError, match="expects input_cols"):
            EvaluateFaithfulnessBlock(
                block_name="test_faithfulness",
                input_cols=["wrong", "columns"],
                output_cols=[
                    "faithfulness_explanation",
                    "faithfulness_judgment",
                ],
                prompt_config_path=test_yaml_config,
            )

    def test_init_with_invalid_output_cols(self, test_yaml_config):
        """Test initialization with invalid output columns."""
        with pytest.raises(ValueError, match="expects output_cols"):
            EvaluateFaithfulnessBlock(
                block_name="test_faithfulness",
                input_cols=["document", "response"],
                output_cols=["wrong", "columns"],
                prompt_config_path=test_yaml_config,
            )

    def test_get_internal_blocks_info(self, test_yaml_config):
        """Test getting information about internal blocks."""
        block = EvaluateFaithfulnessBlock(
            block_name="test_faithfulness",
            input_cols=["document", "response"],
            output_cols=[
                "faithfulness_explanation", 
                "faithfulness_judgment"
            ],
            prompt_config_path=test_yaml_config,
            model="openai/gpt-4",
        )

        info = block.get_internal_blocks_info()
        assert "prompt_builder" in info
        assert "llm_chat" in info
        assert "text_parser" in info
        assert "filter" in info

    def test_repr_method(self, test_yaml_config):
        """Test string representation."""
        block = EvaluateFaithfulnessBlock(
            block_name="test_faithfulness",
            input_cols=["document", "response"],
            output_cols=[
                "faithfulness_explanation",
                "faithfulness_judgment",
            ],
            prompt_config_path=test_yaml_config,
            model="openai/gpt-4",
        )

        repr_str = repr(block)
        assert "EvaluateFaithfulnessBlock" in repr_str
        assert "test_faithfulness" in repr_str
        assert "openai/gpt-4" in repr_str

    def test_generate_with_model_validation(self, test_yaml_config):
        """Test that generate method validates model is configured."""
        block = EvaluateFaithfulnessBlock(
            block_name="test_faithfulness",
            input_cols=["document", "response"],
            output_cols=[
                "faithfulness_explanation",
                "faithfulness_judgment",
            ],
            prompt_config_path=test_yaml_config,
            # No model provided
        )

        # Setup test data
        test_dataset = Dataset.from_list([
            {"document": "test doc", "response": "test response"}
        ])

        # Should raise BlockValidationError
        from sdg_hub.core.utils.error_handling import BlockValidationError
        with pytest.raises(BlockValidationError, match="Model not configured"):
            block.generate(test_dataset)