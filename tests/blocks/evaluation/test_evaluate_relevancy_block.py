# SPDX-License-Identifier: Apache-2.0
"""Tests for EvaluateRelevancyBlock."""

# Standard
from unittest.mock import MagicMock, patch
import os
import tempfile

# Third Party
from datasets import Dataset

# First Party
from sdg_hub import BlockRegistry
from sdg_hub.core.blocks.evaluation.evaluate_relevancy_block import (
    EvaluateRelevancyBlock,
)
import pytest


class TestEvaluateRelevancyBlock:
    """Test cases for EvaluateRelevancyBlock."""

    @pytest.fixture
    def test_yaml_config(self):
        """Create a temporary YAML config file for testing."""
        yaml_content = """- role: "user"
  content: |
    Please evaluate the relevancy of the following response to the given question.
    
    Question: {{ question }}
    
    Response: {{ response }}
    
    Please provide your evaluation in the following format:
    
    [Start of Feedback]
    Provide feedback on the relevancy.
    [End of Feedback]
    
    [Start of Score]
    2.0
    [End of Score]"""

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        temp_file.write(yaml_content)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)

    def test_block_registry(self):
        """Test that EvaluateRelevancyBlock is properly registered."""
        block_class = BlockRegistry._get("EvaluateRelevancyBlock")
        assert block_class == EvaluateRelevancyBlock

        # Check category
        eval_blocks = BlockRegistry.list_blocks(category="evaluation")
        assert "EvaluateRelevancyBlock" in eval_blocks

    def test_init_with_valid_params(self, test_yaml_config):
        """Test initialization with valid parameters."""
        block = EvaluateRelevancyBlock(
            block_name="test_relevancy",
            input_cols=["question", "response"],
            output_cols=[
                "relevancy_explanation",
                "relevancy_score",
            ],
            prompt_config_path=test_yaml_config,
            model="openai/gpt-4",
            start_tags=["[Start of Feedback]", "[Start of Score]"],
            end_tags=["[End of Feedback]", "[End of Score]"],
        )

        assert block.block_name == "test_relevancy"
        assert block.input_cols == ["question", "response"]
        assert block.output_cols == [
            "relevancy_explanation",
            "relevancy_score",
        ]

    def test_init_with_invalid_input_cols(self, test_yaml_config):
        """Test initialization with invalid input columns."""
        with pytest.raises(ValueError, match="expects input_cols"):
            EvaluateRelevancyBlock(
                block_name="test_relevancy",
                input_cols=["wrong", "columns"],
                output_cols=[
                    "relevancy_explanation",
                    "relevancy_score",
                ],
                prompt_config_path=test_yaml_config,
            )

    def test_init_with_invalid_output_cols(self, test_yaml_config):
        """Test initialization with invalid output columns."""
        with pytest.raises(ValueError, match="expects output_cols"):
            EvaluateRelevancyBlock(
                block_name="test_relevancy",
                input_cols=["question", "response"],
                output_cols=["wrong", "columns"],
                prompt_config_path=test_yaml_config,
            )
