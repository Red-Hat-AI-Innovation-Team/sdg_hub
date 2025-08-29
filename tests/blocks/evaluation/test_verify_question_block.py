# SPDX-License-Identifier: Apache-2.0
"""Tests for VerifyQuestionBlock."""

# Standard
import os
import tempfile

# First Party
from sdg_hub import BlockRegistry
from sdg_hub.core.blocks.evaluation.verify_question_block import VerifyQuestionBlock

# Third Party
import pytest


class TestVerifyQuestionBlock:
    """Test cases for VerifyQuestionBlock."""

    @pytest.fixture
    def test_yaml_config(self):
        """Create a temporary YAML config file for testing."""
        yaml_content = """- role: "user"
  content: |
    Please verify the quality of the following question.
    
    Question: {{ question }}
    
    Please provide your evaluation in the following format:
    
    [Start of Explanation]
    Provide explanation of the quality assessment.
    [End of Explanation]
    
    [Start of Rating]
    1.0
    [End of Rating]"""

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        temp_file.write(yaml_content)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)

    def test_block_registry(self):
        """Test that VerifyQuestionBlock is properly registered."""
        block_class = BlockRegistry._get("VerifyQuestionBlock")
        assert block_class == VerifyQuestionBlock

        # Check category
        eval_blocks = BlockRegistry.list_blocks(category="evaluation")
        assert "VerifyQuestionBlock" in eval_blocks

    def test_init_with_valid_params(self, test_yaml_config):
        """Test initialization with valid parameters."""
        block = VerifyQuestionBlock(
            block_name="test_verify",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
            ],
            prompt_config_path=test_yaml_config,
            model="openai/gpt-4",
            start_tags=["[Start of Explanation]", "[Start of Rating]"],
            end_tags=["[End of Explanation]", "[End of Rating]"],
        )

        assert block.block_name == "test_verify"
        assert block.input_cols == ["question"]
        assert block.output_cols == [
            "verification_explanation",
            "verification_rating",
        ]

    def test_init_with_invalid_input_cols(self, test_yaml_config):
        """Test initialization with invalid input columns."""
        with pytest.raises(ValueError, match="expects input_cols"):
            VerifyQuestionBlock(
                block_name="test_verify",
                input_cols=["wrong"],
                output_cols=[
                    "verification_explanation",
                    "verification_rating",
                ],
                prompt_config_path=test_yaml_config,
            )

    def test_init_with_invalid_output_cols(self, test_yaml_config):
        """Test initialization with invalid output columns."""
        with pytest.raises(ValueError, match="expects output_cols"):
            VerifyQuestionBlock(
                block_name="test_verify",
                input_cols=["question"],
                output_cols=["wrong", "columns"],
                prompt_config_path=test_yaml_config,
            )
