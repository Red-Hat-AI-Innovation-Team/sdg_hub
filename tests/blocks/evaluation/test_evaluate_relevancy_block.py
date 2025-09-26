# SPDX-License-Identifier: Apache-2.0
"""Tests for EvaluateRelevancyBlock."""

# Standard
import os
import tempfile

# First Party
from sdg_hub import BlockRegistry
from sdg_hub.core.blocks.evaluation.evaluate_relevancy_block import (
    EvaluateRelevancyBlock,
)

# Third Party
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

    def test_flow_set_model_config_detection(self, test_yaml_config):
        """Test that hasattr() works for Flow.set_model_config() detection."""
        block = EvaluateRelevancyBlock(
            block_name="test_relevancy",
            input_cols=["question", "response"],
            output_cols=["relevancy_explanation", "relevancy_score"],
            prompt_config_path=test_yaml_config,
        )

        # Critical parameters that were failing before
        critical_params = [
            "model",
            "api_base",
            "api_key",
            "extra_body",
            "extra_headers",
            "temperature",
            "max_tokens",
            "top_p",
        ]

        for param in critical_params:
            assert hasattr(block, param), (
                f"EvaluateRelevancyBlock must have attribute '{param}' "
                f"for Flow.set_model_config() detection"
            )

    def test_runtime_parameter_forwarding_to_internal_blocks(self, test_yaml_config):
        """Test that runtime parameter updates forward to internal LLM blocks."""

        block = EvaluateRelevancyBlock(
            block_name="test_relevancy",
            input_cols=["question", "response"],
            output_cols=["relevancy_explanation", "relevancy_score"],
            prompt_config_path=test_yaml_config,
        )

        test_params = {
            "model": "openai/gpt-4o",
            "api_base": "http://localhost:7000/v1",
            "api_key": "test-key",
            "extra_headers": {"X-Custom": "header"},
            "extra_body": {"test": "value"},
            "temperature": 0.2,
            "max_tokens": 512,
        }

        # Test hasattr, setattr, getattr, and forwarding
        for param_name in test_params:
            assert hasattr(block, param_name)

        for param_name, param_value in test_params.items():
            setattr(block, param_name, param_value)

        for param_name, expected_value in test_params.items():
            # Check composite block
            actual_value = getattr(block, param_name)
            assert actual_value == expected_value

            # Check internal LLM block
            internal_value = getattr(block.llm_chat, param_name)
            assert internal_value == expected_value

    def test_meaningful_defaults_for_relevancy(self, test_yaml_config):
        """Test meaningful defaults specific to relevancy evaluation."""
        block = EvaluateRelevancyBlock(
            block_name="test_relevancy",
            input_cols=["question", "response"],
            output_cols=["relevancy_explanation", "relevancy_score"],
            prompt_config_path=test_yaml_config,
        )

        # Verify meaningful defaults for EvaluateRelevancyBlock
        assert (
            block.filter_value == 2.0
        ), "EvaluateRelevancyBlock should default to score 2.0"
        assert block.operation == "eq"
        assert block.convert_dtype == "float"
        assert block.start_tags == ["[Start of Feedback]", "[Start of Score]"]
        assert block.end_tags == ["[End of Feedback]", "[End of Score]"]

        # Test that defaults are properly forwarded to internal blocks
        assert block.filter_block.filter_value == 2.0
        assert block.filter_block.operation == "eq"
        assert block.text_parser.start_tags == [
            "[Start of Feedback]",
            "[Start of Score]",
        ]
