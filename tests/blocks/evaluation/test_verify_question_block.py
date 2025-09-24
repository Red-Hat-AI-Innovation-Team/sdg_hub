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

    def test_flow_set_model_config_detection(self, test_yaml_config):
        """Test that hasattr() works for Flow.set_model_config() detection.

        This was the core issue - Flow.set_model_config() uses hasattr() to detect
        which blocks support LLM parameters, but composite blocks were returning False.
        """
        block = VerifyQuestionBlock(
            block_name="test_verify",
            input_cols=["question"],
            output_cols=["verification_explanation", "verification_rating"],
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
            # This is the exact check Flow.set_model_config() uses
            assert hasattr(block, param), (
                f"VerifyQuestionBlock must have attribute '{param}' "
                f"for Flow.set_model_config() detection"
            )

    def test_runtime_parameter_forwarding_to_internal_blocks(self, test_yaml_config):
        """Test that runtime parameter updates forward to internal LLM blocks.

        This simulates the exact Flow.set_model_config() workflow and verifies
        that parameters reach the internal LLM blocks correctly.
        """
        block = VerifyQuestionBlock(
            block_name="test_verify",
            input_cols=["question"],
            output_cols=["verification_explanation", "verification_rating"],
            prompt_config_path=test_yaml_config,
        )

        # Simulate exact Flow.set_model_config() parameters from user's issue
        test_params = {
            "model": "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            "api_base": "http://localhost:9000/v1",
            "api_key": "EMPTY",
            "extra_headers": {"XXX": "YYY"},
            "extra_body": {"guided_choice": ["YES", "NO"]},
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        # Step 1: Flow.set_model_config() checks hasattr() - must pass
        for param_name in test_params:
            assert hasattr(block, param_name), f"hasattr check failed for {param_name}"

        # Step 2: Flow.set_model_config() sets parameters - must work
        for param_name, param_value in test_params.items():
            setattr(block, param_name, param_value)

        # Step 3: Composite block must have the parameters accessible
        for param_name, expected_value in test_params.items():
            actual_value = getattr(block, param_name)
            assert (
                actual_value == expected_value
            ), f"Composite block {param_name}: expected {expected_value}, got {actual_value}"

        # Step 4: CRITICAL - Internal LLM block must receive the parameters
        for param_name, expected_value in test_params.items():
            internal_value = getattr(block.llm_chat, param_name)
            assert (
                internal_value == expected_value
            ), f"Internal LLM block {param_name}: expected {expected_value}, got {internal_value}"

    def test_meaningful_defaults_are_provided(self, test_yaml_config):
        """Test that meaningful defaults are provided for required internal block parameters.

        Users should be able to create composite blocks without specifying every parameter,
        and the blocks should provide sensible defaults for their specific use cases.
        """
        # Test that blocks can be created with minimal parameters
        block = VerifyQuestionBlock(
            block_name="test_verify",
            input_cols=["question"],
            output_cols=["verification_explanation", "verification_rating"],
            prompt_config_path=test_yaml_config,
            # No filter/parser params specified - should use meaningful defaults
        )

        # Verify meaningful defaults for VerifyQuestionBlock
        assert (
            block.filter_value == 1.0
        ), "VerifyQuestionBlock should default to rating 1.0"
        assert (
            block.operation == "eq"
        ), "VerifyQuestionBlock should default to 'eq' operation"
        assert (
            block.convert_dtype == "float"
        ), "VerifyQuestionBlock should default to float conversion"
        assert block.start_tags == [
            "[Start of Explanation]",
            "[Start of Rating]",
        ]
        assert block.end_tags == ["[End of Explanation]", "[End of Rating]"]

        # Test that defaults are properly forwarded to internal blocks
        assert block.filter_block.filter_value == 1.0
        assert block.filter_block.operation == "eq"
        assert block.text_parser.start_tags == [
            "[Start of Explanation]",
            "[Start of Rating]",
        ]

    def test_parameter_overrides_work_correctly(self, test_yaml_config):
        """Test that user-provided parameters override defaults correctly.

        When users provide explicit parameters, they should take precedence over defaults,
        and both initialization-time and runtime parameter setting should work.
        """
        # Test initialization-time parameter override
        block = VerifyQuestionBlock(
            block_name="test_verify",
            input_cols=["question"],
            output_cols=["verification_explanation", "verification_rating"],
            prompt_config_path=test_yaml_config,
            # Override defaults
            filter_value=0.8,
            operation="ge",
            start_tags=["<custom_explanation>", "<custom_rating>"],
            temperature=0.9,
            extra_body={"init_param": "value"},
        )

        # Verify overrides worked
        assert block.filter_value == 0.8, "Initialization override failed"
        assert block.operation == "ge", "Initialization override failed"
        assert block.start_tags == ["<custom_explanation>", "<custom_rating>"]
        assert block.temperature == 0.9
        assert block.extra_body == {"init_param": "value"}

        # Verify overrides forwarded to internal blocks
        assert block.filter_block.filter_value == 0.8
        assert block.filter_block.operation == "ge"
        assert block.text_parser.start_tags == [
            "<custom_explanation>",
            "<custom_rating>",
        ]
        assert block.llm_chat.temperature == 0.9
        assert block.llm_chat.extra_body == {"init_param": "value"}

        # Test runtime parameter override (simulating Flow.set_model_config)
        block.filter_value = 0.5
        block.temperature = 0.3
        block.extra_body = {"runtime_param": "new_value"}

        # Verify runtime overrides worked
        assert block.filter_value == 0.5, "Runtime override failed"
        assert block.temperature == 0.3, "Runtime override failed"
        assert block.extra_body == {"runtime_param": "new_value"}

        # Verify runtime overrides forwarded to internal blocks
        assert block.filter_block.filter_value == 0.5
        assert block.llm_chat.temperature == 0.3
        assert block.llm_chat.extra_body == {"runtime_param": "new_value"}
