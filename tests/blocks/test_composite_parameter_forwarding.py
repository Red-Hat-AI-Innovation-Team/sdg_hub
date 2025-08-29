"""Critical unit tests for composite block parameter forwarding.

Tests the fix for the issue where extra_body, extra_headers, and other LLM parameters
were not being properly forwarded from Flow.set_model_config() to internal LLM blocks
in composite blocks.
"""

import pytest
from unittest.mock import patch, MagicMock

# Test imports
from sdg_hub.core.blocks.evaluation.verify_question_block import VerifyQuestionBlock
from sdg_hub.core.blocks.evaluation.evaluate_faithfulness_block import EvaluateFaithfulnessBlock
from sdg_hub.core.blocks.evaluation.evaluate_relevancy_block import EvaluateRelevancyBlock
from sdg_hub.core.blocks.llm.llm_chat_with_parsing_retry_block import LLMChatWithParsingRetryBlock


class TestCompositeBlockParameterForwarding:
    """Critical tests for parameter forwarding in composite blocks."""

    @patch('sdg_hub.core.blocks.llm.prompt_builder_block.PromptTemplateConfig')
    def test_flow_set_model_config_detection(self, mock_config):
        """Critical Test 1: hasattr() must work for Flow.set_model_config() detection.
        
        This was the core issue - Flow.set_model_config() uses hasattr() to detect
        which blocks support LLM parameters, but composite blocks were returning False.
        """
        mock_config.return_value = MagicMock()
        
        # Test all evaluation blocks
        blocks = [
            VerifyQuestionBlock(
                block_name="test_verify",
                input_cols=["question"],
                output_cols=["verification_explanation", "verification_rating"],
                prompt_config_path="dummy.yaml",
            ),
            EvaluateFaithfulnessBlock(
                block_name="test_faithfulness",
                input_cols=["document", "response"],
                output_cols=["faithfulness_explanation", "faithfulness_judgment"],
                prompt_config_path="dummy.yaml",
            ),
            EvaluateRelevancyBlock(
                block_name="test_relevancy",
                input_cols=["question", "response"],
                output_cols=["relevancy_explanation", "relevancy_score"],
                prompt_config_path="dummy.yaml",
            ),
        ]
        
        # Critical parameters that were failing before
        critical_params = [
            "model", "api_base", "api_key", 
            "extra_body", "extra_headers",
            "temperature", "max_tokens", "top_p"
        ]
        
        for block in blocks:
            for param in critical_params:
                # This is the exact check Flow.set_model_config() uses
                assert hasattr(block, param), (
                    f"{block.__class__.__name__} must have attribute '{param}' "
                    f"for Flow.set_model_config() detection"
                )

    @patch('sdg_hub.core.blocks.llm.prompt_builder_block.PromptTemplateConfig')
    def test_runtime_parameter_forwarding_to_internal_blocks(self, mock_config):
        """Critical Test 2: Runtime parameter updates must forward to internal LLM blocks.
        
        This simulates the exact Flow.set_model_config() workflow and verifies
        that parameters reach the internal LLM blocks correctly.
        """
        mock_config.return_value = MagicMock()
        
        block = VerifyQuestionBlock(
            block_name="test_verify",
            input_cols=["question"],
            output_cols=["verification_explanation", "verification_rating"],
            prompt_config_path="dummy.yaml",
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
            assert actual_value == expected_value, (
                f"Composite block {param_name}: expected {expected_value}, got {actual_value}"
            )
        
        # Step 4: CRITICAL - Internal LLM block must receive the parameters
        for param_name, expected_value in test_params.items():
            internal_value = getattr(block.llm_chat, param_name)
            assert internal_value == expected_value, (
                f"Internal LLM block {param_name}: expected {expected_value}, got {internal_value}"
            )
        
        # Step 5: Client manager reinitialization must work
        with patch.object(block.llm_chat, '_reinitialize_client_manager') as mock_reinit:
            block._reinitialize_client_manager()
            mock_reinit.assert_called_once()

    def test_llm_chat_with_parsing_retry_parameter_forwarding(self):
        """Critical Test 3: LLMChatWithParsingRetryBlock parameter forwarding.
        
        This block has a different structure but must follow the same parameter
        forwarding pattern as evaluation blocks.
        """
        block = LLMChatWithParsingRetryBlock(
            block_name="test_retry",
            input_cols=["messages"],
            output_cols=["parsed_output"],
            parsing_pattern=r"test pattern",  # Required for TextParser
            parsing_max_retries=3,
        )
        
        # Test LLM parameters
        llm_params = {
            "model": "test-model",
            "extra_body": {"test": "value"},
            "extra_headers": {"X-Test": "header"},
            "temperature": 0.8,
        }
        
        # Test parser parameters  
        parser_params = {
            "start_tags": ["<start>"],
            "end_tags": ["<end>"],
        }
        
        all_params = {**llm_params, **parser_params}
        
        # hasattr() must work for Flow detection
        for param_name in all_params:
            assert hasattr(block, param_name), (
                f"LLMChatWithParsingRetryBlock must have attribute '{param_name}'"
            )
        
        # Parameter setting must work
        for param_name, param_value in all_params.items():
            setattr(block, param_name, param_value)
        
        # Parameters must be accessible
        for param_name, expected_value in all_params.items():
            actual_value = getattr(block, param_name)
            assert actual_value == expected_value
        
        # LLM parameters must forward to internal LLM block
        for param_name, expected_value in llm_params.items():
            internal_value = getattr(block.llm_chat, param_name)
            assert internal_value == expected_value, (
                f"LLM parameter {param_name} not forwarded to internal LLM block"
            )
        
        # Parser parameters must forward to internal parser block
        for param_name, expected_value in parser_params.items():
            internal_value = getattr(block.text_parser, param_name)
            assert internal_value == expected_value, (
                f"Parser parameter {param_name} not forwarded to internal parser block"
            )

    @patch('sdg_hub.core.blocks.llm.prompt_builder_block.PromptTemplateConfig')
    def test_meaningful_defaults_are_provided(self, mock_config):
        """Critical Test 4: Meaningful defaults must be provided for required internal block parameters.
        
        Users should be able to create composite blocks without specifying every parameter,
        and the blocks should provide sensible defaults for their specific use cases.
        """
        mock_config.return_value = MagicMock()
        
        # Test that blocks can be created with minimal parameters
        verify_block = VerifyQuestionBlock(
            block_name="test_verify",
            input_cols=["question"],
            output_cols=["verification_explanation", "verification_rating"],
            prompt_config_path="dummy.yaml",
            # No filter/parser params specified - should use meaningful defaults
        )
        
        # Verify meaningful defaults for VerifyQuestionBlock
        assert verify_block.filter_value == 1.0, "VerifyQuestionBlock should default to rating 1.0"
        assert verify_block.operation == "eq", "VerifyQuestionBlock should default to 'eq' operation"
        assert verify_block.convert_dtype == "float", "VerifyQuestionBlock should default to float conversion"
        assert verify_block.start_tags == ["[Start of Explanation]", "[Start of Rating]"]
        assert verify_block.end_tags == ["[End of Explanation]", "[End of Rating]"]
        
        # Test that defaults are properly forwarded to internal blocks
        assert verify_block.filter_block.filter_value == 1.0
        assert verify_block.filter_block.operation == "eq" 
        assert verify_block.text_parser.start_tags == ["[Start of Explanation]", "[Start of Rating]"]
        
        faithfulness_block = EvaluateFaithfulnessBlock(
            block_name="test_faithfulness",
            input_cols=["document", "response"],
            output_cols=["faithfulness_explanation", "faithfulness_judgment"],
            prompt_config_path="dummy.yaml",
        )
        
        # Verify meaningful defaults for EvaluateFaithfulnessBlock
        assert faithfulness_block.filter_value == "YES", "EvaluateFaithfulnessBlock should default to 'YES'"
        assert faithfulness_block.operation == "eq"
        assert faithfulness_block.start_tags == ["[Start of Explanation]", "[Start of Answer]"]
        
        relevancy_block = EvaluateRelevancyBlock(
            block_name="test_relevancy", 
            input_cols=["question", "response"],
            output_cols=["relevancy_explanation", "relevancy_score"],
            prompt_config_path="dummy.yaml",
        )
        
        # Verify meaningful defaults for EvaluateRelevancyBlock
        assert relevancy_block.filter_value == 2.0, "EvaluateRelevancyBlock should default to score 2.0"
        assert relevancy_block.operation == "eq"
        assert relevancy_block.convert_dtype == "float"
        assert relevancy_block.start_tags == ["[Start of Feedback]", "[Start of Score]"]

    @patch('sdg_hub.core.blocks.llm.prompt_builder_block.PromptTemplateConfig')
    def test_parameter_overrides_work_correctly(self, mock_config):
        """Critical Test 5: User-provided parameters must override defaults correctly.
        
        When users provide explicit parameters, they should take precedence over defaults,
        and both initialization-time and runtime parameter setting should work.
        """
        mock_config.return_value = MagicMock()
        
        # Test initialization-time parameter override
        block = VerifyQuestionBlock(
            block_name="test_verify",
            input_cols=["question"],
            output_cols=["verification_explanation", "verification_rating"],
            prompt_config_path="dummy.yaml",
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
        assert block.text_parser.start_tags == ["<custom_explanation>", "<custom_rating>"]
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