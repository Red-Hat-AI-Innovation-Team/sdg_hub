"""Edge case tests for composite block parameter forwarding.

Tests error handling, unknown parameters, and edge cases in the parameter
forwarding system for composite blocks.
"""

import pytest
from unittest.mock import patch, MagicMock

# Test imports
from sdg_hub.core.blocks.evaluation.verify_question_block import VerifyQuestionBlock
from sdg_hub.core.blocks.llm.llm_chat_with_parsing_retry_block import LLMChatWithParsingRetryBlock


class TestCompositeBlockParameterEdgeCases:
    """Edge case tests for parameter forwarding."""

    @patch('sdg_hub.core.blocks.llm.prompt_builder_block.PromptTemplateConfig')
    def test_unknown_parameter_handling(self, mock_config):
        """Test that unknown parameters raise appropriate errors."""
        mock_config.return_value = MagicMock()
        
        block = VerifyQuestionBlock(
            block_name="test_verify",
            input_cols=["question"],
            output_cols=["verification_explanation", "verification_rating"],
            prompt_config_path="dummy.yaml",
        )
        
        # hasattr() should return False for unknown parameters
        assert not hasattr(block, "completely_unknown_parameter")
        assert not hasattr(block, "fake_llm_param")
        assert not hasattr(block, "nonexistent_attribute")
        
        # Accessing unknown parameters should raise AttributeError
        with pytest.raises(AttributeError):
            _ = block.completely_unknown_parameter
        
        with pytest.raises(AttributeError):
            _ = block.fake_llm_param
        
        # Our implementation is strict - unknown parameters that aren't in any 
        # internal block's model_fields are not accessible. This is good behavior
        # because it prevents typos and ensures clean parameter forwarding.
        
        # Verify that no unknown parameters end up on internal blocks
        assert not hasattr(block.llm_chat, "completely_unknown_parameter")
        assert not hasattr(block.filter_block, "fake_filter_param")
        assert not hasattr(block.text_parser, "unknown_parser_param")

    @patch('sdg_hub.core.blocks.llm.prompt_builder_block.PromptTemplateConfig')
    def test_none_values_handling(self, mock_config):
        """Test that None values are handled correctly."""
        mock_config.return_value = MagicMock()
        
        # Create block with some None values
        block = VerifyQuestionBlock(
            block_name="test_verify",
            input_cols=["question"],
            output_cols=["verification_explanation", "verification_rating"],
            prompt_config_path="dummy.yaml",
            model=None,
            api_base=None,
            temperature=None,
        )
        
        # None values should be accessible
        assert block.model is None
        assert block.api_base is None
        assert block.temperature is None
        
        # None values should be forwarded to internal blocks
        assert block.llm_chat.model is None
        assert block.llm_chat.api_base is None
        assert block.llm_chat.temperature is None
        
        # Setting None at runtime should work
        block.extra_body = None
        block.extra_headers = None
        
        assert block.extra_body is None
        assert block.extra_headers is None
        assert block.llm_chat.extra_body is None
        assert block.llm_chat.extra_headers is None

    @patch('sdg_hub.core.blocks.llm.prompt_builder_block.PromptTemplateConfig')
    def test_parameter_forwarding_order_independence(self, mock_config):
        """Test that parameter forwarding works regardless of setting order."""
        mock_config.return_value = MagicMock()
        
        block = VerifyQuestionBlock(
            block_name="test_verify",
            input_cols=["question"],
            output_cols=["verification_explanation", "verification_rating"],
            prompt_config_path="dummy.yaml",
        )
        
        # Set parameters in different orders and verify they all work
        test_scenarios = [
            # Scenario 1: LLM params first
            {"model": "test1", "temperature": 0.1, "filter_value": 1.1},
            # Scenario 2: Filter params first  
            {"filter_value": 2.2, "model": "test2", "temperature": 0.2},
            # Scenario 3: Mixed order
            {"temperature": 0.3, "filter_value": 3.3, "extra_body": {"test": 3}, "model": "test3"},
        ]
        
        for scenario in test_scenarios:
            # Reset block state (create new instance)
            block = VerifyQuestionBlock(
                block_name="test_verify",
                input_cols=["question"],
                output_cols=["verification_explanation", "verification_rating"],
                prompt_config_path="dummy.yaml",
            )
            
            # Set parameters in the order specified by scenario
            for param_name, param_value in scenario.items():
                setattr(block, param_name, param_value)
            
            # Verify all parameters are set correctly on composite block
            for param_name, expected_value in scenario.items():
                actual_value = getattr(block, param_name)
                assert actual_value == expected_value, (
                    f"Scenario {scenario}: {param_name} not set correctly on composite block"
                )
            
            # Verify LLM parameters are forwarded to internal LLM block
            llm_params = {"model", "temperature", "extra_body"}
            for param_name, expected_value in scenario.items():
                if param_name in llm_params:
                    internal_value = getattr(block.llm_chat, param_name)
                    assert internal_value == expected_value, (
                        f"Scenario {scenario}: {param_name} not forwarded to internal LLM block"
                    )
            
            # Verify filter parameters are forwarded to internal filter block
            filter_params = {"filter_value"}
            for param_name, expected_value in scenario.items():
                if param_name in filter_params:
                    internal_value = getattr(block.filter_block, param_name)
                    assert internal_value == expected_value, (
                        f"Scenario {scenario}: {param_name} not forwarded to internal filter block"
                    )

    def test_llm_chat_with_parsing_retry_validation_requirements(self):
        """Test that LLMChatWithParsingRetryBlock properly validates parsing requirements."""
        # Should work with parsing_pattern
        block1 = LLMChatWithParsingRetryBlock(
            block_name="test1",
            input_cols=["messages"],
            output_cols=["output"],
            parsing_pattern=r"test",
        )
        assert block1.parsing_pattern == r"test"
        
        # Should work with start_tags/end_tags
        block2 = LLMChatWithParsingRetryBlock(
            block_name="test2", 
            input_cols=["messages"],
            output_cols=["output"],
            start_tags=["<start>"],
            end_tags=["<end>"],
        )
        assert block2.start_tags == ["<start>"]
        assert block2.end_tags == ["<end>"]
        
        # Should work with both (parsing_pattern takes precedence)
        block3 = LLMChatWithParsingRetryBlock(
            block_name="test3",
            input_cols=["messages"],
            output_cols=["output"],
            parsing_pattern=r"test",
            start_tags=["<start>"],
            end_tags=["<end>"],
        )
        assert block3.parsing_pattern == r"test"
        assert block3.start_tags == ["<start>"]

    @patch('sdg_hub.core.blocks.llm.prompt_builder_block.PromptTemplateConfig')
    def test_all_composite_blocks_consistent_behavior(self, mock_config):
        """Test that all composite blocks behave consistently for parameter forwarding."""
        mock_config.return_value = MagicMock()
        
        # Import all composite blocks
        from sdg_hub.core.blocks.evaluation.evaluate_faithfulness_block import EvaluateFaithfulnessBlock
        from sdg_hub.core.blocks.evaluation.evaluate_relevancy_block import EvaluateRelevancyBlock
        
        # Define test configurations for each block
        block_configs = [
            (VerifyQuestionBlock, {
                "input_cols": ["question"],
                "output_cols": ["verification_explanation", "verification_rating"],
                "prompt_config_path": "dummy.yaml",
            }),
            (EvaluateFaithfulnessBlock, {
                "input_cols": ["document", "response"],
                "output_cols": ["faithfulness_explanation", "faithfulness_judgment"],
                "prompt_config_path": "dummy.yaml",
            }),
            (EvaluateRelevancyBlock, {
                "input_cols": ["question", "response"],
                "output_cols": ["relevancy_explanation", "relevancy_score"],
                "prompt_config_path": "dummy.yaml",
            }),
        ]
        
        # Test parameters that should work consistently across all blocks
        test_params = {
            "model": "test-model",
            "api_base": "http://test:8000/v1",
            "api_key": "test-key",
            "temperature": 0.7,
            "max_tokens": 100,
            "extra_body": {"test": "value"},
            "extra_headers": {"X-Test": "header"},
        }
        
        for block_class, config in block_configs:
            block = block_class(
                block_name=f"test_{block_class.__name__.lower()}",
                **config
            )
            
            # Test 1: hasattr() must work for all parameters
            for param_name in test_params:
                assert hasattr(block, param_name), (
                    f"{block_class.__name__} missing hasattr() for {param_name}"
                )
            
            # Test 2: Parameter setting must work
            for param_name, param_value in test_params.items():
                setattr(block, param_name, param_value)
            
            # Test 3: Parameters must be accessible
            for param_name, expected_value in test_params.items():
                actual_value = getattr(block, param_name)
                assert actual_value == expected_value, (
                    f"{block_class.__name__} {param_name}: expected {expected_value}, got {actual_value}"
                )
            
            # Test 4: Parameters must be forwarded to internal LLM block
            for param_name, expected_value in test_params.items():
                internal_value = getattr(block.llm_chat, param_name)
                assert internal_value == expected_value, (
                    f"{block_class.__name__} internal LLM {param_name}: expected {expected_value}, got {internal_value}"
                )
            
            # Test 5: Client manager reinitialization must work
            with patch.object(block.llm_chat, '_reinitialize_client_manager') as mock_reinit:
                block._reinitialize_client_manager()
                mock_reinit.assert_called_once()