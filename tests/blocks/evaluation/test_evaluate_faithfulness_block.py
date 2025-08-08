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

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        return Dataset.from_dict(
            {
                "document": [
                    "The sky is blue due to Rayleigh scattering.",
                    "Water boils at 100°C at sea level.",
                ],
                "response": [
                    "The sky appears blue because of light scattering.",
                    "Water reaches boiling point at 100 degrees Celsius.",
                ],
            }
        )

    def test_block_registry(self):
        """Test that EvaluateFaithfulnessBlock is properly registered."""
        block_class = BlockRegistry.get("EvaluateFaithfulnessBlock")
        assert block_class == EvaluateFaithfulnessBlock

        # Check category
        eval_blocks = BlockRegistry.category("evaluation")
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

        assert block.block_name == "test_faithfulness"
        assert block.input_cols == ["document", "response"]
        assert block.output_cols == [
            "faithfulness_explanation",
            "faithfulness_judgment",
        ]
        assert block.model == "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"
        assert block.filter_value == "YES"  # default
        assert block.operation == "eq"  # default

        # Check internal blocks are created
        assert block.prompt_builder is not None
        assert block.llm_chat is not None
        assert block.text_parser is not None
        assert block.filter_block is not None

    def test_init_with_custom_params(self, test_yaml_config):
        """Test initialization with custom parameters."""
        block = EvaluateFaithfulnessBlock(
            block_name="custom_faithfulness",
            input_cols=["document", "response"],
            output_cols=[
                "faithfulness_explanation",
                "faithfulness_judgment",
            ],
            prompt_config_path=test_yaml_config,
            model="openai/gpt-4",
            filter_value="FAITHFUL",
            operation="ne",
            convert_dtype="float",  # Use supported dtype
            async_mode=False,
            temperature=0.7,
            max_tokens=1024,
            start_tags=["<explanation>", "<judgment>"],
            end_tags=["</explanation>", "</judgment>"],
        )

        assert block.filter_value == "FAITHFUL"
        assert block.operation == "ne"
        assert block.convert_dtype == "float"
        assert block.async_mode is False
        assert block.temperature == 0.7
        assert block.max_tokens == 1024
        assert block.start_tags == ["<explanation>", "<judgment>"]
        assert block.end_tags == ["</explanation>", "</judgment>"]

    def test_init_with_invalid_input_cols(self, test_yaml_config):
        """Test initialization with invalid input columns."""
        with pytest.raises(
            ValueError, match="EvaluateFaithfulnessBlock expects input_cols"
        ):
            EvaluateFaithfulnessBlock(
                block_name="test_faithfulness",
                input_cols=["wrong", "columns"],
                output_cols=[
                    "faithfulness_explanation",
                    "faithfulness_judgment",
                ],
                prompt_config_path=test_yaml_config,
                model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            )

    def test_init_with_invalid_output_cols(self, test_yaml_config):
        """Test initialization with invalid output columns."""
        with pytest.raises(
            ValueError, match="EvaluateFaithfulnessBlock expects output_cols"
        ):
            EvaluateFaithfulnessBlock(
                block_name="test_faithfulness",
                input_cols=["document", "response"],
                output_cols=["wrong", "columns"],
                prompt_config_path=test_yaml_config,
                model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            )

    def test_init_with_invalid_yaml(self):
        """Test initialization with invalid YAML file."""
        with pytest.raises(FileNotFoundError):
            EvaluateFaithfulnessBlock(
                block_name="test_faithfulness",
                input_cols=["document", "response"],
                output_cols=[
                    "faithfulness_explanation",
                    "faithfulness_judgment",
                ],
                prompt_config_path="/nonexistent/path.yaml",
                model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            )

    def test_validate_custom_with_valid_dataset(self, test_yaml_config, sample_dataset):
        """Test _validate_custom with valid dataset."""
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

        # Should not raise any exception
        block._validate_custom(sample_dataset)

    def test_validate_custom_with_missing_columns(self, test_yaml_config):
        """Test _validate_custom with missing required columns."""
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

        # Dataset missing required columns
        invalid_dataset = Dataset.from_dict({"wrong_column": ["value"]})

        with pytest.raises(
            ValueError, match="EvaluateFaithfulnessBlock requires columns"
        ):
            block._validate_custom(invalid_dataset)

    def test_validate_custom_with_uninitialized_blocks(
        self, test_yaml_config, sample_dataset
    ):
        """Test _validate_custom with uninitialized internal blocks."""
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

        # Manually set one of the internal blocks to None
        block.prompt_builder = None

        with pytest.raises(ValueError, match="All internal blocks must be initialized"):
            block._validate_custom(sample_dataset)

    @patch("sdg_hub.core.blocks.evaluation.evaluate_faithfulness_block.LLMChatBlock")
    @patch(
        "sdg_hub.core.blocks.evaluation.evaluate_faithfulness_block.PromptBuilderBlock"
    )
    @patch("sdg_hub.core.blocks.evaluation.evaluate_faithfulness_block.TextParserBlock")
    @patch(
        "sdg_hub.core.blocks.evaluation.evaluate_faithfulness_block.ColumnValueFilterBlock"
    )
    def test_generate_method_calls_internal_blocks(
        self,
        mock_filter,
        mock_parser,
        mock_prompt,
        mock_llm,
        test_yaml_config,
        sample_dataset,
    ):
        """Test that generate method calls all internal blocks in correct order."""
        # Set up mocks to return expected datasets
        mock_prompt_instance = MagicMock()
        mock_llm_instance = MagicMock()
        mock_parser_instance = MagicMock()
        mock_filter_instance = MagicMock()

        mock_prompt.return_value = mock_prompt_instance
        mock_llm.return_value = mock_llm_instance
        mock_parser.return_value = mock_parser_instance
        mock_filter.return_value = mock_filter_instance

        # Mock the generate methods to return progressive datasets
        step1_dataset = sample_dataset.add_column(
            "eval_faithfulness_prompt",
            [
                [{"role": "user", "content": "test prompt 1"}],
                [{"role": "user", "content": "test prompt 2"}],
            ],
        )
        step2_dataset = step1_dataset.add_column(
            "raw_eval_faithfulness",
            [
                "[Start of Explanation]Good explanation[End of Explanation]\n[Start of Answer]YES[End of Answer]",
                "[Start of Explanation]Bad explanation[End of Explanation]\n[Start of Answer]NO[End of Answer]",
            ],
        )
        step3_dataset = step2_dataset.add_column(
            "faithfulness_explanation", ["Good explanation", "Bad explanation"]
        ).add_column("faithfulness_judgment", ["YES", "NO"])
        step4_dataset = Dataset.from_dict(
            {
                "document": ["The sky is blue due to Rayleigh scattering."],
                "response": ["The sky appears blue because of light scattering."],
                "eval_faithfulness_prompt": [
                    [{"role": "user", "content": "test prompt 1"}]
                ],
                "raw_eval_faithfulness": [
                    "[Start of Explanation]Good explanation[End of Explanation]\n[Start of Answer]YES[End of Answer]"
                ],
                "faithfulness_explanation": ["Good explanation"],
                "faithfulness_judgment": ["YES"],
            }
        )

        mock_prompt_instance.generate.return_value = step1_dataset
        mock_llm_instance.generate.return_value = step2_dataset
        mock_parser_instance.generate.return_value = step3_dataset
        mock_filter_instance.generate.return_value = step4_dataset

        # Create block and call generate
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

        result = block.generate(sample_dataset)

        # Verify all internal blocks were called in order
        mock_prompt_instance.generate.assert_called_once()
        mock_llm_instance.generate.assert_called_once()
        mock_parser_instance.generate.assert_called_once()
        mock_filter_instance.generate.assert_called_once()

        # Verify result contains expected columns
        assert "faithfulness_explanation" in result.column_names
        assert "faithfulness_judgment" in result.column_names

    def test_get_internal_blocks_info(self, test_yaml_config):
        """Test get_internal_blocks_info method."""
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

        info = block.get_internal_blocks_info()

        assert "prompt_builder" in info
        assert "llm_chat" in info
        assert "text_parser" in info
        assert "filter" in info

        # Check that each internal block info is not None
        assert info["prompt_builder"] is not None
        assert info["llm_chat"] is not None
        assert info["text_parser"] is not None
        assert info["filter"] is not None

    def test_get_info_method(self, test_yaml_config):
        """Test get_info method from BaseBlock."""
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

        info = block.get_info()

        assert info["block_name"] == "test_faithfulness"
        assert info["block_type"] == "EvaluateFaithfulnessBlock"
        assert info["input_cols"] == ["document", "response"]
        assert info["output_cols"] == [
            "faithfulness_explanation",
            "faithfulness_judgment",
        ]
        assert info["model"] == "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"

    def test_repr_method(self, test_yaml_config):
        """Test __repr__ method."""
        block = EvaluateFaithfulnessBlock(
            block_name="test_faithfulness",
            input_cols=["document", "response"],
            output_cols=[
                "faithfulness_explanation",
                "faithfulness_judgment",
            ],
            prompt_config_path=test_yaml_config,
            model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            filter_value="FAITHFUL",
        )

        repr_str = repr(block)
        assert "EvaluateFaithfulnessBlock" in repr_str
        assert "test_faithfulness" in repr_str
        assert "FAITHFUL" in repr_str

    def test_internal_block_configuration(self, test_yaml_config):
        """Test that internal blocks are configured correctly."""
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
            temperature=0.8,
            max_tokens=512,
            filter_value="FAITHFUL",
            operation="ne",
        )

        # Check PromptBuilderBlock configuration
        assert block.prompt_builder.block_name == "test_faithfulness_prompt_builder"
        assert block.prompt_builder.input_cols == ["document", "response"]
        assert block.prompt_builder.output_cols == ["eval_faithfulness_prompt"]

        # Check LLMChatBlock configuration
        assert block.llm_chat.block_name == "test_faithfulness_llm_chat"
        assert block.llm_chat.model == "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"
        assert block.llm_chat.temperature == 0.8
        assert block.llm_chat.max_tokens == 512

        # Check TextParserBlock configuration
        assert block.text_parser.block_name == "test_faithfulness_text_parser"
        assert block.text_parser.input_cols == ["raw_eval_faithfulness"]
        assert block.text_parser.output_cols == [
            "faithfulness_explanation",
            "faithfulness_judgment",
        ]

        # Check ColumnValueFilterBlock configuration
        assert block.filter_block.block_name == "test_faithfulness_filter"
        assert block.filter_block.input_cols == ["faithfulness_judgment"]
        assert block.filter_block.filter_value == "FAITHFUL"
        assert block.filter_block.operation == "ne"

    def test_error_handling_in_generate(self, test_yaml_config, sample_dataset):
        """Test error handling in generate method."""
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

        # Mock the prompt builder to raise an exception
        with (
            patch.object(
                block.prompt_builder, "generate", side_effect=Exception("Test error")
            ),
            pytest.raises(Exception, match="Test error"),
        ):
            block.generate(sample_dataset)

    def test_validation_with_empty_dataset(self, test_yaml_config):
        """Test validation with empty dataset."""
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

        # Create empty dataset with required columns
        empty_dataset = Dataset.from_dict({"document": [], "response": []})

        # Should not raise exception even with empty dataset
        block._validate_custom(empty_dataset)
