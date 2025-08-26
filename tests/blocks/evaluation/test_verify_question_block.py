# SPDX-License-Identifier: Apache-2.0
"""Tests for VerifyQuestionBlock."""

# Standard
from unittest.mock import MagicMock, patch
import os
import tempfile

# Third Party
from datasets import Dataset

# First Party
from sdg_hub import BlockRegistry
from sdg_hub.core.blocks.evaluation.verify_question_block import VerifyQuestionBlock
import pytest


class TestVerifyQuestionBlock:
    """Test cases for VerifyQuestionBlock."""

    @pytest.fixture
    def test_yaml_config(self):
        """Create a temporary YAML config file for testing."""
        yaml_content = """- role: "user"
  content: |
    Please evaluate the quality and validity of the following question.
    
    Question: {{ question }}
    
    Please provide your evaluation in the following format:
    
    [Start of Explanation]
    Provide a detailed explanation of why the question is or is not well-formed and valid.
    [End of Explanation]
    
    [Start of Rating]
    2.0 (for high quality) or 1.0 (for acceptable quality) or 0.0 (for poor quality)
    [End of Rating]"""

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
                "question": [
                    "What color is the sky?",
                    "What is the meaning of life?",
                ],
            }
        )

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
            block_name="test_verify_question",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
            ],
            prompt_config_path=test_yaml_config,
            model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
        )

        assert block.block_name == "test_verify_question"
        assert block.input_cols == ["question"]
        assert block.output_cols == [
            "verification_explanation",
            "verification_rating",
        ]
        assert block.model == "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"
        assert block.filter_value == 1.0  # default
        assert block.operation == "ge"  # default
        assert block.convert_dtype == "float"  # default

        # Check internal blocks are created
        assert block.prompt_builder is not None
        assert block.llm_chat is not None
        assert block.text_parser is not None
        assert block.filter_block is not None

    def test_init_with_custom_params(self, test_yaml_config):
        """Test initialization with custom parameters."""
        block = VerifyQuestionBlock(
            block_name="custom_verify_question",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
            ],
            prompt_config_path=test_yaml_config,
            model="openai/gpt-4",
            filter_value=1.5,
            operation="ge",
            convert_dtype="float",
            async_mode=False,
            temperature=0.7,
            max_tokens=1024,
            start_tags=["<explanation>", "<rating>"],
            end_tags=["</explanation>", "</rating>"],
        )

        assert block.filter_value == 1.5
        assert block.operation == "ge"
        assert block.convert_dtype == "float"
        assert block.async_mode is False
        assert block.temperature == 0.7
        assert block.max_tokens == 1024
        assert block.start_tags == ["<explanation>", "<rating>"]
        assert block.end_tags == ["</explanation>", "</rating>"]

    def test_init_with_invalid_input_cols(self, test_yaml_config):
        """Test initialization with invalid input columns."""
        with pytest.raises(ValueError, match="VerifyQuestionBlock expects input_cols"):
            VerifyQuestionBlock(
                block_name="test_verify_question",
                input_cols=["wrong", "columns"],
                output_cols=[
                    "verification_explanation",
                    "verification_rating",
                ],
                prompt_config_path=test_yaml_config,
                model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            )

    def test_init_with_invalid_output_cols(self, test_yaml_config):
        """Test initialization with invalid output columns."""
        with pytest.raises(ValueError, match="VerifyQuestionBlock expects output_cols"):
            VerifyQuestionBlock(
                block_name="test_verify_question",
                input_cols=["question"],
                output_cols=["wrong", "columns"],
                prompt_config_path=test_yaml_config,
                model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            )

    def test_init_with_invalid_yaml(self):
        """Test initialization with invalid YAML file."""
        with pytest.raises(FileNotFoundError):
            VerifyQuestionBlock(
                block_name="test_verify_question",
                input_cols=["question"],
                output_cols=[
                    "verification_explanation",
                    "verification_rating",
                ],
                prompt_config_path="/nonexistent/path.yaml",
                model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            )

    def test_validate_custom_with_valid_dataset(self, test_yaml_config, sample_dataset):
        """Test _validate_custom with valid dataset."""
        block = VerifyQuestionBlock(
            block_name="test_verify_question",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
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
        block = VerifyQuestionBlock(
            block_name="test_verify_question",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
            ],
            prompt_config_path=test_yaml_config,
            model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
        )

        # Dataset missing required columns
        invalid_dataset = Dataset.from_dict({"wrong_column": ["value"]})

        with pytest.raises(ValueError, match="VerifyQuestionBlock requires columns"):
            block._validate_custom(invalid_dataset)

    def test_validate_custom_with_uninitialized_blocks(
        self, test_yaml_config, sample_dataset
    ):
        """Test _validate_custom with uninitialized internal blocks."""
        block = VerifyQuestionBlock(
            block_name="test_verify_question",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
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

    @patch("sdg_hub.core.blocks.evaluation.verify_question_block.LLMChatBlock")
    @patch("sdg_hub.core.blocks.evaluation.verify_question_block.PromptBuilderBlock")
    @patch("sdg_hub.core.blocks.evaluation.verify_question_block.TextParserBlock")
    @patch(
        "sdg_hub.core.blocks.evaluation.verify_question_block.ColumnValueFilterBlock"
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
            "verify_question_prompt",
            [
                [{"role": "user", "content": "test prompt 1"}],
                [{"role": "user", "content": "test prompt 2"}],
            ],
        )
        step2_dataset = step1_dataset.add_column(
            "raw_verify_question",
            [
                "[Start of Explanation]Good question[End of Explanation]\n[Start of Rating]2.0[End of Rating]",
                "[Start of Explanation]Poor question[End of Explanation]\n[Start of Rating]0.0[End of Rating]",
            ],
        )
        step3_dataset = step2_dataset.add_column(
            "verification_explanation", ["Good question", "Poor question"]
        ).add_column("verification_rating", ["2.0", "0.0"])
        step4_dataset = Dataset.from_dict(
            {
                "question": ["What color is the sky?"],
                "verify_question_prompt": [
                    [{"role": "user", "content": "test prompt 1"}]
                ],
                "raw_verify_question": [
                    "[Start of Explanation]Good question[End of Explanation]\n[Start of Rating]2.0[End of Rating]"
                ],
                "verification_explanation": ["Good question"],
                "verification_rating": ["2.0"],
            }
        )

        mock_prompt_instance.generate.return_value = step1_dataset
        mock_llm_instance.generate.return_value = step2_dataset
        mock_parser_instance.generate.return_value = step3_dataset
        mock_filter_instance.generate.return_value = step4_dataset

        # Create block and call generate
        block = VerifyQuestionBlock(
            block_name="test_verify_question",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
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
        assert "verification_explanation" in result.column_names
        assert "verification_rating" in result.column_names

    def test_get_internal_blocks_info(self, test_yaml_config):
        """Test get_internal_blocks_info method."""
        block = VerifyQuestionBlock(
            block_name="test_verify_question",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
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
        block = VerifyQuestionBlock(
            block_name="test_verify_question",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
            ],
            prompt_config_path=test_yaml_config,
            model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
        )

        info = block.get_info()

        assert info["block_name"] == "test_verify_question"
        assert info["block_type"] == "VerifyQuestionBlock"
        assert info["input_cols"] == ["question"]
        assert info["output_cols"] == [
            "verification_explanation",
            "verification_rating",
        ]
        assert info["model"] == "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"

    def test_repr_method(self, test_yaml_config):
        """Test __repr__ method."""
        block = VerifyQuestionBlock(
            block_name="test_verify_question",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
            ],
            prompt_config_path=test_yaml_config,
            model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            filter_value=1.5,
        )

        repr_str = repr(block)
        assert "VerifyQuestionBlock" in repr_str
        assert "test_verify_question" in repr_str
        assert "1.5" in repr_str

    def test_internal_block_configuration(self, test_yaml_config):
        """Test that internal blocks are configured correctly."""
        block = VerifyQuestionBlock(
            block_name="test_verify_question",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
            ],
            prompt_config_path=test_yaml_config,
            model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
            temperature=0.8,
            max_tokens=512,
            filter_value=1.5,
            operation="ge",
        )

        # Check PromptBuilderBlock configuration
        assert block.prompt_builder.block_name == "test_verify_question_prompt_builder"
        assert block.prompt_builder.input_cols == ["question"]
        assert block.prompt_builder.output_cols == ["verify_question_prompt"]

        # Check LLMChatBlock configuration
        assert block.llm_chat.block_name == "test_verify_question_llm_chat"
        assert block.llm_chat.model == "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"
        assert block.llm_chat.temperature == 0.8
        assert block.llm_chat.max_tokens == 512

        # Check TextParserBlock configuration
        assert block.text_parser.block_name == "test_verify_question_text_parser"
        assert block.text_parser.input_cols == ["raw_verify_question"]
        assert block.text_parser.output_cols == [
            "verification_explanation",
            "verification_rating",
        ]

        # Check ColumnValueFilterBlock configuration
        assert block.filter_block.block_name == "test_verify_question_filter"
        assert block.filter_block.input_cols == ["verification_rating"]
        assert block.filter_block.filter_value == 1.5
        assert block.filter_block.operation == "ge"

    def test_error_handling_in_generate(self, test_yaml_config, sample_dataset):
        """Test error handling in generate method."""
        block = VerifyQuestionBlock(
            block_name="test_verify_question",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
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
        block = VerifyQuestionBlock(
            block_name="test_verify_question",
            input_cols=["question"],
            output_cols=[
                "verification_explanation",
                "verification_rating",
            ],
            prompt_config_path=test_yaml_config,
            model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
        )

        # Create empty dataset with required columns
        empty_dataset = Dataset.from_dict({"question": []})

        # Should not raise exception even with empty dataset
        block._validate_custom(empty_dataset)
