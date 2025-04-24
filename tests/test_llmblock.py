import unittest
from unittest.mock import MagicMock, patch

from src.sdg_hub.blocks.llmblock import LLMBlock


class TestLLMBlockParsing(unittest.TestCase):
    """Test cases for the LLMBlock parsing functionality."""

    def setUp(self):
        """Set up a mock LLMBlock for testing."""
        # Create a mock block configuration
        self.mock_config = {
            "start_tags": [""],
            "end_tags": [""],
            "system": "Test system prompt",
            "introduction": "Test introduction",
            "principles": "Test principles",
            "examples": "Test examples",
            "generation": "Test generation",
        }

        # Create a mock client
        self.mock_client = MagicMock()

        # Mock the _load_config and server_supports_batched methods
        with patch.object(LLMBlock, '_load_config', return_value=self.mock_config):
            self.block = LLMBlock(
                block_name="test_block",
                config_path="mock_path",
                client=self.mock_client,
                output_cols=["output"],
            )

    def test_simple_parse(self):
        """Test case when no start and end tags are provided"""
        generated_string = "Test output"
        results = self.block._parse(generated_string)
        self.assertEqual(results["output"], ["Test output"])

    def test_parse_with_newline_end_tags(self):
        """Test parsing when the end tag is a newline."""
        # Update the block config to use explicit newlines as end tags
        with patch.object(self.block, 'block_config', {
            "start_tags": ["Q.", "A."],
            "end_tags": ["\n", "\n"],  # Explicit newline end tags
        }):
            self.block.output_cols = ["question", "answer"]

            # Test input text
            generated_string = """Here's a question and answer pair:

            Q. A question?
            A. An answer.

            Q. Another question?
            A. Another answer."""

            # Call the parse method
            results = self.block._parse(generated_string)

            # Verify that the parsing worked correctly
            self.assertEqual(len(results["question"]), 2, "Should have extracted 2 questions")
            self.assertEqual(len(results["answer"]), 2, "Should have extracted 2 answers")
            self.assertIn("A question?", results["question"][0])
            self.assertIn("An answer.", results["answer"][0])
            self.assertIn("Another question?", results["question"][1])
            self.assertIn("Another answer.", results["answer"][1])
