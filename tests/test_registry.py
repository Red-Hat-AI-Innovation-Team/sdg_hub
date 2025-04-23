import unittest
from unittest.mock import patch, MagicMock
from sdg_hub.registry import PromptRegistry  # Assuming Registry is the class name

class TestRegistry(unittest.TestCase):
    def setUp(self):
        # Clear the LRU cache before each test
        PromptRegistry.template_from_model.cache_clear()

    def test_template_from_registered_model(self):
        """Test successful retrieval of chat template from a known model."""

        # return the template as a jinja2 template
        template = PromptRegistry.get_template("blank")
        # get the template string
        template_string = template.render(messages=[{"role": "user", "content": "test"}])
        self.assertEqual(template_string, "[{'role': 'user', 'content': 'test'}]")

    def test_template_from_model_successful(self):
        """Test successful retrieval of chat template from a known model."""
        test_model = "test_model"
        with patch("sdg_hub.registry.AutoTokenizer") as mock_tokenizer:
            # Setup mock
            mock_instance = MagicMock()
            mock_instance.chat_template = "{{ messages[0]['content'] }}"
            mock_tokenizer.from_pretrained.return_value = mock_instance

            # Test the method
            template = PromptRegistry.template_from_model(test_model)
            template_string = template.render(messages=[{"role": "user", "content": "test"}])

            # Assertions
            self.assertEqual(template_string, "test")
            mock_tokenizer.from_pretrained.assert_called_once_with(test_model)

    def test_template_from_model_caching(self):
        """Test that the LRU cache is working properly."""
        test_model = "test_model"
        with patch("sdg_hub.registry.AutoTokenizer") as mock_tokenizer:
            # Setup mock
            mock_instance = MagicMock()
            mock_instance.chat_template = "test template"
            mock_tokenizer.from_pretrained.return_value = mock_instance

            # Call the method twice
            template1 = PromptRegistry.template_from_model(test_model)
            template2 = PromptRegistry.template_from_model(test_model)

            # Assertions
            self.assertEqual(template1, template2)
            # Should only be called once due to caching
            mock_tokenizer.from_pretrained.assert_called_once_with(test_model)

    def test_template_from_model_invalid_model(self):
        """Test handling of invalid model names."""
        invalid_model = "nonexistent-model"
        with patch("sdg_hub.registry.AutoTokenizer") as mock_tokenizer:
            mock_tokenizer.from_pretrained.side_effect = OSError("Model not found")

            # Test that it raises an exception
            with self.assertRaises(OSError):
                PromptRegistry.template_from_model(invalid_model)

if __name__ == "__main__":
    unittest.main()
