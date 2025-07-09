# SPDX-License-Identifier: Apache-2.0

"""Tests for different model backend configurations."""

# Standard
import os
import unittest
from unittest.mock import Mock, patch, MagicMock

# Third Party
from datasets import Dataset
from openai import OpenAI

# Local
from sdg_hub.flow import Flow
from sdg_hub.sdg import SDG


class TestModelBackends(unittest.TestCase):
    """Test cases for different model backend configurations."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sample_data = [
            {"document": "This is a test document for synthetic data generation."},
            {"document": "Another test document with different content."}
        ]
        self.dataset = Dataset.from_list(self.sample_data)

    @patch('openai.OpenAI')
    def test_openai_backend_configuration(self, mock_openai: Mock) -> None:
        """Test OpenAI backend configuration."""
        # Mock OpenAI client and responses
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock models list
        mock_model = Mock()
        mock_model.id = "gpt-4"
        mock_client.models.list.return_value.data = [mock_model]
        
        # Test client initialization
        client = OpenAI(
            api_key="test-key",
            base_url="https://api.openai.com/v1"
        )
        
        self.assertIsNotNone(client)
        mock_openai.assert_called_with(
            api_key="test-key",
            base_url="https://api.openai.com/v1"
        )

    @patch('openai.OpenAI')
    def test_vllm_backend_configuration(self, mock_openai: Mock) -> None:
        """Test vLLM backend configuration."""
        # Mock vLLM client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock vLLM model response (with leading slash)
        mock_model = Mock()
        mock_model.id = "/model/meta-llama/Llama-3.1-8B-Instruct"
        mock_client.models.list.return_value.data = [mock_model]
        
        # Test vLLM client initialization
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
        
        self.assertIsNotNone(client)
        mock_openai.assert_called_with(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )

    @patch('openai.OpenAI')
    def test_ollama_backend_configuration(self, mock_openai: Mock) -> None:
        """Test Ollama backend configuration."""
        # Mock Ollama client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock Ollama model response
        mock_model = Mock()
        mock_model.id = "llama3.1"
        mock_client.models.list.return_value.data = [mock_model]
        
        # Test Ollama client initialization
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:11434/v1"
        )
        
        self.assertIsNotNone(client)
        mock_openai.assert_called_with(
            api_key="EMPTY",
            base_url="http://localhost:11434/v1"
        )

    @patch('openai.OpenAI')
    def test_model_selection_logic(self, mock_openai: Mock) -> None:
        """Test model selection logic with different available models."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Test case 1: Specified model is available
        mock_model1 = Mock()
        mock_model1.id = "gpt-4"
        mock_model2 = Mock()
        mock_model2.id = "gpt-3.5-turbo"
        mock_client.models.list.return_value.data = [mock_model1, mock_model2]
        
        models = mock_client.models.list().data
        model_ids = [m.id for m in models]
        
        # Should find the specified model
        specified_model = "gpt-4"
        if specified_model in model_ids:
            selected_model = specified_model
        else:
            selected_model = models[0].id
            
        self.assertEqual(selected_model, "gpt-4")
        
        # Test case 2: Specified model not available, use first available
        specified_model = "non-existent-model"
        if specified_model in model_ids:
            selected_model = specified_model
        else:
            selected_model = models[0].id
            
        self.assertEqual(selected_model, "gpt-4")

    def test_environment_variable_configuration(self) -> None:
        """Test configuration using environment variables."""
        # Test environment variable reading
        test_cases = [
            ("OPENAI_API_KEY", "test-key", "default-key"),
            ("SEED_DATA_PATH", "/path/to/data.json", "default.json"),
            ("OUTPUT_DIR", "./outputs", "./default"),
            ("CHECKPOINT_DIR", "./checkpoints", "./default_checkpoints")
        ]
        
        for env_var, test_value, default_value in test_cases:
            # Test with environment variable set
            with patch.dict(os.environ, {env_var: test_value}):
                value = os.getenv(env_var, default_value)
                self.assertEqual(value, test_value)
            
            # Test with environment variable not set
            with patch.dict(os.environ, {}, clear=True):
                value = os.getenv(env_var, default_value)
                self.assertEqual(value, default_value)

    @patch('openai.OpenAI')
    def test_connection_verification(self, mock_openai: Mock) -> None:
        """Test connection verification for different backends."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Test successful connection
        mock_model = Mock()
        mock_model.id = "test-model"
        mock_client.models.list.return_value.data = [mock_model]
        
        try:
            models = mock_client.models.list()
            connection_successful = True
            available_models = [m.id for m in models.data]
        except Exception:
            connection_successful = False
            available_models = []
        
        self.assertTrue(connection_successful)
        self.assertEqual(available_models, ["test-model"])
        
        # Test failed connection
        mock_client.models.list.side_effect = Exception("Connection failed")
        
        try:
            models = mock_client.models.list()
            connection_successful = True
        except Exception:
            connection_successful = False
        
        self.assertFalse(connection_successful)

    @patch('openai.OpenAI')
    @patch('sdg_hub.flow.Flow.get_flow_from_file')
    def test_config_mapping_logic(self, mock_flow: Mock, mock_openai: Mock) -> None:
        """Test configuration file mapping based on model type."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_flow.return_value = Mock()
        
        # Test config mapping
        config_map = {
            "gpt-4": "synth_knowledge1.5_llama3.3.yaml",
            "gpt-3.5-turbo": "synth_knowledge1.5_llama3.3.yaml",
            "meta-llama/Llama-3.3-70B-Instruct": "synth_knowledge1.5_llama3.3.yaml",
            "llama3.1": "synth_knowledge1.5_llama3.3.yaml"
        }
        
        # Test different model types
        test_models = ["gpt-4", "gpt-3.5-turbo", "unknown-model"]
        
        for model in test_models:
            config_file = config_map.get(model, "synth_knowledge1.5_llama3.3.yaml")
            self.assertEqual(config_file, "synth_knowledge1.5_llama3.3.yaml")

    def test_context_length_considerations(self) -> None:
        """Test context length limits for different models."""
        # Model context lengths (simplified)
        context_limits = {
            "gpt-3.5-turbo": 16385,
            "gpt-4": 128000,
            "meta-llama/Llama-3.1-8B-Instruct": 4096,
            "mixtral-8x7b": 32768
        }
        
        # Recommended max_tokens based on context
        for model, context_limit in context_limits.items():
            if context_limit < 8000:
                recommended_max_tokens = min(1024, context_limit // 4)
                self.assertLessEqual(recommended_max_tokens, 1024)
            else:
                recommended_max_tokens = 4000
                self.assertEqual(recommended_max_tokens, 4000)

    def test_chunking_requirement_detection(self) -> None:
        """Test logic for determining when chunking is needed."""
        # Sample text lengths
        test_cases = [
            ("Short text", 50, False),
            ("Medium text" * 50, 500, False),  # ~500 chars
            ("Long text" * 200, 2000, True),   # ~2000 chars
            ("Very long text" * 500, 5000, True)  # ~5000 chars
        ]
        
        chunk_threshold = 1000  # Characters
        
        for text, expected_length, should_chunk in test_cases:
            text_length = len(text)
            needs_chunking = text_length > chunk_threshold
            
            # Verify our expectation matches the logic
            self.assertEqual(needs_chunking, should_chunk)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for different deployment scenarios."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sample_data = [{"document": "Test document for integration testing."}]
        self.dataset = Dataset.from_list(self.sample_data)

    @patch('openai.OpenAI')
    @patch('sdg_hub.flow.Flow')
    def test_openai_integration_scenario(self, mock_flow_class: Mock, mock_openai: Mock) -> None:
        """Test complete OpenAI integration scenario."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock model
        mock_model = Mock()
        mock_model.id = "gpt-4"
        mock_client.models.list.return_value.data = [mock_model]
        
        # Mock flow
        mock_flow_instance = Mock()
        mock_flow_class.return_value = mock_flow_instance
        mock_flow_instance.get_flow_from_file.return_value = mock_flow_instance
        
        # Test the integration
        client = mock_openai(api_key="test-key", base_url="https://api.openai.com/v1")
        models = client.models.list()
        
        self.assertEqual(len(models.data), 1)
        self.assertEqual(models.data[0].id, "gpt-4")

    @patch('openai.OpenAI')
    @patch('sdg_hub.flow.Flow')
    def test_local_vllm_integration_scenario(self, mock_flow_class: Mock, mock_openai: Mock) -> None:
        """Test complete local vLLM integration scenario."""
        # Mock vLLM client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock vLLM model with leading slash
        mock_model = Mock()
        mock_model.id = "/model/meta-llama/Llama-3.1-8B-Instruct"
        mock_client.models.list.return_value.data = [mock_model]
        
        # Mock successful completion
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].text = "Test response"
        mock_client.completions.create.return_value = mock_response
        
        # Test the integration
        client = mock_openai(api_key="EMPTY", base_url="http://localhost:8000/v1")
        models = client.models.list()
        
        # Verify vLLM-specific behavior
        self.assertEqual(len(models.data), 1)
        self.assertTrue(models.data[0].id.startswith("/model/"))
        
        # Test completion
        response = client.completions.create(
            model=models.data[0].id,
            prompt="Test",
            max_tokens=10
        )
        self.assertEqual(response.choices[0].text, "Test response")

    def test_error_handling_scenarios(self) -> None:
        """Test error handling for various failure scenarios."""
        error_scenarios = [
            ("Invalid API key", "Authentication failed"),
            ("Model not found", "The model does not exist"),
            ("Context length exceeded", "maximum context length"),
            ("Rate limit exceeded", "rate limit exceeded"),
            ("Server unavailable", "Connection refused")
        ]
        
        for scenario, error_message in error_scenarios:
            # Each scenario should be handled gracefully
            with self.subTest(scenario=scenario):
                # Simulate the error condition
                self.assertIn("failed", error_message.lower() + scenario.lower())

    def test_configuration_validation(self) -> None:
        """Test validation of different configuration options."""
        valid_configs = [
            {
                "backend": "openai",
                "api_key": "sk-test123",
                "base_url": "https://api.openai.com/v1",
                "model_id": "gpt-4"
            },
            {
                "backend": "vllm",
                "api_key": "EMPTY",
                "base_url": "http://localhost:8000/v1",
                "model_id": "meta-llama/Llama-3.1-8B-Instruct"
            },
            {
                "backend": "ollama",
                "api_key": "EMPTY",
                "base_url": "http://localhost:11434/v1",
                "model_id": "llama3.1"
            }
        ]
        
        for config in valid_configs:
            with self.subTest(backend=config["backend"]):
                # Basic validation checks
                self.assertIn("backend", config)
                self.assertIn("api_key", config)
                self.assertIn("base_url", config)
                self.assertIn("model_id", config)
                
                # Backend-specific validations
                if config["backend"] in ["vllm", "ollama"]:
                    self.assertEqual(config["api_key"], "EMPTY")
                    self.assertIn("localhost", config["base_url"])


if __name__ == "__main__":
    unittest.main()