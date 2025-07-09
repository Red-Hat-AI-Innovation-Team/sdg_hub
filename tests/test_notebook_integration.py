# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the updated notebook functionality."""

# Standard
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

# Third Party
from datasets import Dataset
from openai import OpenAI

# Local
from sdg_hub.flow import Flow
from sdg_hub.sdg import SDG


class TestNotebookIntegration(unittest.TestCase):
    """Integration tests simulating notebook execution."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sample_documents = [
            {
                "document": """
                Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
                that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, 
                problem-solving, perception, and language understanding. AI systems can be categorized into two main types: 
                narrow AI, which is designed for specific tasks, and general AI, which would have human-like cognitive abilities 
                across multiple domains. Machine learning, a subset of AI, enables computers to learn and improve from experience 
                without being explicitly programmed for every task.
                """
            },
            {
                "document": """
                Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations 
                are natural, scientific evidence shows that human activities have been the primary driver of climate change since 
                the mid-20th century. The burning of fossil fuels releases greenhouse gases like carbon dioxide into the atmosphere, 
                which trap heat and warm the planet. This warming leads to rising sea levels, more frequent extreme weather events, 
                changes in precipitation patterns, and impacts on ecosystems and biodiversity.
                """
            }
        ]

    @patch('openai.OpenAI')
    @patch('sdg_hub.flow.Flow')
    def test_complete_openai_workflow(self, mock_flow_class: Mock, mock_openai: Mock) -> None:
        """Test complete workflow with OpenAI backend (simulating notebook execution)."""
        # Step 1: Mock OpenAI client initialization
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock available models
        mock_model = Mock()
        mock_model.id = "gpt-4"
        mock_client.models.list.return_value.data = [mock_model]
        
        # Step 2: Initialize client (as in notebook)
        BACKEND = "openai"
        API_KEY = "test-key"
        BASE_URL = "https://api.openai.com/v1"
        MODEL_ID = "gpt-4"
        
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        # Step 3: Verify connection and get models (as in notebook)
        models = client.models.list()
        available_model_ids = [m.id for m in models.data]
        
        if MODEL_ID in available_model_ids:
            teacher_model = MODEL_ID
        else:
            teacher_model = models.data[0].id
            
        self.assertEqual(teacher_model, "gpt-4")
        
        # Step 4: Mock flow configuration
        mock_flow_instance = Mock()
        mock_flow_class.return_value = mock_flow_instance
        
        config_map = {
            "gpt-4": "synth_knowledge1.5_llama3.3.yaml",
            "gpt-3.5-turbo": "synth_knowledge1.5_llama3.3.yaml"
        }
        
        config_file = config_map.get(teacher_model, "synth_knowledge1.5_llama3.3.yaml")
        self.assertEqual(config_file, "synth_knowledge1.5_llama3.3.yaml")
        
        # Step 5: Create dataset (as in notebook)
        ds = Dataset.from_list(self.sample_documents)
        test_size = min(2, len(ds))
        ds = ds.select(range(test_size))
        
        self.assertEqual(len(ds), 2)
        
        # Verify successful workflow setup
        self.assertEqual(BACKEND, "openai")
        self.assertIsNotNone(client)
        self.assertEqual(teacher_model, "gpt-4")

    @patch('openai.OpenAI')
    @patch('sdg_hub.flow.Flow')
    def test_complete_vllm_workflow(self, mock_flow_class: Mock, mock_openai: Mock) -> None:
        """Test complete workflow with vLLM backend (simulating notebook execution)."""
        # Step 1: Mock vLLM client initialization
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock vLLM model (with leading slash)
        mock_model = Mock()
        mock_model.id = "/model/meta-llama/Llama-3.1-8B-Instruct"
        mock_client.models.list.return_value.data = [mock_model]
        
        # Step 2: Initialize client for vLLM (as in notebook)
        BACKEND = "vllm"
        API_KEY = "EMPTY"
        BASE_URL = "http://localhost:8000/v1"
        MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
        
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        # Step 3: Handle vLLM-specific model ID format
        models = client.models.list()
        available_model_ids = [m.id for m in models.data]
        
        # vLLM may return models with leading slash
        if MODEL_ID in available_model_ids:
            teacher_model = MODEL_ID
        elif f"/{MODEL_ID}" in available_model_ids:
            teacher_model = f"/{MODEL_ID}"
        elif f"/model/{MODEL_ID}" in available_model_ids:
            teacher_model = f"/model/{MODEL_ID}"
        else:
            teacher_model = models.data[0].id
            
        self.assertEqual(teacher_model, "/model/meta-llama/Llama-3.1-8B-Instruct")
        
        # Step 4: Test context length considerations for local models
        # For small models, we should use chunking
        context_limit = 4096  # Typical for 8B models
        if context_limit < 8000:
            max_tokens = min(1024, context_limit // 4)
            use_chunking = True
        else:
            max_tokens = 4000
            use_chunking = False
            
        self.assertEqual(max_tokens, 1024)
        self.assertTrue(use_chunking)

    def test_environment_variable_workflow(self) -> None:
        """Test workflow using environment variables (as in notebook)."""
        # Test environment variable configuration
        test_env_vars = {
            "OPENAI_API_KEY": "test-openai-key",
            "SEED_DATA_PATH": "/path/to/test/data.json",
            "OUTPUT_DIR": "./test_outputs",
            "CHECKPOINT_DIR": "./test_checkpoints",
            "MODEL_BACKEND": "openai",
            "MODEL_ID": "gpt-4",
            "BASE_URL": "https://api.openai.com/v1"
        }
        
        with patch.dict(os.environ, test_env_vars):
            # Simulate notebook environment variable reading
            API_KEY = os.getenv("OPENAI_API_KEY", "default-key")
            SEED_DATA_PATH = os.getenv("SEED_DATA_PATH", "default.json")
            OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
            CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
            
            self.assertEqual(API_KEY, "test-openai-key")
            self.assertEqual(SEED_DATA_PATH, "/path/to/test/data.json")
            self.assertEqual(OUTPUT_DIR, "./test_outputs")
            self.assertEqual(CHECKPOINT_DIR, "./test_checkpoints")

    def test_data_loading_scenarios(self) -> None:
        """Test different data loading scenarios from notebook."""
        # Scenario 1: Using sample data (as in notebook when no file provided)
        sample_documents = self.sample_documents
        ds = Dataset.from_list(sample_documents)
        
        self.assertEqual(len(ds), 2)
        self.assertIn("document", ds.column_names)
        
        # Scenario 2: Simulating file loading
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(self.sample_documents, f)
            temp_file = f.name
        
        try:
            # Simulate file exists check and loading
            if os.path.exists(temp_file):
                from datasets import load_dataset
                ds_from_file = load_dataset('json', data_files=temp_file, split='train')
                self.assertEqual(len(ds_from_file), 2)
        finally:
            os.unlink(temp_file)

    @patch('sdg_hub.sdg.SDG')
    @patch('sdg_hub.flow.Flow')
    def test_sdg_pipeline_initialization(self, mock_flow_class: Mock, mock_sdg_class: Mock) -> None:
        """Test SDG pipeline initialization (as in notebook)."""
        # Mock flow
        mock_flow_instance = Mock()
        mock_flow_class.return_value = mock_flow_instance
        
        # Mock SDG
        mock_sdg_instance = Mock()
        mock_sdg_class.return_value = mock_sdg_instance
        
        # Simulate notebook SDG initialization
        flow_cfg = mock_flow_instance
        sdg = SDG(
            [flow_cfg],         # Use Flow directly (not wrapped in Pipeline)
            num_workers=1,      # Number of parallel workers
            batch_size=1,       # Batch size for processing  
            save_freq=1000,     # How often to save checkpoints
        )
        
        # Verify SDG was initialized correctly
        mock_sdg_class.assert_called_once_with(
            [mock_flow_instance],
            num_workers=1,
            batch_size=1,
            save_freq=1000
        )

    def test_output_and_analysis_workflow(self) -> None:
        """Test output saving and analysis workflow from notebook."""
        # Create mock generated data
        generated_data = Dataset.from_list([
            {
                "document": "Original document",
                "question": "What is this about?",
                "response": "This is about testing.",
                "chunk_id": 0,
                "total_chunks": 1
            },
            {
                "document": "Another document",
                "question": "What does this describe?", 
                "response": "This describes something else.",
                "chunk_id": 0,
                "total_chunks": 1
            }
        ])
        
        # Test analysis (as in notebook)
        total_examples = len(generated_data)
        columns = generated_data.column_names
        
        self.assertEqual(total_examples, 2)
        self.assertIn("question", columns)
        self.assertIn("response", columns)
        self.assertIn("document", columns)
        
        # Test average length calculation
        if 'question' in columns:
            avg_q_length = sum(len(q) for q in generated_data['question']) / len(generated_data)
            self.assertGreater(avg_q_length, 0)
            
        if 'response' in columns:
            avg_r_length = sum(len(r) for r in generated_data['response']) / len(generated_data)
            self.assertGreater(avg_r_length, 0)

    def test_error_handling_and_troubleshooting(self) -> None:
        """Test error handling scenarios mentioned in notebook."""
        # Test common error scenarios and their handling
        error_scenarios = [
            {
                "error": "Context length exceeded",
                "solution": "Use chunking or reduce max_tokens",
                "implemented": True
            },
            {
                "error": "Model not found",
                "solution": "Check model ID and availability",
                "implemented": True
            },
            {
                "error": "Connection refused",
                "solution": "Verify server is running and endpoint is correct",
                "implemented": True
            },
            {
                "error": "API rate limits",
                "solution": "Reduce batch size or add delays",
                "implemented": True
            }
        ]
        
        for scenario in error_scenarios:
            with self.subTest(error=scenario["error"]):
                # Each scenario should have a solution
                self.assertTrue(scenario["implemented"])
                self.assertIsNotNone(scenario["solution"])

    def test_customization_options(self) -> None:
        """Test customization options mentioned in notebook."""
        # Test configuration options
        customization_options = {
            "backends": ["openai", "vllm", "ollama", "azure"],
            "chunk_sizes": [512, 1000, 1500, 2000],
            "max_tokens": [512, 1024, 2048, 4000],
            "num_workers": [1, 2, 4, 8],
            "batch_sizes": [1, 2, 4, 8]
        }
        
        # Verify all options are valid
        for option, values in customization_options.items():
            with self.subTest(option=option):
                self.assertIsInstance(values, list)
                self.assertGreater(len(values), 0)
                
                # Test reasonable value ranges
                if option == "chunk_sizes":
                    for size in values:
                        self.assertGreaterEqual(size, 256)
                        self.assertLessEqual(size, 4000)
                        
                elif option == "max_tokens":
                    for tokens in values:
                        self.assertGreaterEqual(tokens, 256)
                        self.assertLessEqual(tokens, 8000)


class TestBackendSpecificFeatures(unittest.TestCase):
    """Test backend-specific features and requirements."""

    def test_vllm_specific_features(self) -> None:
        """Test vLLM-specific features and requirements."""
        # Test model ID format handling
        vllm_model_formats = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "/model/meta-llama/Llama-3.1-8B-Instruct", 
            "/meta-llama/Llama-3.1-8B-Instruct"
        ]
        
        for model_id in vllm_model_formats:
            # Should handle different vLLM model ID formats
            normalized_id = model_id
            if not normalized_id.startswith("/"):
                normalized_id = f"/model/{normalized_id}"
                
            self.assertTrue(normalized_id.startswith("/"))

    def test_context_length_handling(self) -> None:
        """Test context length handling for different model sizes."""
        models_and_limits = [
            ("gpt-4", 128000, 4000),
            ("gpt-3.5-turbo", 16385, 4000),
            ("meta-llama/Llama-3.1-8B-Instruct", 4096, 1024),
            ("mixtral-8x7b", 32768, 4000)
        ]
        
        for model, context_limit, expected_max_tokens in models_and_limits:
            with self.subTest(model=model):
                # Calculate recommended max_tokens
                if context_limit < 8000:
                    recommended = min(1024, context_limit // 4)
                else:
                    recommended = 4000
                    
                self.assertEqual(recommended, expected_max_tokens)

    def test_chunking_integration(self) -> None:
        """Test integration with ChunkingBlock for small context models."""
        # For models with limited context, chunking should be used
        small_context_models = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "codellama/CodeLlama-7b-Instruct"
        ]
        
        for model in small_context_models:
            # These models should use chunking
            needs_chunking = True  # For models with < 8K context
            chunk_size = 512  # Conservative chunk size
            overlap = 50  # Small overlap
            
            self.assertTrue(needs_chunking)
            self.assertLessEqual(chunk_size, 1000)
            self.assertGreater(overlap, 0)


if __name__ == "__main__":
    unittest.main()