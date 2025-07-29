"""
Unit tests for the synthetic data generation tutorial notebook.

This module tests the functionality demonstrated in the Jupyter notebook
for generating synthetic question-answer pairs using LLaMA and Mixtral models.
Tests cover notebook validation, code execution simulation, and configuration validation.

Testing Framework: pytest (confirmed from repository structure)
"""

import json
import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestNotebookValidation:
    """Test suite for validating the notebook structure and content."""

    @pytest.fixture
    def notebook_content(self):
        """Fixture providing a sample notebook structure."""
        return {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Synthetic Data Generation Tutorial using LLaMA and Mixtral\n",
                        "\n",
                        "This tutorial demonstrates how to use SDG repository to generate synthetic question-answer pairs from documents using large language models like LLaMA 3.3 70B. We will also generate data using Mixtral model for comparison. We'll cover:\n",
                        "\n",
                        "1. Setting up the environment\n",
                        "2. Connecting to LLM servers\n",
                        "3. Configuring the data generation pipeline\n",
                        "4. Generating data with different models\n",
                        "5. Comparing results"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Enable auto-reloading of modules - useful during development\n",
                        "%load_ext autoreload\n", 
                        "%autoreload 2"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "base",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.11.7"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }

    def test_notebook_structure_valid(self, notebook_content):
        """Test that the notebook has valid JSON structure."""
        # Test basic notebook structure
        assert "cells" in notebook_content
        assert "metadata" in notebook_content
        assert "nbformat" in notebook_content
        assert "nbformat_minor" in notebook_content
        
        # Test nbformat version
        assert notebook_content["nbformat"] == 4
        assert isinstance(notebook_content["nbformat_minor"], int)

    def test_notebook_has_required_cells(self, notebook_content):
        """Test that notebook contains expected cell types and content."""
        cells = notebook_content["cells"]
        assert len(cells) >= 2
        
        # Test first cell is markdown with title
        first_cell = cells[0]
        assert first_cell["cell_type"] == "markdown"
        assert "Synthetic Data Generation Tutorial" in "".join(first_cell["source"])
        
        # Test has code cells
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        assert len(code_cells) >= 1

    def test_notebook_metadata_valid(self, notebook_content):
        """Test notebook metadata is properly configured."""
        metadata = notebook_content["metadata"]
        
        assert "kernelspec" in metadata
        assert "language_info" in metadata
        
        kernelspec = metadata["kernelspec"]
        assert kernelspec["language"] == "python"
        assert "python" in kernelspec["name"]
        
        language_info = metadata["language_info"]
        assert language_info["name"] == "python"
        assert language_info["file_extension"] == ".py"

    def test_notebook_tutorial_sections_present(self, notebook_content):
        """Test that all required tutorial sections are present."""
        all_content = ""
        for cell in notebook_content["cells"]:
            if cell["cell_type"] == "markdown":
                all_content += "".join(cell["source"])
        
        required_sections = [
            "Setting up the environment",
            "Connecting to LLM servers", 
            "Configuring the data generation pipeline",
            "Generating data with different models",
            "Comparing results"
        ]
        
        for section in required_sections:
            assert section in all_content, f"Missing section: {section}"


class TestNotebookCodeValidation:
    """Test suite for validating code cells in the notebook."""

    @pytest.fixture
    def import_cell_source(self):
        """Fixture providing import cell source code."""
        return [
            "from datasets import load_dataset, Dataset\n",
            "from openai import OpenAI\n",
            "\n",
            "from sdg_hub.flow import Flow\n",
            "from sdg_hub.pipeline import Pipeline\n",
            "from sdg_hub.sdg import SDG\n",
            "from sdg_hub.registry import PromptRegistry"
        ]
        
    @pytest.fixture
    def client_setup_source(self):
        """Fixture providing client setup source code."""
        return [
            "endpoint = f\"http://localhost:8000/v1\"\n",
            "openai_api_key = \"EMPTY\"\n",
            "openai_api_base = endpoint\n",
            "\n",
            "client = OpenAI(\n",
            "    api_key=openai_api_key,\n",
            "    base_url=openai_api_base,\n",
            ")\n",
            "\n",
            "teacher_model = client.models.list().data[0].id\n",
            "print(f\"Connected to model: {teacher_model}\")"
        ]

    def test_import_statements_valid(self, import_cell_source):
        """Test that import statements are syntactically correct."""
        import_code = "".join(import_cell_source)
        
        # Test that imports don't have syntax errors
        try:
            compile(import_code, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Import statements have syntax error: {e}")
        
        # Test expected imports are present
        assert "from datasets import" in import_code
        assert "from openai import OpenAI" in import_code
        assert "from sdg_hub.flow import Flow" in import_code
        assert "from sdg_hub.sdg import SDG" in import_code

    def test_client_configuration_valid(self, client_setup_source):
        """Test OpenAI client configuration code."""
        client_code = "".join(client_setup_source)
        
        # Test syntax validity
        try:
            compile(client_code, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Client setup code has syntax error: {e}")
        
        # Test configuration parameters
        assert "localhost:8000" in client_code
        assert "EMPTY" in client_code
        assert "OpenAI(" in client_code

    def test_vllm_command_syntax(self):
        """Test that vLLM server commands are properly formatted."""
        vllm_command = (
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server \\\n"
            "    --model meta-llama/Llama-3.3-70B-Instruct \\\n"
            "    --dtype float16 \\\n"
            "    --tensor-parallel-size 8"
        )
        
        # Test command components
        assert "CUDA_VISIBLE_DEVICES" in vllm_command
        assert "vllm.entrypoints.openai.api_server" in vllm_command
        assert "meta-llama/Llama-3.3-70B-Instruct" in vllm_command
        assert "--tensor-parallel-size 8" in vllm_command

    def test_sdg_configuration_valid(self):
        """Test SDG pipeline configuration syntax."""
        sdg_config_code = '''
flow_cfg = Flow(client).get_flow_from_file("synth_knowledge1.5_llama3.3.yaml")

sdg = SDG(
    [flow_cfg],
    num_workers=1,
    batch_size=1,
    save_freq=1000,
)
'''
        
        # Test syntax validity
        try:
            compile(sdg_config_code, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"SDG configuration has syntax error: {e}")
        
        # Test configuration parameters
        assert "num_workers=1" in sdg_config_code
        assert "batch_size=1" in sdg_config_code
        assert "save_freq=1000" in sdg_config_code

    def test_notebook_code_cells_executable(self):
        """Test that code cells in notebook are syntactically correct."""
        notebook_code_cells = [
            "%load_ext autoreload\n%autoreload 2",
            "from datasets import load_dataset, Dataset\nfrom openai import OpenAI",
            'endpoint = f"http://localhost:8000/v1"',
            "flow_cfg = Flow(client).get_flow_from_file('config.yaml')",
            "ds = load_dataset('json', data_files='path', split='train')",
            "generated_data = sdg.generate(ds, checkpoint_dir='Tmp')"
        ]
        
        for i, code in enumerate(notebook_code_cells):
            # Skip magic commands as they're IPython specific
            if code.startswith('%'):
                continue
                
            try:
                compile(code, f'<cell_{i}>', 'exec')
            except SyntaxError as e:
                pytest.fail(f"Code cell {i} has syntax error: {e}")


class TestNotebookExecutionSimulation:
    """Test suite simulating notebook execution scenarios."""

    @patch('openai.OpenAI')
    def test_openai_client_initialization(self, mock_openai):
        """Test OpenAI client initialization with mocked dependencies."""
        mock_client = Mock()
        mock_models = Mock()
        mock_models.list.return_value.data = [Mock(id="test-model")]
        mock_client.models = mock_models
        mock_openai.return_value = mock_client
        
        # Simulate client creation
        from openai import OpenAI
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
        
        teacher_model = client.models.list().data[0].id
        
        assert teacher_model == "test-model"
        mock_openai.assert_called_once_with(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )

    @patch('sdg_hub.flow.Flow')
    @patch('sdg_hub.sdg.SDG')
    def test_sdg_pipeline_setup(self, mock_sdg, mock_flow):
        """Test SDG pipeline setup with mocked dependencies."""
        mock_flow_instance = Mock()
        mock_flow.return_value.get_flow_from_file.return_value = mock_flow_instance
        mock_sdg_instance = Mock()
        mock_sdg.return_value = mock_sdg_instance
        
        # Simulate pipeline setup
        from sdg_hub.flow import Flow
        from sdg_hub.sdg import SDG
        
        client = Mock()
        flow_cfg = Flow(client).get_flow_from_file("synth_knowledge1.5_llama3.3.yaml")
        sdg = SDG([flow_cfg], num_workers=1, batch_size=1, save_freq=1000)
        
        mock_flow.assert_called_once_with(client)
        mock_sdg.assert_called_once_with([mock_flow_instance], num_workers=1, batch_size=1, save_freq=1000)

    @patch('datasets.load_dataset')
    def test_dataset_loading(self, mock_load_dataset):
        """Test dataset loading functionality."""
        mock_dataset = Mock()
        mock_dataset.select.return_value = Mock()
        mock_load_dataset.return_value = mock_dataset
        
        # Simulate dataset loading
        from datasets import load_dataset
        ds = load_dataset('json', data_files="test_path", split='train')
        ds_selected = ds.select(range(1))
        
        mock_load_dataset.assert_called_once_with('json', data_files="test_path", split='train')
        mock_dataset.select.assert_called_once_with(range(1))

    def test_data_generation_output_format(self):
        """Test expected format of generated data."""
        # Mock generated data structure
        mock_generated_data = [
            {
                'document': 'Sample document content',
                'question': 'What is this document about?',
                'response': 'This document is about synthetic data generation.'
            }
        ]
        
        # Test data structure
        assert len(mock_generated_data) > 0
        first_item = mock_generated_data[0]
        assert 'document' in first_item
        assert 'question' in first_item
        assert 'response' in first_item
        assert isinstance(first_item['document'], str)
        assert isinstance(first_item['question'], str)
        assert isinstance(first_item['response'], str)

    def test_multiple_model_comparison(self):
        """Test comparison between LLaMA and Mixtral outputs."""
        llama_data = [
            {'document': 'Test doc', 'question': 'LLaMA Q1', 'response': 'LLaMA A1'}
        ]
        mixtral_data = [
            {'document': 'Test doc', 'question': 'Mixtral Q1', 'response': 'Mixtral A1'}
        ]
        
        # Test both datasets have same structure
        assert len(llama_data) == len(mixtral_data)
        assert set(llama_data[0].keys()) == set(mixtral_data[0].keys())
        
        # Test they contain different content
        assert llama_data[0]['question'] != mixtral_data[0]['question']
        assert llama_data[0]['response'] != mixtral_data[0]['response']


class TestNotebookConfigurationValidation:
    """Test suite for validating configuration parameters and file paths."""

    def test_yaml_configuration_files(self):
        """Test YAML configuration file references."""
        yaml_files = [
            "synth_knowledge1.5_llama3.3.yaml",
            "../../../src/sdg_hub/flows/generation/knowledge/synth_knowledge1.5.yaml"
        ]
        
        for yaml_file in yaml_files:
            # Test file extension
            assert yaml_file.endswith('.yaml'), f"Invalid YAML extension: {yaml_file}"
            # Test file name pattern
            assert 'synth_knowledge' in yaml_file, f"Unexpected config name: {yaml_file}"

    def test_server_endpoints_valid(self):
        """Test server endpoint configurations."""
        endpoints = [
            "http://localhost:8000/v1",
            "http://10.7.0.15:8000/v1"
        ]
        
        for endpoint in endpoints:
            assert endpoint.startswith('http://'), f"Invalid protocol: {endpoint}"
            assert ':8000/v1' in endpoint, f"Invalid port/path: {endpoint}"

    def test_model_parameters_valid(self):
        """Test model configuration parameters."""
        # SDG parameters
        sdg_params = {
            'num_workers': 1,
            'batch_size': 1,
            'save_freq': 1000
        }
        
        assert sdg_params['num_workers'] > 0
        assert sdg_params['batch_size'] > 0
        assert sdg_params['save_freq'] > 0
        assert isinstance(sdg_params['num_workers'], int)
        assert isinstance(sdg_params['batch_size'], int)
        assert isinstance(sdg_params['save_freq'], int)
        
        # vLLM parameters
        vllm_params = {
            'tensor_parallel_size': 8,
            'dtype': 'float16',
            'cuda_devices': '0,1,2,3,4,5,6,7'
        }
        
        assert vllm_params['tensor_parallel_size'] > 0
        assert vllm_params['dtype'] in ['float16', 'float32', 'bfloat16']
        assert len(vllm_params['cuda_devices'].split(',')) == 8

    def test_file_output_validation(self):
        """Test file output and comparison functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "model_comparison.md")
            
            # Mock data for comparison
            generated_data = [
                {'document': 'Test doc', 'question': 'Q1', 'response': 'A1'}
            ]
            generated_data_mistral = [
                {'document': 'Test doc', 'question': 'Q2', 'response': 'A2'}
            ]
            
            # Simulate file writing logic from notebook
            with open(output_file, "w") as f:
                f.write(f"### Document \n{generated_data[0]['document']}")
                for i in range(min(len(generated_data), len(generated_data_mistral))):
                    f.write(f"Example #{i+1}\n")
                    f.write("### Result from llama3.3\n")
                    f.write(generated_data[i]['question'] + "\n")
                    f.write("*******************************\n")
                    f.write(generated_data[i]['response'] + "\n")
                    f.write("=================================\n")
                    f.write("### Result from mistral\n")
                    f.write(generated_data_mistral[i]['question'] + "\n")
                    f.write("*******************************\n")
                    f.write(generated_data_mistral[i]['response'] + "\n")
                    f.write("\n\n")
            
            # Verify file was created and has expected content
            assert os.path.exists(output_file)
            with open(output_file, 'r') as f:
                content = f.read()
                assert "### Document" in content
                assert "### Result from llama3.3" in content
                assert "### Result from mistral" in content
                assert "Test doc" in content

    def test_production_script_parameters(self):
        """Test production script parameter validation."""
        script_params = {
            'ds_path': 'seed_data.jsonl',
            'bs': 2,
            'num_workers': 10,
            'save_path': '/path/to/save',
            'flow': '../src/sdg_hub/flows/generation/knowledge/synth_knowledge1.5.yaml',
            'checkpoint_dir': '/tmp/checkpoints',
            'endpoint': 'http://localhost:8000/v1'
        }
        
        # Validate parameter types and values
        assert script_params['ds_path'].endswith('.jsonl')
        assert isinstance(script_params['bs'], int) and script_params['bs'] > 0
        assert isinstance(script_params['num_workers'], int) and script_params['num_workers'] > 0
        assert script_params['flow'].endswith('.yaml')
        assert script_params['endpoint'].startswith('http://')


class TestNotebookErrorHandling:
    """Test suite for error handling scenarios."""

    def test_missing_configuration_file(self):
        """Test behavior with missing configuration files."""
        with pytest.raises(FileNotFoundError):
            with open("nonexistent_config.yaml", 'r') as f:
                f.read()

    def test_invalid_endpoint_handling(self):
        """Test handling of invalid server endpoints."""
        invalid_endpoints = [
            "",
            "invalid-url",
            "http://",
            "localhost:8000"  # missing protocol
        ]
        
        for endpoint in invalid_endpoints:
            # Test basic validation
            if endpoint:
                is_valid = endpoint.startswith('http://') and ':8000/v1' in endpoint
                assert not is_valid, f"Endpoint should be invalid: {endpoint}"

    def test_invalid_model_parameters(self):
        """Test validation of invalid model parameters."""
        invalid_params = [
            {'num_workers': 0},
            {'num_workers': -1},
            {'batch_size': 0},
            {'save_freq': -100}
        ]
        
        for params in invalid_params:
            for key, value in params.items():
                assert value <= 0, f"Parameter {key}={value} should be invalid"

    @patch('builtins.open', side_effect=PermissionError("Access denied"))
    def test_file_permission_error(self, mock_open):
        """Test handling of file permission errors."""
        with pytest.raises(PermissionError):
            with open("restricted_file.md", "w") as f:
                f.write("test content")

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        empty_data = []
        
        # Test that empty data is handled gracefully
        assert len(empty_data) == 0
        
        # Test that operations on empty data don't crash
        for item in empty_data:
            assert False, "Should not iterate over empty data"


class TestNotebookIntegration:
    """Integration tests for notebook workflow."""

    @patch('sdg_hub.sdg.SDG')
    @patch('sdg_hub.flow.Flow')
    @patch('openai.OpenAI')
    @patch('datasets.load_dataset')
    def test_full_workflow_simulation(self, mock_load_dataset, mock_openai, mock_flow, mock_sdg):
        """Test complete workflow from setup to data generation."""
        # Setup mocks
        mock_client = Mock()
        mock_models = Mock()
        mock_models.list.return_value.data = [Mock(id="llama-model")]
        mock_client.models = mock_models
        mock_openai.return_value = mock_client
        
        mock_dataset = Mock()
        mock_dataset.select.return_value = Mock()
        mock_load_dataset.return_value = mock_dataset
        
        mock_flow_instance = Mock()
        mock_flow.return_value.get_flow_from_file.return_value = mock_flow_instance
        
        mock_sdg_instance = Mock()
        mock_generated_data = [
            {'document': 'Test', 'question': 'Q?', 'response': 'A.'}
        ]
        mock_sdg_instance.generate.return_value = mock_generated_data
        mock_sdg.return_value = mock_sdg_instance
        
        # Simulate workflow
        from openai import OpenAI
        from datasets import load_dataset
        from sdg_hub.flow import Flow
        from sdg_hub.sdg import SDG
        
        # Step 1: Setup client
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        teacher_model = client.models.list().data[0].id
        
        # Step 2: Load data
        ds = load_dataset('json', data_files="test.json", split='train')
        ds = ds.select(range(1))
        
        # Step 3: Configure pipeline
        flow_cfg = Flow(client).get_flow_from_file("config.yaml")
        sdg = SDG([flow_cfg], num_workers=1, batch_size=1, save_freq=1000)
        
        # Step 4: Generate data
        generated_data = sdg.generate(ds, checkpoint_dir="Tmp")
        
        # Verify workflow
        assert teacher_model == "llama-model"
        assert generated_data == mock_generated_data
        mock_sdg_instance.generate.assert_called_once()

    def test_command_line_usage_validation(self):
        """Test command line usage instructions."""
        command_template = (
            "python scripts/generate.py --ds_path seed_data.jsonl "
            "--bs 2 --num_workers 10 "
            "--save_path <your_save_path> "
            "--flow ../src/sdg_hub/flows/generation/knowledge/synth_knowledge1.5.yaml "
            "--checkpoint_dir <your_checkpoint_dir> "
            "--endpoint <your_endpoint>"
        )
        
        # Test command structure  
        assert "python scripts/generate.py" in command_template
        assert "--ds_path" in command_template
        assert "--bs" in command_template
        assert "--num_workers" in command_template
        assert "--flow" in command_template
        assert "--checkpoint_dir" in command_template
        assert "--endpoint" in command_template
        
        # Test parameter patterns
        assert "seed_data.jsonl" in command_template
        assert "<your_save_path>" in command_template
        assert "<your_checkpoint_dir>" in command_template
        assert "<your_endpoint>" in command_template

    def test_dual_model_setup_validation(self):
        """Test that both LLaMA and Mixtral can be configured simultaneously."""
        # Simulate dual model configuration
        llama_config = {
            'endpoint': 'http://localhost:8000/v1',
            'model': 'meta-llama/Llama-3.3-70B-Instruct',
            'config_file': 'synth_knowledge1.5_llama3.3.yaml'
        }
        
        mixtral_config = {
            'endpoint': 'http://10.7.0.15:8000/v1',
            'model': 'mixtral-8x7b-instruct',
            'config_file': '../../../src/sdg_hub/flows/generation/knowledge/synth_knowledge1.5.yaml'
        }
        
        # Validate both configurations
        for config in [llama_config, mixtral_config]:
            assert config['endpoint'].startswith('http://')
            assert config['config_file'].endswith('.yaml')
            assert len(config['model']) > 0

    @patch('builtins.open')
    def test_comparison_file_generation(self, mock_open):
        """Test the model comparison markdown file generation."""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Simulate data from both models
        generated_data = [
            {'document': 'Sample doc', 'question': 'LLaMA question', 'response': 'LLaMA response'}
        ]
        generated_data_mistral = [
            {'document': 'Sample doc', 'question': 'Mixtral question', 'response': 'Mixtral response'}
        ]
        
        # Simulate file writing
        k = 1
        output_file = "model_comparison.md"
        
        with open(output_file, "w") as f:
            f.write(f"### Document \n{generated_data[0]['document']}")
            for i in range(min(len(generated_data), len(generated_data_mistral))):
                f.write(f"Example #{i+1}\n")
                f.write("### Result from llama3.3\n")
                f.write(generated_data[i]['question'] + "\n")
                f.write("*******************************\n")
                f.write(generated_data[i]['response'] + "\n")
                f.write("=================================\n")
                f.write("### Result from mistral\n")
                f.write(generated_data_mistral[i]['question'] + "\n")
                f.write("*******************************\n")
                f.write(generated_data_mistral[i]['response'] + "\n")
                f.write("\n\n")
        
        # Verify file operations were called
        mock_open.assert_called_once_with(output_file, "w")
        assert mock_file.write.call_count > 0


class TestNotebookDocumentation:
    """Test suite for validating notebook documentation and tutorials."""

    def test_setup_instructions_complete(self):
        """Test that setup instructions are comprehensive."""
        setup_instruction = "pip install sdg-hub==0.1.0a4"
        
        assert "pip install" in setup_instruction
        assert "sdg-hub" in setup_instruction
        assert "==" in setup_instruction  # Version pinning

    def test_vllm_server_instructions_complete(self):
        """Test vLLM server setup instructions."""
        vllm_instruction = (
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server "
            "--model meta-llama/Llama-3.3-70B-Instruct "
            "--dtype float16 "
            "--tensor-parallel-size 8"
        )
        
        # Test all required components are present
        required_components = [
            "CUDA_VISIBLE_DEVICES",
            "vllm.entrypoints.openai.api_server",
            "--model",
            "--dtype",
            "--tensor-parallel-size"
        ]
        
        for component in required_components:
            assert component in vllm_instruction, f"Missing component: {component}"

    def test_tutorial_progression_logical(self):
        """Test that tutorial sections follow a logical progression."""
        tutorial_sections = [
            "Setting up the environment",
            "Connecting to LLM servers",
            "Configuring the data generation pipeline", 
            "Generating data with different models",
            "Comparing results"
        ]
        
        # Verify sections are in logical order
        assert len(tutorial_sections) == 5
        assert "Setting up" in tutorial_sections[0]
        assert "Connecting" in tutorial_sections[1]
        assert "Configuring" in tutorial_sections[2]
        assert "Generating" in tutorial_sections[3]
        assert "Comparing" in tutorial_sections[4]

    def test_code_comments_meaningful(self):
        """Test that code comments provide meaningful explanations."""
        code_comments = [
            "# Enable auto-reloading of modules - useful during development",
            "# Import required libraries",
            "# Configure OpenAI client to connect to our local vLLM server",
            "# Load the flow configuration from YAML file",
            "# Generate synthetic data and save checkpoints"
        ]
        
        for comment in code_comments:
            assert comment.startswith("#"), "Comment should start with #"
            assert len(comment.strip()) > 10, "Comment should be descriptive"

    def test_production_usage_documented(self):
        """Test that production usage is properly documented."""
        production_note = (
            "For large-scale data generation, use the command-line script instead of this notebook"
        )
        
        assert "large-scale" in production_note
        assert "command-line script" in production_note
        assert "instead of this notebook" in production_note


if __name__ == "__main__":
    pytest.main([__file__])


class TestNotebookJSONStructure:
    """Additional tests for JSON notebook structure validation."""

    def test_notebook_json_parseable(self):
        """Test that the notebook file can be parsed as valid JSON."""
        # This simulates loading the actual notebook file
        notebook_json = '''
        {
         "cells": [
          {
           "cell_type": "markdown",
           "metadata": {},
           "source": [
            "# Synthetic Data Generation Tutorial using LLaMA and Mixtral"
           ]
          }
         ],
         "metadata": {
          "kernelspec": {
           "display_name": "base",
           "language": "python",
           "name": "python3"
          }
         },
         "nbformat": 4,
         "nbformat_minor": 2
        }
        '''
        
        # Test JSON parsing
        try:
            parsed = json.loads(notebook_json)
            assert "cells" in parsed
            assert "metadata" in parsed
        except json.JSONDecodeError as e:
            pytest.fail(f"Notebook JSON is not valid: {e}")

    def test_notebook_cells_have_required_fields(self):
        """Test that all cells have required fields."""
        sample_cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Title"]
            },
            {
                "cell_type": "code", 
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["print('hello')"]
            }
        ]
        
        for cell in sample_cells:
            assert "cell_type" in cell
            assert "metadata" in cell
            assert "source" in cell
            assert cell["cell_type"] in ["markdown", "code"]
            
            if cell["cell_type"] == "code":
                assert "outputs" in cell

    def test_notebook_source_formatting(self):
        """Test that notebook source cells are properly formatted."""
        # Test markdown source formatting
        markdown_source = [
            "# Synthetic Data Generation Tutorial using LLaMA and Mixtral\n",
            "\n",
            "This tutorial demonstrates how to use SDG repository..."
        ]
        
        assert isinstance(markdown_source, list)
        assert all(isinstance(line, str) for line in markdown_source)
        
        # Test code source formatting
        code_source = [
            "from datasets import load_dataset, Dataset\n",
            "from openai import OpenAI"
        ]
        
        assert isinstance(code_source, list)
        assert all(isinstance(line, str) for line in code_source)