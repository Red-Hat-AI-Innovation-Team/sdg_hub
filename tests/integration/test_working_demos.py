# SPDX-License-Identifier: Apache-2.0

"""Working integration test demonstrations."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.integration.notebook_utils import (
    execute_notebook_with_params,
    validate_notebook_execution,
    extract_notebook_outputs,
)

pytestmark = pytest.mark.integration

DEMO_NOTEBOOK_PATH = Path(__file__).parent.parent.parent / "examples" / "integration_test_demo.ipynb"


def test_notebook_exists():
    """Verify demo notebook exists."""
    assert DEMO_NOTEBOOK_PATH.exists()


def test_parameter_injection(temp_output_dir: Path):
    """Test parameter injection into notebooks."""
    test_parameters = {
        "api_key": "injected-test-key",
        "api_base": "http://mock-server:9999/v1", 
        "model_name": "injected-test-model",
        "test_mode": True,
        "sample_size": 2
    }
    
    with patch('openai.OpenAI') as mock_openai_class:
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "injected-test-model"
        mock_client.models.list.return_value.data = [mock_model]
        
        mock_responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="Business"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Sci/Tech"))]),
        ]
        mock_client.chat.completions.create.side_effect = mock_responses
        mock_openai_class.return_value = mock_client
        
        executed_notebook_path = execute_notebook_with_params(
            DEMO_NOTEBOOK_PATH,
            test_parameters,
            output_dir=temp_output_dir
        )
        
        with open(executed_notebook_path, 'r') as f:
            executed_notebook = json.load(f)
        
        # Check that injected parameters are found
        injected_parameters_found = False
        for cell in executed_notebook['cells']:
            if cell.get('cell_type') == 'code':
                cell_source = ''.join(cell.get('source', []))
                if ("injected-test-key" in cell_source or 
                    "http://mock-server:9999/v1" in cell_source or
                    "injected-test-model" in cell_source):
                    injected_parameters_found = True
                    break
        
        if not injected_parameters_found:
            for cell in executed_notebook['cells']:
                if 'outputs' in cell:
                    for output in cell['outputs']:
                        if output.get('output_type') == 'stream' and 'text' in output:
                            output_text = ''.join(output['text'])
                            if ("injected-test-key" in output_text or 
                                "http://mock-server:9999/v1" in output_text):
                                injected_parameters_found = True
                                break
        
        assert injected_parameters_found


def test_output_extraction(temp_output_dir: Path):
    """Test extracting outputs from executed notebooks."""
    test_parameters = {
        "api_key": "test-key",
        "api_base": "http://test-server/v1",
        "model_name": "test-model",
        "test_mode": True,
        "sample_size": 2
    }
    
    with patch('openai.OpenAI') as mock_openai_class:
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "test-model"
        mock_client.models.list.return_value.data = [mock_model]
        
        mock_responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="Business"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Sci/Tech"))]),
        ]
        mock_client.chat.completions.create.side_effect = mock_responses
        mock_openai_class.return_value = mock_client
        
        executed_notebook_path = execute_notebook_with_params(
            DEMO_NOTEBOOK_PATH,
            test_parameters,
            output_dir=temp_output_dir
        )
        
        # Test output extraction
        outputs = extract_notebook_outputs(executed_notebook_path, cell_tags=["generate", "output"])
        
        # Validate notebook executed
        with open(executed_notebook_path, 'r') as f:
            executed_notebook = json.load(f)
        
        executed_cells = 0
        total_code_cells = 0
        for cell in executed_notebook['cells']:
            if cell.get('cell_type') == 'code':
                total_code_cells += 1
                if cell.get('execution_count') is not None:
                    executed_cells += 1
        
        assert executed_cells == total_code_cells, f"Expected all {total_code_cells} code cells to execute, but only {executed_cells} executed"