# SPDX-License-Identifier: Apache-2.0

"""
Utilities for loading mock cells for knowledge generation notebook testing.
"""

from pathlib import Path


def load_mock_cell(cell_name: str) -> dict:
    """
    Load a mock cell from a Python file.
    
    Args:
        cell_name: Name of the mock cell file (without .py extension)
        
    Returns:
        Dictionary representing a Jupyter notebook cell
    """
    cell_file = Path(__file__).parent / "mock_cells" / f"{cell_name}.py"
    
    with open(cell_file, 'r') as f:
        code = f.read()
    
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [cell_name]},
        "outputs": [],
        "source": code.split('\n')
    }


def get_knowledge_generation_mock_cells() -> list:
    """
    Get all mock cells needed for knowledge generation notebook testing.
    
    Returns:
        List of mock cell dictionaries
    """
    return [
        load_mock_cell("llm_setup"),
        load_mock_cell("test_data_setup")
    ]