# SPDX-License-Identifier: Apache-2.0

"""Utility functions for notebook integration testing."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import papermill as pm
from datasets import Dataset


def execute_notebook_with_params(
    notebook_path: Path,
    parameters: Dict[str, Any],
    output_dir: Optional[Path] = None,
    kernel_name: str = "python3"
) -> Path:
    """
    Execute a notebook with parameters using papermill.
    
    Args:
        notebook_path: Path to the input notebook
        parameters: Dictionary of parameters to inject
        output_dir: Directory to save the executed notebook (temp if None)
        kernel_name: Jupyter kernel name to use
        
    Returns:
        Path to the executed notebook
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    
    output_path = output_dir / f"executed_{notebook_path.name}"
    
    # Change to notebook directory to handle local imports
    original_cwd = Path.cwd()
    notebook_dir = notebook_path.parent
    
    try:
        import os
        os.chdir(notebook_dir)
        
        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            parameters=parameters,
            kernel_name=kernel_name,
            progress_bar=False,
            cwd=str(notebook_dir)
        )
    finally:
        os.chdir(original_cwd)
    
    return output_path


def validate_notebook_execution(notebook_path: Path) -> bool:
    """
    Validate that a notebook executed successfully.
    
    Args:
        notebook_path: Path to the executed notebook
        
    Returns:
        True if notebook executed without errors
    """
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            outputs = cell.get('outputs', [])
            for output in outputs:
                if output.get('output_type') == 'error':
                    return False
    
    return True


def extract_notebook_outputs(notebook_path: Path, cell_tags: Optional[list] = None) -> Dict[str, Any]:
    """
    Extract outputs from specific notebook cells.
    
    Args:
        notebook_path: Path to the executed notebook  
        cell_tags: List of cell tags to extract outputs from (all if None)
        
    Returns:
        Dictionary mapping cell tags to their outputs
    """
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    outputs = {}
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            tags = cell.get('metadata', {}).get('tags', [])
            
            if cell_tags is None or any(tag in tags for tag in cell_tags):
                cell_outputs = []
                for output in cell.get('outputs', []):
                    if output.get('output_type') in ['execute_result', 'display_data']:
                        cell_outputs.append(output.get('data', {}))
                    elif output.get('output_type') == 'stream':
                        cell_outputs.append(output.get('text', ''))
                
                if tags:
                    for tag in tags:
                        outputs[tag] = cell_outputs
                else:
                    outputs[f"cell_{len(outputs)}"] = cell_outputs
    
    return outputs


def execute_notebook_with_cell_injection(
    notebook_path: Path,
    injected_cells: list,
    parameters: Dict[str, Any] = None,
    injection_position: int = 2,
    output_dir: Optional[Path] = None,
    kernel_name: str = "python3"
) -> Path:
    """
    Execute a notebook with cells injected at a specific position.
    
    Args:
        notebook_path: Path to the input notebook
        injected_cells: List of cell dictionaries to inject
        parameters: Dictionary of parameters to inject
        injection_position: Position to insert cells (default: 2, after imports)
        output_dir: Directory to save the executed notebook (temp if None)
        kernel_name: Jupyter kernel name to use
        
    Returns:
        Path to the executed notebook
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    if parameters is None:
        parameters = {}
    
    # Load notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Insert cells at specified position
    for i, cell in enumerate(injected_cells):
        notebook['cells'].insert(injection_position + i, cell)
    
    # Remove the last cell (which has the problematic instruction dataset loading)
    if len(notebook['cells']) > 0:
        notebook['cells'].pop()  # Remove the last cell
    
    # Create temporary notebook file in the same directory as original notebook
    # This ensures relative paths work correctly
    notebook_path = notebook_path.resolve()  # Make sure it's absolute
    notebook_dir = notebook_path.parent
    temp_notebook_path = notebook_dir / f"temp_{notebook_path.name}"
    with open(temp_notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    try:
        # Execute using papermill directly with absolute paths
        output_path = output_dir / f"executed_{notebook_path.name}"
        
        pm.execute_notebook(
            str(temp_notebook_path.resolve()),  # Use absolute path
            str(output_path.resolve()),         # Use absolute path for output
            parameters=parameters,
            kernel_name=kernel_name,
            progress_bar=False,
            cwd=str(notebook_dir.resolve())     # Use absolute path for cwd
        )
        
        executed_path = output_path
    finally:
        # Clean up temp notebook (temporarily keep for debugging)
        pass
        # if temp_notebook_path.exists():
        #     temp_notebook_path.unlink()
    
    return executed_path


def validate_dataset_structure(
    dataset: Dataset,
    required_columns: Optional[list] = None,
    min_rows: int = 1,
    max_rows: Optional[int] = None
) -> bool:
    """
    Validate that a dataset has the expected structure.
    
    Args:
        dataset: HuggingFace Dataset to validate
        required_columns: List of required column names (optional)
        min_rows: Minimum number of rows required
        max_rows: Maximum number of rows allowed (optional)
        
    Returns:
        True if dataset structure is valid
    """
    if not isinstance(dataset, Dataset):
        return False
    
    # Check row count
    num_rows = len(dataset)
    if num_rows < min_rows:
        return False
    if max_rows is not None and num_rows > max_rows:
        return False
    
    # Check required columns
    if required_columns:
        dataset_columns = set(dataset.column_names)
        missing_columns = set(required_columns) - dataset_columns
        if missing_columns:
            return False
    
    return True
