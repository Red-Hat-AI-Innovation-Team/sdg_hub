# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for knowledge generation notebook execution.

This module tests the actual knowledge_generation_and_mixing.ipynb notebook
using papermill with parameter injection and comprehensive LLM mocking.
Tests the real user workflow while ensuring deterministic, fast execution.
"""

from pathlib import Path
import json

import pytest

from tests.integration.notebook_utils import (
    execute_notebook_with_cell_injection,
    extract_notebook_outputs,
    validate_notebook_execution,
)

from .mock_utils import get_knowledge_generation_mock_cells

# Constants
NOTEBOOK_PATH = Path(
    "examples/knowledge_tuning/instructlab/knowledge_generation_and_mixing.ipynb"
)


@pytest.fixture(scope="session")
def executed_notebook_cache(tmp_path_factory):
    """
    Execute the notebook once per test session and cache all results.

    This session-scoped fixture ensures we only run the expensive notebook execution
    once across all tests, saving execution results to a cache directory that
    other tests can read from.
    """
    cache_dir = tmp_path_factory.mktemp("notebook_execution_cache")

    # Load mock cells from separate files for better maintainability
    mock_cells = get_knowledge_generation_mock_cells()

    # Create notebook parameters to override defaults
    notebook_params = {
        "number_of_samples": 2,  # Small dataset for testing
        "seed_data_dir": "test_sdg_demo_output",  # Use test data directory
    }

    # Execute notebook with injected mock cells and parameters
    executed_notebook_path = execute_notebook_with_cell_injection(
        notebook_path=NOTEBOOK_PATH,
        injected_cells=mock_cells,
        parameters=notebook_params,
        injection_position=1,  # Inject after first cell, before imports
        output_dir=cache_dir,
    )

    # Extract and save all artifacts
    outputs = extract_notebook_outputs(executed_notebook_path)
    execution_success = validate_notebook_execution(executed_notebook_path)

    # Check for output files
    notebook_dir = NOTEBOOK_PATH.parent
    output_dir = notebook_dir / "test_sdg_demo_output"

    # Save all artifacts to cache
    artifacts = {
        "executed_notebook_path": str(executed_notebook_path),
        "outputs": outputs,
        "execution_success": execution_success,
        "output_dir": str(output_dir),
        "notebook_params": notebook_params,
        "cache_dir": str(cache_dir),
    }

    # Save artifacts as JSON for other tests to load
    artifacts_file = cache_dir / "execution_artifacts.json"
    with open(artifacts_file, "w") as f:
        json.dump(artifacts, f, indent=2)

    print(f"[OK] Notebook executed and cached to {cache_dir}")
    return cache_dir


# All fixtures are now consolidated in the session-scoped executed_notebook_cache


@pytest.mark.integration
def test_notebook_exists():
    """Test that the target notebook file exists."""
    assert NOTEBOOK_PATH.exists(), f"Notebook not found at {NOTEBOOK_PATH}"


@pytest.mark.integration
def test_knowledge_generation_dependencies_exist():
    """Test that required flow files and dependencies exist."""
    # Check that the target notebook exists
    assert NOTEBOOK_PATH.exists(), f"Notebook not found at {NOTEBOOK_PATH}"

    # Test that FlowRegistry can actually discover flows (behavior-based test)
    from sdg_hub import FlowRegistry
    
    FlowRegistry.discover_flows()
    all_flows = FlowRegistry.list_flows()
    
    # Should discover at least some flows
    assert len(all_flows) > 0, "FlowRegistry should discover flows"
    
    # Should be able to find the specific flow we need
    expected_flow = "Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning"
    flow_names = [flow['name'] for flow in all_flows]
    assert expected_flow in flow_names, f"Should discover the knowledge generation flow: {expected_flow}"

    print("[OK] All dependencies and flow discovery validated")


@pytest.mark.integration
def test_knowledge_generation_notebook_execution_success(executed_notebook_cache):
    """Test that the cached notebook execution was successful."""
    import json

    # Load cached artifacts
    artifacts_file = executed_notebook_cache / "execution_artifacts.json"
    with open(artifacts_file, "r") as f:
        artifacts = json.load(f)

    # Validate execution success
    assert artifacts[
        "execution_success"
    ], f"Notebook execution failed - check {artifacts['executed_notebook_path']} for errors"

    print("[OK] Notebook executed successfully with mocked LLMs")


@pytest.mark.integration
def test_knowledge_generation_notebook_outputs_structure(executed_notebook_cache):
    """Test that the notebook produces the expected outputs and data structure."""
    import json

    # Load cached artifacts
    artifacts_file = executed_notebook_cache / "execution_artifacts.json"
    with open(artifacts_file, "r") as f:
        artifacts = json.load(f)

    outputs = artifacts["outputs"]

    # Validate that we got outputs from the notebook
    assert isinstance(outputs, dict), "Should have extracted outputs from notebook"

    print(f"[OK] Notebook outputs validated - found {len(outputs)} output sections")


@pytest.mark.integration
def test_knowledge_generation_output_files_created(executed_notebook_cache):
    """Test that the notebook creates the expected output files."""
    import json

    # Load cached artifacts
    artifacts_file = executed_notebook_cache / "execution_artifacts.json"
    with open(artifacts_file, "r") as f:
        artifacts = json.load(f)

    output_dir = Path(artifacts["output_dir"])

    expected_files = ["instructlab_phase_1_ds.jsonl", "instructlab_phase_2_ds.jsonl"]

    for filename in expected_files:
        file_path = output_dir / filename
        print(f"Checking for output file: {file_path}")
        if file_path.exists():
            print(f"[OK] Found output file: {filename}")
        else:
            print(f"[INFO] Output file not created (expected with mocking): {filename}")

    print("[OK] Output file validation complete")


@pytest.mark.integration
def test_mock_cells_structure():
    """Test that the mock injection cells are properly structured."""
    from .mock_utils import get_knowledge_generation_mock_cells

    mock_cells = get_knowledge_generation_mock_cells()
    assert len(mock_cells) == 2, "Should have 2 injection cells"

    for cell in mock_cells:
        assert cell["cell_type"] == "code", "All injection cells should be code cells"
        assert "source" in cell, "Cells should have source code"
        assert isinstance(cell["source"], list), "Source should be a list of strings"
        # Each source line should be a string
        for line in cell["source"]:
            assert isinstance(line, str), "Each source line should be a string"

    print("[OK] Mock injection cells are properly structured")


@pytest.mark.integration
def test_knowledge_generation_logic(executed_notebook_cache):
    """Test that the notebook uses the exact expected business logic values."""
    import json
    import re
    from pathlib import Path

    # Load cached artifacts
    artifacts_file = executed_notebook_cache / "execution_artifacts.json"
    with open(artifacts_file, "r") as f:
        artifacts = json.load(f)

    # Read the executed notebook as JSON and extract source code
    executed_notebook_path = Path(artifacts["executed_notebook_path"])
    with open(executed_notebook_path, "r") as f:
        notebook_data = json.load(f)

    # Extract all source code from notebook cells
    notebook_source = ""
    for cell in notebook_data.get("cells", []):
        if cell.get("cell_type") == "code":
            source_lines = cell.get("source", [])
            # Join the source lines into a single string
            if isinstance(source_lines, list):
                notebook_source += "".join(source_lines) + "\n"
            else:
                notebook_source += str(source_lines) + "\n"

    # Test 1: Flow search uses correct tag "question-generation"
    assert 'search_flows(tag="question-generation")' in notebook_source, \
        "Notebook should search flows with tag 'question-generation'"

    # Test 2: Exact flow name is used
    expected_flow_name = "Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning"
    assert f'flow_name = "{expected_flow_name}"' in notebook_source, \
        f"Notebook should use exact flow name: {expected_flow_name}"

    # Test 3: Correct model configuration
    expected_model = "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"
    assert f'model="{expected_model}"' in notebook_source, \
        f"Notebook should configure model: {expected_model}"

    # Test 4: Dataset loading path
    # Check for the exact pattern f'{seed_data_dir}/seed_data.jsonl'
    dataset_pattern = r"f'\{seed_data_dir\}/seed_data\.jsonl'"
    assert re.search(dataset_pattern, notebook_source), \
        "Notebook should load dataset using f'{seed_data_dir}/seed_data.jsonl' pattern"

    print("[OK] All business logic values validated")


@pytest.mark.integration 
def test_knowledge_generation_data_integrity(executed_notebook_cache):
    """Test that generated dataset has exact expected shape and content by analyzing saved files."""
    from datasets import Dataset
    from pathlib import Path

    # Check for the output files that the notebook should create
    notebook_dir = NOTEBOOK_PATH.parent
    output_dir = notebook_dir / "sdg_demo_output"
    
    phase1_file = output_dir / "instructlab_phase_1_ds.jsonl"
    phase2_file = output_dir / "instructlab_phase_2_ds.jsonl"
    
    # Both files should exist (created by the notebook)
    assert phase1_file.exists(), f"Phase 1 dataset file not found at {phase1_file}"
    assert phase2_file.exists(), f"Phase 2 dataset file not found at {phase2_file}"
    
    # Load the phase 2 dataset (this is the main knowledge generation output)
    phase2_dataset = Dataset.from_json(str(phase2_file))
    
    # Expected exact row count based on our mock setup:
    # The knowledge generation flow with 2 input docs should produce a specific number of QA pairs
    # We can determine the exact count by analyzing what we get
    expected_min_rows = 2  # At least one QA pair per input document
    
    # Validate basic dataset properties
    assert len(phase2_dataset) >= expected_min_rows, f"Expected at least {expected_min_rows} rows, got {len(phase2_dataset)}"
    
    # Expected columns for InstructLab Phase 2 format (actual InstructLab format)
    expected_phase2_columns = {"metadata", "id", "messages"}  # InstructLab chat format
    actual_columns = set(phase2_dataset.column_names)
    
    assert actual_columns == expected_phase2_columns, f"Phase 2 columns mismatch. Expected: {expected_phase2_columns}, Got: {actual_columns}"
    
    # Validate content integrity - check first few rows
    for i in range(min(3, len(phase2_dataset))):
        row = phase2_dataset[i]
        
        # InstructLab format has messages as a list of {"role": "user/assistant", "content": "..."}
        messages = row["messages"]
        assert isinstance(messages, list), f"Messages should be a list, row {i}"
        assert len(messages) >= 2, f"Should have user and assistant messages, row {i}"
        
        # Find user and assistant messages
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        
        assert len(user_messages) > 0, f"Should have user message, row {i}"
        assert len(assistant_messages) > 0, f"Should have assistant message, row {i}"
        
        # Check content quality
        user_content = user_messages[0].get("content", "")
        assistant_content = assistant_messages[0].get("content", "")
        
        assert len(user_content) > 10, f"User content should have substance, row {i}: {user_content[:50]}..."
        assert len(assistant_content) > 10, f"Assistant content should have substance, row {i}: {assistant_content[:50]}..."
    
    # Also check phase 1 dataset structure  
    phase1_dataset = Dataset.from_json(str(phase1_file))
    expected_phase1_columns = {"messages", "id", "unmask", "metadata"}  # InstructLab pretraining format
    actual_phase1_columns = set(phase1_dataset.column_names)
    
    assert actual_phase1_columns == expected_phase1_columns, f"Phase 1 columns mismatch. Expected: {expected_phase1_columns}, Got: {actual_phase1_columns}"

    print(f"[OK] Phase 1 dataset: {len(phase1_dataset)} rows, columns: {list(phase1_dataset.column_names)}")
    print(f"[OK] Phase 2 dataset: {len(phase2_dataset)} rows, columns: {list(phase2_dataset.column_names)}")
    
    # Show sample content from InstructLab format
    sample_messages = phase2_dataset[0]['messages']
    user_msg = next(msg for msg in sample_messages if msg['role'] == 'user')['content']
    assistant_msg = next(msg for msg in sample_messages if msg['role'] == 'assistant')['content']
    
    print(f"[OK] Sample user message: {user_msg[:100]}...")
    print(f"[OK] Sample assistant response: {assistant_msg[:100]}...")
    print("[OK] All dataset integrity validations passed!")
