# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for enhanced summary knowledge generation notebook.

Simple test that:
1. Converts notebook to Python script
2. Executes the script without errors
3. Verifies output datasets are saved and loadable
"""

import os
import subprocess

from datasets import Dataset
import pytest


@pytest.mark.integration
def test_notebook_execution_and_output_validity(
    test_env_setup, tmp_path, notebook_path
):
    """
    Test that output datasets are created and can be loaded.

    This test runs after the notebook execution and verifies outputs.
    """
    output_folder = tmp_path / "output_data"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create output subdirectories that the notebook expects
    for subdir in [
        "extractive_summary",
        "detailed_summary",
        "key_facts_to_qa",
        "document_based_qa",
    ]:
        (output_folder / subdir).mkdir(parents=True, exist_ok=True)

    # Override output folder and set minimal runtime parameters for testing
    env = os.environ.copy()
    env["OUTPUT_DATA_FOLDER"] = str(output_folder)
    # Use test seed data (single row) from the test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_seed_data = os.path.join(test_dir, "test_data", "test_seed_data.jsonl")
    env["SEED_DATA_PATH"] = test_seed_data
    env["NUMBER_OF_SUMMARIES"] = "1"  # Generate only 1 summary per document
    env["MAX_CONCURRENCY"] = "20"  # Reduce concurrency for testing

    # Convert and run
    converted_script = tmp_path / "knowledge_generation.py"
    subprocess.run(
        [
            "python",
            "-m",
            "nbconvert",
            "--to",
            "script",
            str(notebook_path),
            "--output",
            str(converted_script.stem),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )

    # Validate that notebook uses the correct flows
    expected_flows = [
        "Extractive Summary Knowledge Tuning Dataset Generation Flow",
        "Detailed Summary Knowledge Tuning Dataset Generation Flow",
        "Key Facts Knowledge Tuning Dataset Generation Flow",
        "Document Based Knowledge Tuning Dataset Generation Flow",
    ]

    with open(converted_script, "r") as f:
        script_content = f.read()

    for flow_name in expected_flows:
        assert (
            flow_name in script_content
        ), f"Expected flow not found in notebook: {flow_name}"

    print(f"✓ Validated all {len(expected_flows)} required flows are present")

    notebook_dir = notebook_path.parent
    subprocess.run(
        ["python", str(converted_script)],
        cwd=str(notebook_dir),
        env=env,
        timeout=720,
        check=True,
    )

    # Expected output files based on notebook cells
    expected_outputs = [
        output_folder / "extractive_summary" / "gen.jsonl",
        output_folder / "detailed_summary" / "gen.jsonl",
        output_folder / "key_facts_to_qa" / "gen.jsonl",
        output_folder / "document_based_qa" / "gen.jsonl",
    ]

    # Verify each output file exists and is loadable
    for output_file in expected_outputs:
        assert output_file.exists(), f"Output file not found: {output_file}"

        # Verify file can be loaded as a dataset
        dataset = Dataset.from_json(str(output_file))
        assert len(dataset) > 0, f"Dataset is empty: {output_file}"
        assert len(dataset.column_names) > 0, f"Dataset has no columns: {output_file}"

        print(
            f"✓ {output_file.name}: {len(dataset)} records, {len(dataset.column_names)} columns"
        )
