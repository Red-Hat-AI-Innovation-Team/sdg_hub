# SPDX-License-Identifier: Apache-2.0
"""Integration tests for InstructLab Q&A generation flow.

Tests:
1. Converts the demo notebook to a Python script
2. Validates the script references the correct flow
3. Executes the script end-to-end (requires OPENAI_API_KEY)
4. Verifies output files are generated
"""

import os
import subprocess

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY environment variable for LLM API calls",
)
def test_notebook_execution_and_output(test_env_setup, tmp_path, notebook_path):
    """Test that the demo notebook runs end-to-end and produces output files."""
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"

    # Convert notebook to script
    converted_script = tmp_path / "demo.py"
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

    # Validate script references the correct flow
    with open(converted_script) as f:
        script_content = f.read()

    assert "bright-coral-421" in script_content, "Flow ID not found in notebook"
    assert "FlowRegistry" in script_content, "FlowRegistry usage not found"

    # Execute the notebook script
    env = os.environ.copy()
    notebook_dir = notebook_path.parent
    subprocess.run(
        ["python", str(converted_script)],
        cwd=str(notebook_dir),
        env=env,
        timeout=600,
        check=True,
    )

    # Verify output directory was created with expected files
    output_dir = notebook_dir / "output"
    assert output_dir.exists(), "Output directory was not created"

    # Should have at least one taxonomy path directory with qna.yaml
    qna_files = list(output_dir.rglob("qna.yaml"))
    attr_files = list(output_dir.rglob("attribution.txt"))

    assert len(qna_files) > 0, "No qna.yaml files generated"
    assert len(attr_files) > 0, "No attribution.txt files generated"

    # Validate qna.yaml content
    for qna_file in qna_files:
        content = qna_file.read_text()
        assert "version: 3" in content, f"Missing version in {qna_file}"
        assert "seed_examples:" in content, f"Missing seed_examples in {qna_file}"
        assert "question:" in content, f"Missing questions in {qna_file}"
        assert "answer:" in content, f"Missing answers in {qna_file}"

    print(f"\nGenerated {len(qna_files)} qna.yaml file(s)")
    for f in qna_files:
        print(f"  {f.relative_to(output_dir)}")
