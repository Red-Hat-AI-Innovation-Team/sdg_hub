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

import pandas as pd
import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY environment variable for LLM API calls",
)
def test_notebook_execution_and_output_validity(
    test_env_setup, tmp_path, notebook_path
):
    """
    Test that output datasets are created and can be loaded.

    This test runs after the notebook execution and verifies outputs.
    Requires OPENAI_API_KEY environment variable to be set.
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
    env["NUMBER_OF_SUMMARIES"] = "2"  # Generate only 2 summaries per document
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

    # Verify each output file exists and is loadable, and collect statistics
    print("\n" + "=" * 90)
    print("📊 GENERATED DATA STATISTICS")
    print("=" * 90)

    stats = {}
    for output_file in expected_outputs:
        assert output_file.exists(), f"Output file not found: {output_file}"

        # Verify file can be loaded as a dataset
        dataset = pd.read_json(str(output_file), lines=True)
        assert len(dataset) > 0, f"Dataset is empty: {output_file}"
        assert (
            len(dataset.columns.tolist()) > 0
        ), f"Dataset has no columns: {output_file}"

        # Calculate statistics
        summary_type = output_file.parent.name
        total_records = len(dataset)

        # Count unique raw documents and generated summaries/key facts
        unique_raw_docs = (
            len(set(dataset["raw_document"]))
            if "raw_document" in dataset.columns.tolist()
            else 0
        )

        # For key_facts_to_qa, count unique key facts instead of summaries
        if summary_type == "key_facts_to_qa":
            unique_summaries = (
                len(set(dataset["key_fact"]))
                if "key_fact" in dataset.columns.tolist()
                else 0
            )
        else:
            unique_summaries = (
                len(set(dataset["document"]))
                if "document" in dataset.columns.tolist()
                else 0
            )

        # Calculate summaries/key facts per input document
        summaries_per_doc = (
            unique_summaries / unique_raw_docs if unique_raw_docs > 0 else 0
        )

        # Calculate QA pairs per summary/key fact
        qa_per_summary = total_records / unique_summaries if unique_summaries > 0 else 0

        # Store stats
        stats[summary_type] = {
            "summary_type": summary_type.replace("_", " ").title(),
            "total_qa_pairs": total_records,
            "unique_raw_docs": unique_raw_docs,
            "unique_summaries": unique_summaries,
            "summaries_per_doc": summaries_per_doc,
            "qa_per_summary": qa_per_summary,
        }

    # Print statistics table
    print(
        f"\n{'Summary Type':<25} {'Q&A Pairs':<12} {'Raw Docs':<12} "
        f"{'Sum/KeyFacts':<12} {'Per Doc':<10} {'QA/Unit':<10}"
    )
    print("-" * 90)

    for summary_type, stat in stats.items():
        per_doc_str = (
            f"{stat['summaries_per_doc']:.1f}"
            if summary_type not in ["document_based_qa"]
            else "N/A"
        )
        print(
            f"{stat['summary_type']:<25} {stat['total_qa_pairs']:<12} "
            f"{stat['unique_raw_docs']:<12} {stat['unique_summaries']:<12} "
            f"{per_doc_str:<10} {stat['qa_per_summary']:<10.1f}"
        )

    # Print overall summary
    print("-" * 90)
    total_qa = sum(s["total_qa_pairs"] for s in stats.values())
    total_unique_raw = stats.get("extractive_summary", {}).get("unique_raw_docs", 0)
    print(f"{'TOTAL':<25} {total_qa:<12} {total_unique_raw:<12}")
    print("=" * 90 + "\n")
