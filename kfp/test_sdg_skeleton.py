#!/usr/bin/env python3
"""Test script to verify the SDG component works with KFP local runner.

Uses SubprocessRunner for fast local development (no Docker rebuild needed).
For container testing, use test_sdg_docker.py instead.
"""

import os

from kfp import dsl, local

# Get absolute path to test data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDATA_DIR = os.path.join(SCRIPT_DIR, "testdata")
TEST_INPUT_PATH = os.path.join(TESTDATA_DIR, "sample_input.jsonl")

# Use SubprocessRunner for fast local development (no Docker needed)
# use_venv=False to use current environment directly
local.init(runner=local.SubprocessRunner(use_venv=False))

# Import the SDG component
from sdg_hub.kfp import sdg


@dsl.pipeline(name="sdg-input-test")
def test_pipeline_with_input():
    """Test pipeline with actual input file."""
    sdg_task = sdg(
        input_pvc_path=TEST_INPUT_PATH,
        flow_id="test-flow",
        model="hosted_vllm/test-model",
        temperature=0.8,
        max_tokens=1024,
        log_level="INFO",
    )


@dsl.pipeline(name="sdg-no-input-test")
def test_pipeline_no_input():
    """Test pipeline without input (uses dummy data)."""
    sdg_task = sdg(
        flow_id="test-flow",
        log_level="INFO",
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SDG Hub KFP Component - Input Handling")
    print("=" * 60)

    print(f"\nTest input file: {TEST_INPUT_PATH}")
    print(f"File exists: {os.path.exists(TEST_INPUT_PATH)}")

    print("\n--- Test 1: With Input File ---")
    result1 = test_pipeline_with_input()

    print("\n--- Test 2: Without Input (Dummy Data) ---")
    result2 = test_pipeline_no_input()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)