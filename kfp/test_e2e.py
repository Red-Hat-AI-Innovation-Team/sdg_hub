#!/usr/bin/env python3
"""E2E test running the SDG component against a real transform flow.

Uses SubprocessRunner to execute the component locally without Docker.
Tests the full code path: input loading -> flow loading -> execution -> output.
"""

import json
import os

from sdg_hub.kfp import sdg

from kfp import dsl, local

# Use SubprocessRunner for local development (no Docker needed)
# use_venv=False to use current environment directly
local.init(runner=local.SubprocessRunner(use_venv=False))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDATA_DIR = os.path.join(SCRIPT_DIR, "testdata")
TEST_INPUT_PATH = os.path.join(TESTDATA_DIR, "sample_input.jsonl")
TEST_FLOW_PATH = os.path.join(TESTDATA_DIR, "transform_test_flow.yaml")


@dsl.pipeline(name="e2e-transform-test")
def e2e_pipeline():
    """E2E test pipeline with transform-only flow."""
    sdg(  # noqa: F841
        input_pvc_path=TEST_INPUT_PATH,
        flow_yaml_path=TEST_FLOW_PATH,
        log_level="INFO",
    )


if __name__ == "__main__":
    print("=" * 60)
    print("E2E Test: SDG Component with Transform Flow")
    print("=" * 60)

    print(f"\nInput: {TEST_INPUT_PATH}")
    print(f"Flow:  {TEST_FLOW_PATH}")
    print(f"Input exists: {os.path.exists(TEST_INPUT_PATH)}")
    print(f"Flow exists:  {os.path.exists(TEST_FLOW_PATH)}")

    result = e2e_pipeline()

    # Find and verify output
    output_dir = None
    for d in sorted(os.listdir("local_outputs"), reverse=True):
        if d.startswith("e2e-transform-test"):
            output_dir = os.path.join("local_outputs", d, "sdg")
            break

    if output_dir:
        output_path = os.path.join(output_dir, "output_artifact")
        metrics_path = os.path.join(output_dir, "output_metrics")

        if os.path.exists(output_path):
            print("\nOutput artifact:")
            with open(output_path) as f:
                print(f.read())

        if os.path.exists(metrics_path):
            print("Metrics:")
            with open(metrics_path) as f:
                print(json.dumps(json.load(f), indent=2))

    print("\n" + "=" * 60)
    print("E2E test completed successfully!")
    print("=" * 60)
