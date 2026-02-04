#!/usr/bin/env python3
"""Test script to verify the SDG skeleton component works with KFP local runner."""

from kfp import dsl, local

# Initialize local runner with Docker (uses podman via alias)
local.init(runner=local.DockerRunner())

# Import the SDG component
from sdg_hub.kfp import sdg


@dsl.pipeline(name="sdg-skeleton-test")
def test_pipeline():
    """Test pipeline for the SDG skeleton component."""
    sdg_task = sdg(
        input_pvc_path="/mnt/data/test.jsonl",
        flow_id="test-flow",
        model="hosted_vllm/test-model",
        temperature=0.8,
        max_tokens=1024,
        log_level="INFO",
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SDG Hub KFP Skeleton Component")
    print("=" * 60)

    # Run the pipeline locally
    result = test_pipeline()

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)