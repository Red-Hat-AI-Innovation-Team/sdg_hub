#!/usr/bin/env python3
"""Test script to verify KFP local runner works with Podman."""

from kfp import dsl, local

# Initialize local runner with Docker (will use podman via alias)
local.init(runner=local.DockerRunner())


@dsl.component(base_image="python:3.11-slim")
def hello_world(message: str) -> str:
    """Simple hello world component."""
    print(f"Hello from KFP Local Runner: {message}")
    return f"Processed: {message}"


if __name__ == "__main__":
    # Run the component locally
    print("Running KFP component locally with DockerRunner (Podman)...")
    result = hello_world(message="Testing SDG Hub KFP!")
    print(f"\nResult: {result}")
    print("\nKFP Local Runner is working correctly!")