#!/usr/bin/env python3
"""Sample KFP pipeline definition using the SDG Hub component.

This file demonstrates how to compose the sdg component into a Kubeflow Pipeline
with PVC mounting, secrets, and pipeline compilation.

Usage:
    # Compile pipeline to YAML
    uv run python kfp/pipeline.py

    # The compiled YAML can be uploaded to a KFP instance
"""

from sdg_hub.kfp import sdg

from kfp import compiler, dsl
from kfp.kubernetes import mount_pvc, use_secret_as_env


@dsl.pipeline(
    name="sdg-hub-pipeline",
    description="Synthetic data generation pipeline using SDG Hub flows.",
)
def sdg_pipeline(
    flow_id: str = "",
    flow_yaml_path: str = "",
    model: str = "",
    input_pvc_path: str = "/mnt/data/input.jsonl",
    max_concurrency: int = 10,
    checkpoint_pvc_path: str = "",
    save_freq: int = 100,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    log_level: str = "INFO",
):
    """SDG Hub pipeline with configurable flow and model parameters."""
    sdg_task = sdg(
        input_pvc_path=input_pvc_path,
        flow_id=flow_id,
        flow_yaml_path=flow_yaml_path,
        model=model,
        max_concurrency=max_concurrency,
        checkpoint_pvc_path=checkpoint_pvc_path,
        save_freq=save_freq,
        temperature=temperature,
        max_tokens=max_tokens,
        log_level=log_level,
    )

    # Mount input data PVC
    mount_pvc(
        sdg_task,
        pvc_name="sdg-data-pvc",
        mount_path="/mnt/data",
    )

    # Mount checkpoint PVC (optional, for resume support)
    mount_pvc(
        sdg_task,
        pvc_name="sdg-checkpoint-pvc",
        mount_path="/mnt/checkpoints",
    )

    # Mount LLM credentials from Kubernetes Secret
    use_secret_as_env(
        sdg_task,
        secret_name="llm-credentials",
        secret_key_to_env={
            "api-key": "LLM_API_KEY",
            "api-base": "LLM_API_BASE",
        },
    )


if __name__ == "__main__":
    output_path = "kfp/pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=sdg_pipeline,
        package_path=output_path,
    )
    print(f"Pipeline compiled to: {output_path}")
