# SPDX-License-Identifier: Apache-2.0
"""SDG Hub KFP Component Definition.

This module contains the main KFP component for running SDG Hub flows.
It is designed to be self-contained and extractable to a separate repository.
"""

from kfp import dsl
from kfp.dsl import Dataset, Metrics, Output

# Component image - update this for production
# Use localhost/ prefix for local podman images
SDG_HUB_IMAGE = "localhost/sdg-hub-kfp:dev"


@dsl.component(base_image=SDG_HUB_IMAGE)
def sdg(
    # ==================== OUTPUT ====================
    output_artifact: Output[Dataset],
    output_metrics: Output[Metrics],
    # ==================== INPUT OPTIONS ====================
    input_pvc_path: str = "",
    # ==================== FLOW SELECTION ====================
    flow_id: str = "",
    flow_yaml_path: str = "",
    # ==================== MODEL CONFIGURATION ====================
    model: str = "",
    # ==================== EXECUTION ====================
    max_concurrency: int = 10,
    checkpoint_pvc_path: str = "",
    save_freq: int = 100,
    log_level: str = "INFO",
    # ==================== COMPONENT-LEVEL LLM PARAMS ====================
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> None:
    """SDG Hub data generation component for Kubeflow Pipelines.

    Runs a synthetic data generation flow on input data, producing
    enriched output suitable for model training.

    This is currently a skeleton implementation for testing the KFP setup.

    Args:
        output_artifact: KFP Dataset artifact for downstream components
        output_metrics: KFP Metrics artifact with execution stats
        input_pvc_path: Path to JSONL file on mounted PVC
        flow_id: Built-in flow ID from SDG Hub registry
        flow_yaml_path: Path to custom flow YAML (mounted from ConfigMap)
        model: LiteLLM model identifier
        max_concurrency: Maximum concurrent LLM requests
        checkpoint_pvc_path: PVC path for checkpoints (enables resume)
        save_freq: Checkpoint save frequency (samples)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        temperature: LLM temperature (0.0-2.0)
        max_tokens: Maximum response tokens
    """
    import json
    import logging
    import os
    import time

    import pandas as pd

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("SDG Hub KFP Component")
    logger.info("=" * 60)

    # Log configuration
    logger.info(f"Input PVC Path: {input_pvc_path or 'Not provided'}")
    logger.info(f"Flow ID: {flow_id or 'Not provided'}")
    logger.info(f"Flow YAML Path: {flow_yaml_path or 'Not provided'}")
    logger.info(f"Model: {model or 'Not provided'}")
    logger.info(f"Max Concurrency: {max_concurrency}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Max Tokens: {max_tokens}")

    # =========================================================================
    # INPUT HANDLING
    # =========================================================================
    df = None
    input_rows = 0

    if input_pvc_path:
        logger.info(f"Loading input from: {input_pvc_path}")

        if not os.path.exists(input_pvc_path):
            raise FileNotFoundError(f"Input file not found: {input_pvc_path}")

        df = pd.read_json(input_pvc_path, lines=True)
        input_rows = len(df)
        logger.info(f"Loaded {input_rows} rows with columns: {list(df.columns)}")
    else:
        logger.warning("No input_pvc_path provided, creating dummy dataset")
        df = pd.DataFrame(
            {
                "message": ["No input provided - dummy data"],
                "flow_id": [flow_id or "none"],
            }
        )
        input_rows = len(df)

    # =========================================================================
    # FLOW EXECUTION (placeholder - to be implemented in next iteration)
    # =========================================================================
    logger.info("Flow execution not yet implemented - passing through input data")
    output_df = df.copy()
    output_rows = len(output_df)

    # =========================================================================
    # OUTPUT HANDLING
    # =========================================================================
    output_df.to_json(output_artifact.path, orient="records", lines=True)
    logger.info(f"Output written to: {output_artifact.path}")
    logger.info(f"Output: {output_rows} rows with columns: {list(output_df.columns)}")

    # Write metrics
    execution_time = time.time() - start_time
    metrics_data = {
        "metrics": [
            {"name": "input_rows", "numberValue": input_rows},
            {"name": "output_rows", "numberValue": output_rows},
            {"name": "execution_time_seconds", "numberValue": round(execution_time, 2)},
        ]
    }
    with open(output_metrics.path, "w") as f:
        json.dump(metrics_data, f)
    logger.info(f"Metrics written to: {output_metrics.path}")

    logger.info("=" * 60)
    logger.info(f"SDG Hub KFP Component completed in {execution_time:.2f}s")
    logger.info("=" * 60)
