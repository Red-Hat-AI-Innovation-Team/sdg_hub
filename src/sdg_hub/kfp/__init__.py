# SPDX-License-Identifier: Apache-2.0
"""SDG Hub Kubeflow Pipelines Component.

This module provides a KFP component for running SDG Hub flows in Kubeflow Pipelines.
The component is designed to be self-contained and can be extracted to a separate
repository if needed.

Usage:
    from sdg_hub.kfp import sdg

    @dsl.pipeline
    def my_pipeline():
        sdg_task = sdg(
            input_pvc_path="/mnt/data/input.jsonl",
            flow_id="extractive-summary-qa",
        )
"""

from sdg_hub.kfp.component import sdg

__all__ = ["sdg"]
