# SPDX-License-Identifier: Apache-2.0
"""Execution-related request models (dry run, step-by-step testing)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DryRunRequest(BaseModel):
    """Dry run request."""

    sample_size: int = 2
    enable_time_estimation: bool = False
    max_concurrency: Optional[int] = None


class TestStepByStepRequest(BaseModel):
    """Request body for step-by-step test execution."""
    blocks: List[Dict[str, Any]]  # List of block configs (used if workspace_id not provided)
    model_config_data: Dict[str, Any]  # Model configuration (model, api_base, api_key)
    sample_data: Dict[str, Any]  # Sample input data (column -> value)
    workspace_id: Optional[str] = None  # If provided, load blocks from workspace
