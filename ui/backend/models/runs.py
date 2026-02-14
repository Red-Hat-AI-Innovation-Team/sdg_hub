# SPDX-License-Identifier: Apache-2.0
"""Run history and log analysis models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FlowRunRecord(BaseModel):
    """Flow run record for history tracking."""

    run_id: str
    config_id: str
    flow_name: str
    flow_type: str  # 'existing' or 'custom'
    model_name: str
    status: str  # 'running', 'completed', 'failed'
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    input_samples: int
    output_samples: Optional[int] = None
    output_columns: Optional[int] = None
    dataset_file: Optional[str] = None
    output_file: Optional[str] = None  # Path to generated JSONL file
    error_message: Optional[str] = None


class LogAnalysisRequest(BaseModel):
    """Request body for log analysis."""
    raw_logs: str
