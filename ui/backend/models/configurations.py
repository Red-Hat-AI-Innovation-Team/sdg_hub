# SPDX-License-Identifier: Apache-2.0
"""Saved configuration models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SavedConfiguration(BaseModel):
    """Saved configuration model."""

    id: str
    flow_name: str
    flow_id: str
    flow_path: str
    model_configuration: Dict[str, Any]
    dataset_configuration: Dict[str, Any]
    dry_run_configuration: Optional[Dict[str, Any]] = None
    tags: List[str] = []
    status: Optional[str] = "configured"  # configured, not_configured, draft
    created_at: str
    updated_at: str


class SaveConfigurationRequest(BaseModel):
    """Request to save a configuration."""

    flow_name: str
    flow_id: str
    flow_path: str
    model_configuration: Dict[str, Any]
    dataset_configuration: Dict[str, Any]
    dry_run_configuration: Optional[Dict[str, Any]] = None
    tags: List[str] = []
    status: Optional[str] = "configured"  # configured, not_configured, draft
