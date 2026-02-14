# SPDX-License-Identifier: Apache-2.0
"""Flow-related request/response models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FlowSearchRequest(BaseModel):
    """Request model for flow search."""

    tag: Optional[str] = None
    name_filter: Optional[str] = None


class FlowInfo(BaseModel):
    """Flow information model."""

    name: str
    id: str
    path: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = []
    recommended_models: Optional[Dict[str, Any]] = None
    dataset_requirements: Optional[Dict[str, Any]] = None
