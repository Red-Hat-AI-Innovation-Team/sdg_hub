# SPDX-License-Identifier: Apache-2.0
"""Model configuration request/response models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ModelConfigRequest(BaseModel):
    """Model configuration request."""

    model: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    blocks: Optional[List[str]] = None
    additional_params: Optional[Dict[str, Any]] = {}


class ModelTestRequest(BaseModel):
    """Request model for testing model connection."""

    model: str
    api_base: str
    api_key: str = "EMPTY"
    test_prompt: str = "What is the capital of France? Answer in one word."


class ModelTestResponse(BaseModel):
    """Response model for model connection test."""

    success: bool
    response: Optional[str] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None
