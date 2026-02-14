# SPDX-License-Identifier: Apache-2.0
"""Workspace-related request models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CreateWorkspaceRequest(BaseModel):
    source_flow_name: Optional[str] = None  # Template to clone, None for empty workspace

class UpdateWorkspaceFlowRequest(BaseModel):
    metadata: Dict[str, Any]
    blocks: List[Dict[str, Any]]

class UpdateWorkspacePromptRequest(BaseModel):
    prompt_filename: str
    prompt_config: Dict[str, Any]

class FinalizeWorkspaceRequest(BaseModel):
    flow_name: str
