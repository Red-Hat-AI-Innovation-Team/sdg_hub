# SPDX-License-Identifier: Apache-2.0
"""Health check router."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "sdg_hub_api"}
