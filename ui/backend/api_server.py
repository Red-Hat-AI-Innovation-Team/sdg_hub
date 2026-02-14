#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SDG Hub API Server

FastAPI server that exposes sdg_hub functionality for the UI.
Provides endpoints for flow discovery, model configuration, and dataset management.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import uvicorn

from sdg_hub import BlockRegistry, FlowRegistry

# Configure logging with DEBUG level for troubleshooting
logging.getLogger("uvicorn").setLevel(logging.DEBUG)
logging.getLogger("multipart").setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Note: nest_asyncio.apply() is called conditionally in the dry_run endpoint
# to avoid conflicts with uvloop when reload=True

# Import path constants needed for startup
from config import (  # noqa: E402
    CUSTOM_FLOWS_DIR,
    UPLOADS_DIR,
)

# Import the configurations loader (used in startup event)
from routers.configurations import _load_saved_configs_from_disk  # noqa: E402


# Initialize FastAPI app
app = FastAPI(
    title="SDG Hub API",
    description="API for SDG Hub synthetic data generation configuration and execution",
    version="1.0.0",
)

# Configure CORS
# Allow multiple frontend ports for running parallel demo instances
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server (default)
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://localhost:3001",  # Demo instance 1
        "http://127.0.0.1:3001",
        "http://localhost:3002",  # Demo instance 2
        "http://127.0.0.1:3002",
        "http://localhost:3003",  # Demo instance 3
        "http://127.0.0.1:3003",
        "http://localhost:3004",  # Demo instance 4
        "http://127.0.0.1:3004",
        "http://localhost:3005",  # Demo instance 5
        "http://127.0.0.1:3005",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Router modules (extracted endpoint groups)
# ============================================================================

from routers.health import router as health_router  # noqa: E402
from routers.flows import router as flows_router  # noqa: E402
from routers.models import router as models_router  # noqa: E402
from routers.datasets import router as datasets_router  # noqa: E402
from routers.preprocessing import router as preprocessing_router  # noqa: E402
from routers.execution import router as execution_router  # noqa: E402
from routers.workspace import router as workspace_router  # noqa: E402
from routers.config import router as config_router  # noqa: E402
from routers.runs import router as runs_router  # noqa: E402
from routers.configurations import router as configurations_router  # noqa: E402
from routers.custom_flows import router as custom_flows_router  # noqa: E402

app.include_router(health_router)
app.include_router(flows_router)
app.include_router(models_router)
app.include_router(datasets_router)
app.include_router(preprocessing_router)
app.include_router(execution_router)
app.include_router(workspace_router)
app.include_router(config_router)
app.include_router(runs_router)
app.include_router(configurations_router)
app.include_router(custom_flows_router)


# ============================================================================
# Startup Event
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize registries on startup."""
    logger.info("Starting SDG Hub API Server...")
    try:
        # Ensure working directories exist
        CUSTOM_FLOWS_DIR.mkdir(parents=True, exist_ok=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

        # Add custom flows directory to Python path so FlowRegistry can discover it
        import sys

        custom_flows_path = str(CUSTOM_FLOWS_DIR)
        if custom_flows_path not in sys.path:
            sys.path.insert(0, custom_flows_path)

        # Discover flows and blocks
        FlowRegistry.discover_flows()
        BlockRegistry.discover_blocks()
        _load_saved_configs_from_disk()
        logger.info("✅ Successfully discovered flows and blocks")
        logger.info(f"📁 Custom flows directory: {CUSTOM_FLOWS_DIR}")
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    # Apply nest_asyncio before starting uvicorn
    # This allows sdg_hub async blocks to work within FastAPI's async context
    try:
        nest_asyncio.apply()
        logger.info("✅ nest_asyncio applied successfully")
    except Exception as e:
        logger.warning(f"Could not apply nest_asyncio: {e}")

    logger.info("🚀 Starting server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level="info")
