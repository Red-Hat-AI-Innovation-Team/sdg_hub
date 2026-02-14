# SPDX-License-Identifier: Apache-2.0
"""Tests for health check endpoint."""

import sys
from pathlib import Path


# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check_success(self, test_client):
        """Test health check returns healthy status."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "sdg_hub_api"
    
    def test_health_check_returns_json(self, test_client):
        """Test health check returns JSON content type."""
        response = test_client.get("/health")
        assert response.headers["content-type"] == "application/json"

