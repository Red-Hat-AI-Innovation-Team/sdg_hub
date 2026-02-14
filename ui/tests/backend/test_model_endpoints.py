# SPDX-License-Identifier: Apache-2.0
"""Tests for model configuration endpoints."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import yaml

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class TestGetModelRecommendations:
    """Tests for /api/model/recommendations endpoint."""
    
    def test_get_recommendations_no_flow_selected(self, test_client, mock_sdg_hub):
        """Test getting recommendations without flow selected."""
        # Reset current config
        with patch("routers.models.current_config", {"flow": None}):
            response = test_client.get("/api/model/recommendations")
            assert response.status_code == 400
            assert "No flow selected" in response.json()["detail"]
    
    def test_get_recommendations_with_flow(self, test_client, mock_sdg_hub, temp_dir):
        """Test getting recommendations with flow selected."""
        # First select a flow
        flow_path = Path(temp_dir) / "test_flow.yaml"
        flow_yaml = {
            "metadata": {
                "name": "Test Flow",
                "recommended_models": {
                    "default": "test-model",
                    "compatible": ["alt-model"],
                    "experimental": [],
                },
            },
            "blocks": [],
        }
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)):
            # Select flow first
            test_client.post("/api/flows/Test%20Flow/select")
            
            # Get recommendations
            response = test_client.get("/api/model/recommendations")
            assert response.status_code == 200
            data = response.json()
            assert "default_model" in data
            assert "recommendations" in data


class TestConfigureModel:
    """Tests for /api/model/configure endpoint."""
    
    def test_configure_model_no_flow_selected(self, test_client, mock_sdg_hub):
        """Test configuring model without flow selected."""
        with patch("routers.models.current_config", {"flow": None}):
            response = test_client.post("/api/model/configure", json={
                "model": "test-model",
            })
            assert response.status_code == 400
            assert "No flow selected" in response.json()["detail"]
    
    def test_configure_model_with_flow(self, test_client, mock_sdg_hub, temp_dir):
        """Test configuring model with flow selected."""
        # First select a flow
        flow_path = Path(temp_dir) / "test_flow.yaml"
        flow_yaml = {
            "metadata": {"name": "Test Flow"},
            "blocks": [],
        }
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)):
            # Select flow first
            test_client.post("/api/flows/Test%20Flow/select")
            
            # Configure model
            response = test_client.post("/api/model/configure", json={
                "model": "test-model",
                "api_base": "http://localhost:8000/v1",
                "api_key": "EMPTY",
            })
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    def test_configure_model_with_env_key(self, test_client, mock_sdg_hub, temp_dir):
        """Test configuring model with environment variable key."""
        os.environ["TEST_API_KEY_MODEL"] = "resolved-test-key"
        try:
            # First select a flow
            flow_path = Path(temp_dir) / "test_flow.yaml"
            flow_yaml = {
                "metadata": {"name": "Test Flow"},
                "blocks": [],
            }
            with open(flow_path, "w") as f:
                yaml.dump(flow_yaml, f)
            
            with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)):
                test_client.post("/api/flows/Test%20Flow/select")
                
                response = test_client.post("/api/model/configure", json={
                    "model": "test-model",
                    "api_key": "env:TEST_API_KEY_MODEL",
                })
                assert response.status_code == 200
        finally:
            del os.environ["TEST_API_KEY_MODEL"]
    
    def test_configure_model_invalid_key_format(self, test_client, mock_sdg_hub, temp_dir):
        """Test configuring model with invalid API key format."""
        flow_path = Path(temp_dir) / "test_flow.yaml"
        flow_yaml = {
            "metadata": {"name": "Test Flow"},
            "blocks": [],
        }
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)):
            test_client.post("/api/flows/Test%20Flow/select")
            
            response = test_client.post("/api/model/configure", json={
                "model": "test-model",
                "api_key": "short",  # Too short
            })
            assert response.status_code == 400
            assert "Invalid API key" in response.json()["detail"]
    
    def test_configure_model_missing_env_var(self, test_client, mock_sdg_hub, temp_dir):
        """Test configuring model with missing environment variable."""
        flow_path = Path(temp_dir) / "test_flow.yaml"
        flow_yaml = {
            "metadata": {"name": "Test Flow"},
            "blocks": [],
        }
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)):
            test_client.post("/api/flows/Test%20Flow/select")
            
            response = test_client.post("/api/model/configure", json={
                "model": "test-model",
                "api_key": "env:NONEXISTENT_VAR_12345",
            })
            assert response.status_code == 400
            assert "Environment variable not found" in response.json()["detail"]
    
    def test_configure_model_with_additional_params(self, test_client, mock_sdg_hub, temp_dir):
        """Test configuring model with additional parameters."""
        flow_path = Path(temp_dir) / "test_flow.yaml"
        flow_yaml = {
            "metadata": {"name": "Test Flow"},
            "blocks": [],
        }
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)):
            test_client.post("/api/flows/Test%20Flow/select")
            
            response = test_client.post("/api/model/configure", json={
                "model": "test-model",
                "api_key": "EMPTY",
                "additional_params": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
            })
            assert response.status_code == 200

