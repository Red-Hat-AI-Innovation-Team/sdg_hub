# SPDX-License-Identifier: Apache-2.0
"""Tests for configuration management endpoints."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import yaml

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class MockSavedConfiguration:
    """Mock SavedConfiguration for testing."""
    
    def __init__(
        self,
        id: str,
        flow_name: str,
        flow_id: str = "test-flow-id",
        flow_path: str = "/path/to/flow.yaml",
        model_configuration: dict = None,
        dataset_configuration: dict = None,
        dry_run_configuration: dict = None,
        tags: list = None,
        status: str = "configured",
        created_at: str = "2024-01-01T00:00:00",
        updated_at: str = "2024-01-01T00:00:00",
    ):
        self.id = id
        self.flow_name = flow_name
        self.flow_id = flow_id
        self.flow_path = flow_path
        self.model_configuration = model_configuration or {"model": "test-model"}
        self.dataset_configuration = dataset_configuration or {}
        self.dry_run_configuration = dry_run_configuration
        self.tags = tags or []
        self.status = status
        self.created_at = created_at
        self.updated_at = updated_at
    
    def dict(self):
        return {
            "id": self.id,
            "flow_name": self.flow_name,
            "flow_id": self.flow_id,
            "flow_path": self.flow_path,
            "model_configuration": self.model_configuration,
            "dataset_configuration": self.dataset_configuration,
            "dry_run_configuration": self.dry_run_configuration,
            "tags": self.tags,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TestListConfigurations:
    """Tests for /api/configurations/list endpoint."""
    
    def test_list_configurations_empty(self, test_client, mock_sdg_hub, temp_dir):
        """Test listing configurations when none exist."""
        config_file = Path(temp_dir) / "saved_configurations.json"
        with open(config_file, "w") as f:
            json.dump([], f)
        
        with patch("routers.configurations.saved_configurations", {}):
            response = test_client.get("/api/configurations/list")
            assert response.status_code == 200
            data = response.json()
            assert "configurations" in data
            assert isinstance(data["configurations"], list)
    
    def test_list_configurations_with_data(self, test_client, mock_sdg_hub, temp_dir):
        """Test listing configurations with existing data."""
        config_file = Path(temp_dir) / "saved_configurations.json"
        
        # Create mock saved configurations as a dict
        configs = {
            "config-1": MockSavedConfiguration(
                id="config-1",
                flow_name="Test Flow",
                status="configured",
            ),
            "config-2": MockSavedConfiguration(
                id="config-2",
                flow_name="Another Flow",
                status="not_configured",
            ),
        }
        
        with patch("routers.configurations.saved_configurations", configs):
            response = test_client.get("/api/configurations/list")
            assert response.status_code == 200
            data = response.json()
            assert "configurations" in data


class TestGetConfiguration:
    """Tests for /api/configurations/{config_id} endpoint."""
    
    def test_get_configuration_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test getting an existing configuration."""
        config = MockSavedConfiguration(
            id="config-1",
            flow_name="Test Flow",
            model_configuration={"model": "test-model", "api_key": "env:TEST_KEY"},
        )
        configs = {"config-1": config}
        
        with patch("routers.configurations.saved_configurations", configs):
            response = test_client.get("/api/configurations/config-1")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "configuration" in data
    
    def test_get_configuration_not_found(self, test_client, mock_sdg_hub, temp_dir):
        """Test getting a non-existent configuration."""
        with patch("routers.configurations.saved_configurations", {}):
            response = test_client.get("/api/configurations/nonexistent")
            assert response.status_code == 404


class TestSaveConfiguration:
    """Tests for /api/configurations/save endpoint."""
    
    def test_save_configuration_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test saving a new configuration."""
        config_file = Path(temp_dir) / "saved_configurations.json"
        with open(config_file, "w") as f:
            json.dump([], f)
        
        new_config = {
            "flow_name": "Test Flow",
            "flow_id": "test-flow-id",
            "flow_path": "/path/to/flow.yaml",
            "model_configuration": {
                "model": "test-model",
                "api_key": "EMPTY",
            },
            "dataset_configuration": {
                "data_files": "test.jsonl",
            },
            "status": "configured",
        }
        
        with patch("routers.configurations.saved_configurations", {}), \
             patch("routers.configurations._persist_saved_configs"):
            response = test_client.post("/api/configurations/save", json=new_config)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "configuration" in data
    
    def test_save_configuration_with_env_reference(self, test_client, mock_sdg_hub, temp_dir):
        """Test that save keeps env references."""
        config_file = Path(temp_dir) / "saved_configurations.json"
        with open(config_file, "w") as f:
            json.dump([], f)
        
        new_config = {
            "flow_name": "Test Flow",
            "flow_id": "test-flow-id",
            "flow_path": "/path/to/flow.yaml",
            "model_configuration": {
                "model": "test-model",
                "api_key": "env:OPENAI_API_KEY",
            },
            "dataset_configuration": {},
            "status": "configured",
        }
        
        with patch("routers.configurations.saved_configurations", {}), \
             patch("routers.configurations._persist_saved_configs"):
            response = test_client.post("/api/configurations/save", json=new_config)
            assert response.status_code == 200


class TestDeleteConfiguration:
    """Tests for /api/configurations/{config_id} DELETE endpoint."""
    
    def test_delete_configuration_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test deleting an existing configuration."""
        config = MockSavedConfiguration(
            id="config-1",
            flow_name="Test Flow",
        )
        configs = {"config-1": config}
        
        with patch("routers.configurations.saved_configurations", configs), \
             patch("routers.configurations._persist_saved_configs"):
            response = test_client.delete("/api/configurations/config-1")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    def test_delete_configuration_not_found(self, test_client, mock_sdg_hub, temp_dir):
        """Test deleting a non-existent configuration."""
        with patch("routers.configurations.saved_configurations", {}):
            response = test_client.delete("/api/configurations/nonexistent")
            assert response.status_code == 404


class TestLoadConfiguration:
    """Tests for /api/configurations/{config_id}/load endpoint."""
    
    def test_load_configuration_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test loading an existing configuration."""
        # Create a flow file
        flow_path = Path(temp_dir) / "test_flow.yaml"
        flow_yaml = {"metadata": {"name": "Test Flow"}, "blocks": []}
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        config = MockSavedConfiguration(
            id="config-1",
            flow_name="Test Flow",
            flow_path=str(flow_path),
            model_configuration={"model": "test-model"},
            dataset_configuration={},
        )
        configs = {"config-1": config}
        
        with patch("routers.configurations.saved_configurations", configs):
            response = test_client.post("/api/configurations/config-1/load")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    def test_load_configuration_not_found(self, test_client, mock_sdg_hub, temp_dir):
        """Test loading a non-existent configuration."""
        with patch("routers.configurations.saved_configurations", {}):
            response = test_client.post("/api/configurations/nonexistent/load")
            assert response.status_code == 404


class TestGetCurrentConfig:
    """Tests for /api/config/current endpoint."""
    
    def test_get_current_config(self, test_client, mock_sdg_hub):
        """Test getting current configuration state."""
        response = test_client.get("/api/config/current")
        assert response.status_code == 200
        data = response.json()
        # Should return current config state
        assert isinstance(data, dict)


class TestResetConfig:
    """Tests for /api/config/reset endpoint."""
    
    def test_reset_config(self, test_client, mock_sdg_hub):
        """Test resetting configuration."""
        response = test_client.post("/api/config/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
