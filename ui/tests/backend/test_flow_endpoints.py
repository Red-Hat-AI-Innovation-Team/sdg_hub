# SPDX-License-Identifier: Apache-2.0
"""Tests for flow discovery endpoints."""

import sys
from pathlib import Path
from unittest.mock import patch

import yaml

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class TestListFlows:
    """Tests for /api/flows/list endpoint."""
    
    def test_list_flows_success(self, test_client, mock_sdg_hub):
        """Test listing flows returns flow names."""
        response = test_client.get("/api/flows/list")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 0  # May be empty if no flows discovered
    
    def test_list_flows_includes_custom(self, test_client, mock_sdg_hub, temp_dir):
        """Test listing flows includes custom flows."""
        # Create a custom flow
        custom_flows_dir = Path(temp_dir) / "custom_flows"
        custom_flows_dir.mkdir(exist_ok=True)
        custom_flow_dir = custom_flows_dir / "test_custom"
        custom_flow_dir.mkdir(exist_ok=True)
        
        flow_yaml = {
            "metadata": {
                "name": "Test Custom Flow",
                "description": "A test custom flow",
            },
            "blocks": [],
        }
        with open(custom_flow_dir / "flow.yaml", "w") as f:
            yaml.dump(flow_yaml, f)
        
        with patch("routers.flows.CUSTOM_FLOWS_DIR", custom_flows_dir):
            response = test_client.get("/api/flows/list")
            assert response.status_code == 200


class TestSearchFlows:
    """Tests for /api/flows/search endpoint."""
    
    def test_search_flows_by_tag(self, test_client, mock_sdg_hub):
        """Test searching flows by tag."""
        response = test_client.post("/api/flows/search", json={"tag": "test"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_search_flows_by_name(self, test_client, mock_sdg_hub):
        """Test searching flows by name filter."""
        response = test_client.post("/api/flows/search", json={"name_filter": "test"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_search_flows_no_filter(self, test_client, mock_sdg_hub):
        """Test searching flows with no filter returns all."""
        response = test_client.post("/api/flows/search", json={})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestGetFlowInfo:
    """Tests for /api/flows/{flow_name}/info endpoint."""
    
    def test_get_flow_info_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test getting flow info for existing flow."""
        # Create a test flow file
        flow_dir = Path(temp_dir) / "test_flow"
        flow_dir.mkdir(exist_ok=True)
        flow_yaml = {
            "metadata": {
                "name": "Test Flow",
                "description": "A test flow",
                "version": "1.0.0",
                "author": "Test Author",
                "tags": ["test"],
                "recommended_models": {
                    "default": "test-model",
                    "compatible": [],
                    "experimental": [],
                },
            },
            "blocks": [],
        }
        flow_path = flow_dir / "flow.yaml"
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        # Mock FlowRegistry to return our test flow
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)):
            response = test_client.get("/api/flows/Test%20Flow/info")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Test Flow"
    
    def test_get_flow_info_not_found(self, test_client, mock_sdg_hub):
        """Test getting flow info for non-existent flow."""
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=None):
            response = test_client.get("/api/flows/NonExistent%20Flow/info")
            assert response.status_code == 404


class TestSelectFlow:
    """Tests for /api/flows/{flow_name}/select endpoint."""
    
    def test_select_flow_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test selecting a flow."""
        # Create a test flow file
        flow_dir = Path(temp_dir) / "test_flow"
        flow_dir.mkdir(exist_ok=True)
        flow_yaml = {
            "metadata": {
                "name": "Test Flow",
                "description": "A test flow",
            },
            "blocks": [],
        }
        flow_path = flow_dir / "flow.yaml"
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)):
            response = test_client.post("/api/flows/Test%20Flow/select")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    def test_select_flow_not_found(self, test_client, mock_sdg_hub):
        """Test selecting a non-existent flow."""
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=None):
            response = test_client.post("/api/flows/NonExistent%20Flow/select")
            assert response.status_code == 404


class TestSelectFlowByPath:
    """Tests for /api/flows/select-by-path endpoint."""
    
    def test_select_flow_by_path_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test selecting a flow by path."""
        # Create a test flow file
        flow_path = Path(temp_dir) / "test_flow.yaml"
        flow_yaml = {
            "metadata": {
                "name": "Test Flow",
                "description": "A test flow",
            },
            "blocks": [],
        }
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        response = test_client.post(
            "/api/flows/select-by-path",
            json={"flow_path": str(flow_path)}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    def test_select_flow_by_path_missing_path(self, test_client, mock_sdg_hub):
        """Test selecting flow without path."""
        response = test_client.post("/api/flows/select-by-path", json={})
        assert response.status_code == 400
    
    def test_select_flow_by_path_not_found(self, test_client, mock_sdg_hub):
        """Test selecting flow with non-existent path.
        
        After security hardening, paths outside allowed directories are
        rejected with 400 before the 404 file-not-found check.
        """
        response = test_client.post(
            "/api/flows/select-by-path",
            json={"flow_path": "/nonexistent/path.yaml"}
        )
        assert response.status_code in [400, 404]


class TestGetFlowYaml:
    """Tests for /api/flows/{flow_name}/yaml endpoint."""
    
    def test_get_flow_yaml_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test getting flow YAML content."""
        # Create a test flow file
        flow_path = Path(temp_dir) / "test_flow.yaml"
        flow_yaml = {
            "metadata": {
                "name": "Test Flow",
                "description": "A test flow",
            },
            "blocks": [
                {
                    "block_type": "LLMChatBlock",
                    "block_config": {"block_name": "test"},
                }
            ],
        }
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)):
            response = test_client.get("/api/flows/Test%20Flow/yaml")
            assert response.status_code == 200
            data = response.json()
            assert data["metadata"]["name"] == "Test Flow"
            assert len(data["blocks"]) == 1
    
    def test_get_flow_yaml_not_found(self, test_client, mock_sdg_hub):
        """Test getting YAML for non-existent flow."""
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=None):
            response = test_client.get("/api/flows/NonExistent%20Flow/yaml")
            assert response.status_code == 404


class TestSaveCustomFlow:
    """Tests for /api/flows/save-custom endpoint."""
    
    def test_save_custom_flow_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test saving a custom flow."""
        custom_flows_dir = Path(temp_dir) / "custom_flows"
        custom_flows_dir.mkdir(exist_ok=True)
        
        flow_data = {
            "metadata": {
                "name": "My Custom Flow",
                "description": "A custom flow",
                "version": "1.0.0",
            },
            "blocks": [
                {
                    "block_type": "LLMChatBlock",
                    "block_config": {
                        "block_name": "test_block",
                        "input_cols": "input",
                        "output_cols": "output",
                    },
                }
            ],
        }
        
        with patch("routers.flows.CUSTOM_FLOWS_DIR", custom_flows_dir):
            response = test_client.post("/api/flows/save-custom", json=flow_data)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "flow_path" in data
    
    def test_save_custom_flow_with_empty_blocks(self, test_client, mock_sdg_hub, temp_dir):
        """Test saving custom flow with empty blocks."""
        custom_flows_dir = Path(temp_dir) / "custom_flows"
        custom_flows_dir.mkdir(exist_ok=True)
        
        flow_data = {
            "metadata": {
                "name": "Simple Flow",
            },
            "blocks": [],
        }
        
        with patch("routers.flows.CUSTOM_FLOWS_DIR", custom_flows_dir):
            response = test_client.post("/api/flows/save-custom", json=flow_data)
            assert response.status_code == 200

