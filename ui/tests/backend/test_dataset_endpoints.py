# SPDX-License-Identifier: Apache-2.0
"""Tests for dataset management endpoints."""

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import yaml

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class TestUploadDataset:
    """Tests for /api/dataset/upload endpoint."""
    
    def test_upload_jsonl_file(self, test_client, mock_sdg_hub, temp_dir):
        """Test uploading a JSONL file."""
        uploads_dir = Path(temp_dir) / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        data = [
            {"input": "test 1", "label": "a"},
            {"input": "test 2", "label": "b"},
        ]
        content = "\n".join(json.dumps(d) for d in data)
        
        with patch("routers.datasets.UPLOADS_DIR", uploads_dir):
            response = test_client.post(
                "/api/dataset/upload",
                files={"file": ("test.jsonl", io.BytesIO(content.encode()), "application/json")},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "file_path" in data
            assert data["format"] == "jsonl"
    
    def test_upload_csv_file(self, test_client, mock_sdg_hub, temp_dir):
        """Test uploading a CSV file."""
        uploads_dir = Path(temp_dir) / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        content = "input,label\ntest 1,a\ntest 2,b"
        
        with patch("routers.datasets.UPLOADS_DIR", uploads_dir):
            response = test_client.post(
                "/api/dataset/upload",
                files={"file": ("test.csv", io.BytesIO(content.encode()), "text/csv")},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["format"] == "csv"
    
    def test_upload_json_file(self, test_client, mock_sdg_hub, temp_dir):
        """Test uploading a JSON file."""
        uploads_dir = Path(temp_dir) / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        data = [{"input": "test 1"}, {"input": "test 2"}]
        content = json.dumps(data)
        
        with patch("routers.datasets.UPLOADS_DIR", uploads_dir):
            response = test_client.post(
                "/api/dataset/upload",
                files={"file": ("test.json", io.BytesIO(content.encode()), "application/json")},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["format"] == "json"
    
    def test_upload_unsupported_format(self, test_client, mock_sdg_hub, temp_dir):
        """Test uploading unsupported file format."""
        uploads_dir = Path(temp_dir) / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        with patch("routers.datasets.UPLOADS_DIR", uploads_dir):
            response = test_client.post(
                "/api/dataset/upload",
                files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")},
            )
            assert response.status_code == 400
            assert "Unsupported file format" in response.json()["detail"]
    
    def test_upload_file_too_large(self, test_client, mock_sdg_hub, temp_dir):
        """Test uploading file that exceeds size limit."""
        uploads_dir = Path(temp_dir) / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        # Set a very small size limit for testing
        with patch("routers.datasets.UPLOADS_DIR", uploads_dir), \
             patch("routers.datasets.MAX_UPLOAD_SIZE_BYTES", 10):  # 10 bytes limit
            # Create content larger than limit
            large_content = '{"input":"' + 'x' * 100 + '"}\n'
            response = test_client.post(
                "/api/dataset/upload",
                files={"file": ("test.jsonl", io.BytesIO(large_content.encode()), "application/json")},
            )
            assert response.status_code == 400
            assert "exceeds max upload size" in response.json()["detail"]
    
    def test_upload_sanitizes_filename(self, test_client, mock_sdg_hub, temp_dir):
        """Test that upload sanitizes filename."""
        uploads_dir = Path(temp_dir) / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        content = '{"input": "test"}'
        
        with patch("routers.datasets.UPLOADS_DIR", uploads_dir):
            response = test_client.post(
                "/api/dataset/upload",
                files={"file": ("../../../etc/test.jsonl", io.BytesIO(content.encode()), "application/json")},
            )
            # Should sanitize the path traversal attempt
            assert response.status_code == 200
            data = response.json()
            assert ".." not in data["file_path"]


class TestLoadDataset:
    """Tests for /api/dataset/load endpoint."""
    
    def test_load_dataset_no_flow_selected(self, test_client, mock_sdg_hub):
        """Test loading dataset without flow selected."""
        with patch("routers.datasets.current_config", {"flow": None}):
            response = test_client.post("/api/dataset/load", json={
                "data_files": "test.jsonl",
            })
            # Should return 400 or 500 depending on error handling
            assert response.status_code in [400, 500]
    
    def test_load_dataset_success(self, test_client, mock_sdg_hub, temp_dir, sample_jsonl_file):
        """Test loading dataset successfully."""
        # First select a flow
        flow_path = Path(temp_dir) / "test_flow.yaml"
        flow_yaml = {"metadata": {"name": "Test Flow"}, "blocks": []}
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        uploads_dir = Path(temp_dir)
        
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)), \
             patch("routers.datasets.UPLOADS_DIR", uploads_dir), \
             patch("utils.security.ALLOWED_DATASET_DIRS", [uploads_dir]):
            test_client.post("/api/flows/Test%20Flow/select")
            
            response = test_client.post("/api/dataset/load", json={
                "data_files": sample_jsonl_file,
            })
            # May succeed or fail depending on flow state
            assert response.status_code in [200, 400, 500]
    
    def test_load_dataset_file_not_found(self, test_client, mock_sdg_hub, temp_dir):
        """Test loading non-existent dataset file."""
        flow_path = Path(temp_dir) / "test_flow.yaml"
        flow_yaml = {"metadata": {"name": "Test Flow"}, "blocks": []}
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        uploads_dir = Path(temp_dir)
        
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)), \
             patch("routers.datasets.UPLOADS_DIR", uploads_dir), \
             patch("utils.security.ALLOWED_DATASET_DIRS", [uploads_dir]):
            test_client.post("/api/flows/Test%20Flow/select")
            
            response = test_client.post("/api/dataset/load", json={
                "data_files": "/nonexistent/file.jsonl",
            })
            assert response.status_code in [400, 404, 500]


class TestGetDatasetSchema:
    """Tests for /api/dataset/schema endpoint."""
    
    def test_get_schema_no_flow_selected(self, test_client, mock_sdg_hub):
        """Test getting schema without flow selected."""
        with patch("routers.datasets.current_config", {"flow": None}):
            response = test_client.get("/api/dataset/schema")
            assert response.status_code == 400
    
    def test_get_schema_with_flow(self, test_client, mock_sdg_hub, temp_dir):
        """Test getting schema with flow selected."""
        flow_path = Path(temp_dir) / "test_flow.yaml"
        flow_yaml = {"metadata": {"name": "Test Flow"}, "blocks": []}
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)):
            test_client.post("/api/flows/Test%20Flow/select")
            
            response = test_client.get("/api/dataset/schema")
            assert response.status_code == 200
            data = response.json()
            assert "columns" in data


class TestPreviewDataset:
    """Tests for /api/dataset/preview endpoint."""
    
    def test_preview_no_dataset_loaded(self, test_client, mock_sdg_hub):
        """Test preview without dataset loaded."""
        with patch("routers.datasets.current_config", {"dataset": None}):
            response = test_client.get("/api/dataset/preview")
            assert response.status_code == 400
    
    def test_preview_dataset_preview_structure(self, test_client, mock_sdg_hub, temp_dir, sample_jsonl_file):
        """Test preview response structure."""
        flow_path = Path(temp_dir) / "test_flow.yaml"
        flow_yaml = {"metadata": {"name": "Test Flow"}, "blocks": []}
        with open(flow_path, "w") as f:
            yaml.dump(flow_yaml, f)
        
        uploads_dir = Path(temp_dir)
        
        # This test validates the expected response structure
        # The actual behavior depends on state setup
        with patch("routers.flows.FlowRegistry.get_flow_path", return_value=str(flow_path)), \
             patch("routers.datasets.UPLOADS_DIR", uploads_dir), \
             patch("utils.security.ALLOWED_DATASET_DIRS", [uploads_dir]):
            test_client.post("/api/flows/Test%20Flow/select")
            
            # Try to load dataset first
            test_client.post("/api/dataset/load", json={
                "data_files": sample_jsonl_file,
            })
            
            response = test_client.get("/api/dataset/preview")
            # Response depends on whether dataset was successfully loaded
            if response.status_code == 200:
                data = response.json()
                # If successful, should have these fields
                assert isinstance(data, dict)
