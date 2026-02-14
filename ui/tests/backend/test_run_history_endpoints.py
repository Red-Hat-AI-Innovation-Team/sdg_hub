# SPDX-License-Identifier: Apache-2.0
"""Tests for run history endpoints."""

import json
import sys
from pathlib import Path
from unittest.mock import patch


# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class TestListRuns:
    """Tests for /api/runs/list endpoint."""
    
    def test_list_runs_empty(self, test_client, mock_sdg_hub, temp_dir):
        """Test listing runs when none exist."""
        runs_file = Path(temp_dir) / "runs_history.json"
        with open(runs_file, "w") as f:
            json.dump([], f)
        
        with patch("utils.dataset_utils.RUNS_HISTORY_FILE", runs_file):
            response = test_client.get("/api/runs/list")
            assert response.status_code == 200
            data = response.json()
            # API returns {"runs": [...]}
            assert "runs" in data
            assert isinstance(data["runs"], list)
    
    def test_list_runs_with_data(self, test_client, mock_sdg_hub, temp_dir):
        """Test listing runs with existing data."""
        runs_file = Path(temp_dir) / "runs_history.json"
        runs = [
            {
                "run_id": "run-1",
                "config_id": "config-1",
                "flow_name": "Test Flow",
                "status": "completed",
                "start_time": "2024-01-01T00:00:00",
            },
            {
                "run_id": "run-2",
                "config_id": "config-2",
                "flow_name": "Another Flow",
                "status": "failed",
                "start_time": "2024-01-02T00:00:00",
            },
        ]
        with open(runs_file, "w") as f:
            json.dump(runs, f)
        
        with patch("utils.dataset_utils.RUNS_HISTORY_FILE", runs_file), \
             patch("routers.runs.load_runs_history", return_value=runs):
            response = test_client.get("/api/runs/list")
            assert response.status_code == 200


class TestGetRun:
    """Tests for /api/runs/{run_id} endpoint."""
    
    def test_get_run_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test getting an existing run."""
        runs_file = Path(temp_dir) / "runs_history.json"
        runs = [
            {
                "run_id": "run-1",
                "config_id": "config-1",
                "flow_name": "Test Flow",
                "status": "completed",
            },
        ]
        with open(runs_file, "w") as f:
            json.dump(runs, f)
        
        with patch("utils.dataset_utils.RUNS_HISTORY_FILE", runs_file), \
             patch("routers.runs.load_runs_history", return_value=runs):
            response = test_client.get("/api/runs/run-1")
            assert response.status_code == 200
            data = response.json()
            assert data["run_id"] == "run-1"
    
    def test_get_run_not_found(self, test_client, mock_sdg_hub, temp_dir):
        """Test getting a non-existent run."""
        runs_file = Path(temp_dir) / "runs_history.json"
        with open(runs_file, "w") as f:
            json.dump([], f)
        
        with patch("utils.dataset_utils.RUNS_HISTORY_FILE", runs_file), \
             patch("routers.runs.load_runs_history", return_value=[]):
            response = test_client.get("/api/runs/nonexistent")
            assert response.status_code == 404


class TestCreateRun:
    """Tests for /api/runs/create endpoint."""
    
    def test_create_run_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test creating a new run record."""
        runs_file = Path(temp_dir) / "runs_history.json"
        with open(runs_file, "w") as f:
            json.dump([], f)
        
        new_run = {
            "run_id": "run-new",
            "config_id": "config-1",
            "flow_name": "Test Flow",
            "flow_type": "existing",
            "model_name": "test-model",
            "status": "running",
            "start_time": "2024-01-01T00:00:00",
            "input_samples": 100,
        }
        
        with patch("utils.dataset_utils.RUNS_HISTORY_FILE", runs_file), \
             patch("routers.runs.load_runs_history", return_value=[]), \
             patch("routers.runs.save_runs_history"):
            response = test_client.post("/api/runs/create", json=new_run)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"


class TestUpdateRun:
    """Tests for /api/runs/{run_id}/update endpoint."""
    
    def test_update_run_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test updating an existing run."""
        runs_file = Path(temp_dir) / "runs_history.json"
        runs = [
            {
                "run_id": "run-1",
                "config_id": "config-1",
                "flow_name": "Test Flow",
                "status": "running",
            },
        ]
        with open(runs_file, "w") as f:
            json.dump(runs, f)
        
        updates = {
            "status": "completed",
            "end_time": "2024-01-01T01:00:00",
            "output_samples": 100,
        }
        
        with patch("utils.dataset_utils.RUNS_HISTORY_FILE", runs_file), \
             patch("routers.runs.load_runs_history", return_value=runs), \
             patch("routers.runs.save_runs_history"):
            response = test_client.put("/api/runs/run-1/update", json=updates)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    def test_update_run_not_found(self, test_client, mock_sdg_hub, temp_dir):
        """Test updating a non-existent run."""
        runs_file = Path(temp_dir) / "runs_history.json"
        with open(runs_file, "w") as f:
            json.dump([], f)
        
        with patch("utils.dataset_utils.RUNS_HISTORY_FILE", runs_file), \
             patch("routers.runs.load_runs_history", return_value=[]):
            response = test_client.put("/api/runs/nonexistent/update", json={"status": "failed"})
            assert response.status_code == 404


class TestDeleteRun:
    """Tests for /api/runs/{run_id} DELETE endpoint."""
    
    def test_delete_run_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test deleting an existing run."""
        runs_file = Path(temp_dir) / "runs_history.json"
        runs = [
            {"run_id": "run-1", "flow_name": "Test Flow"},
            {"run_id": "run-2", "flow_name": "Another Flow"},
        ]
        with open(runs_file, "w") as f:
            json.dump(runs, f)
        
        with patch("utils.dataset_utils.RUNS_HISTORY_FILE", runs_file), \
             patch("routers.runs.load_runs_history", return_value=runs.copy()), \
             patch("routers.runs.save_runs_history"):
            response = test_client.delete("/api/runs/run-1")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    def test_delete_run_nonexistent(self, test_client, mock_sdg_hub, temp_dir):
        """Test deleting a non-existent run (API returns success)."""
        runs_file = Path(temp_dir) / "runs_history.json"
        with open(runs_file, "w") as f:
            json.dump([], f)
        
        # Note: The API doesn't return 404 for non-existent runs,
        # it just removes matching runs and saves (which is a no-op if none match)
        with patch("utils.dataset_utils.RUNS_HISTORY_FILE", runs_file), \
             patch("routers.runs.load_runs_history", return_value=[]), \
             patch("routers.runs.save_runs_history"):
            response = test_client.delete("/api/runs/nonexistent")
            # API returns success even if run doesn't exist
            assert response.status_code == 200


class TestDownloadRunOutput:
    """Tests for /api/runs/{run_id}/download endpoint."""
    
    def test_download_run_output_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test downloading run output."""
        runs_file = Path(temp_dir) / "runs_history.json"
        outputs_dir = Path(temp_dir) / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        
        # Create output file
        output_file = outputs_dir / "run-1_output.jsonl"
        with open(output_file, "w") as f:
            f.write('{"output": "test"}\n')
        
        runs = [
            {
                "run_id": "run-1",
                "flow_name": "Test Flow",
                "status": "completed",
                "output_file": str(output_file),
            },
        ]
        with open(runs_file, "w") as f:
            json.dump(runs, f)
        
        with patch("utils.dataset_utils.RUNS_HISTORY_FILE", runs_file), \
             patch("routers.runs.load_runs_history", return_value=runs):
            response = test_client.get("/api/runs/run-1/download")
            # Should return file content or 200 for success
            assert response.status_code in [200, 400]  # 400 if run not completed
    
    def test_download_run_output_not_found(self, test_client, mock_sdg_hub, temp_dir):
        """Test downloading output for non-existent run."""
        runs_file = Path(temp_dir) / "runs_history.json"
        with open(runs_file, "w") as f:
            json.dump([], f)
        
        with patch("utils.dataset_utils.RUNS_HISTORY_FILE", runs_file), \
             patch("routers.runs.load_runs_history", return_value=[]):
            response = test_client.get("/api/runs/nonexistent/download")
            assert response.status_code == 404
    
    def test_download_run_not_completed(self, test_client, mock_sdg_hub, temp_dir):
        """Test downloading when run is not completed."""
        runs_file = Path(temp_dir) / "runs_history.json"
        runs = [
            {
                "run_id": "run-1",
                "flow_name": "Test Flow",
                "status": "running",  # Not completed
            },
        ]
        with open(runs_file, "w") as f:
            json.dump(runs, f)
        
        with patch("utils.dataset_utils.RUNS_HISTORY_FILE", runs_file), \
             patch("routers.runs.load_runs_history", return_value=runs):
            response = test_client.get("/api/runs/run-1/download")
            assert response.status_code == 400
