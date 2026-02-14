# SPDX-License-Identifier: Apache-2.0
"""Tests for checkpoint management endpoints."""

import sys
from pathlib import Path
from unittest.mock import patch


# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class TestGetCheckpointInfo:
    """Tests for /api/flow/checkpoints/{config_id} endpoint."""
    
    def test_get_checkpoint_info_no_checkpoints(self, test_client, mock_sdg_hub, temp_dir):
        """Test getting checkpoint info when no checkpoints exist."""
        with patch("utils.checkpoint_utils.CHECKPOINTS_DIR", Path(temp_dir)):
            response = test_client.get("/api/flow/checkpoints/test-config")
            assert response.status_code == 200
            data = response.json()
            assert data["has_checkpoints"] is False
            assert data["checkpoint_count"] == 0
    
    def test_get_checkpoint_info_with_checkpoints(self, test_client, mock_sdg_hub, temp_dir):
        """Test getting checkpoint info when checkpoints exist."""
        checkpoints_dir = Path(temp_dir) / "test-config"
        checkpoints_dir.mkdir(exist_ok=True)
        
        # Create mock checkpoint files
        checkpoint_file = checkpoints_dir / "checkpoint_0001.jsonl"
        with open(checkpoint_file, "w") as f:
            f.write('{"output": "test1"}\n')
            f.write('{"output": "test2"}\n')
        
        with patch("utils.checkpoint_utils.CHECKPOINTS_DIR", Path(temp_dir)):
            response = test_client.get("/api/flow/checkpoints/test-config")
            assert response.status_code == 200
            data = response.json()
            assert data["has_checkpoints"] is True
            assert data["checkpoint_count"] == 1
            assert data["samples_completed"] == 2


class TestClearCheckpoints:
    """Tests for /api/flow/checkpoints/{config_id} DELETE endpoint."""
    
    def test_clear_checkpoints_success(self, test_client, mock_sdg_hub, temp_dir):
        """Test clearing checkpoints successfully."""
        checkpoints_dir = Path(temp_dir) / "test-config"
        checkpoints_dir.mkdir(exist_ok=True)
        
        # Create mock checkpoint files
        checkpoint_file = checkpoints_dir / "checkpoint_0001.jsonl"
        with open(checkpoint_file, "w") as f:
            f.write('{"output": "test"}\n')
        
        with patch("utils.checkpoint_utils.CHECKPOINTS_DIR", Path(temp_dir)):
            response = test_client.delete("/api/flow/checkpoints/test-config")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            
            # Verify directory is deleted
            assert not checkpoints_dir.exists()
    
    def test_clear_checkpoints_no_existing(self, test_client, mock_sdg_hub, temp_dir):
        """Test clearing checkpoints when none exist."""
        with patch("utils.checkpoint_utils.CHECKPOINTS_DIR", Path(temp_dir)):
            response = test_client.delete("/api/flow/checkpoints/nonexistent-config")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

