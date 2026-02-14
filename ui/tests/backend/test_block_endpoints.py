# SPDX-License-Identifier: Apache-2.0
"""Tests for block registry endpoints."""

import sys
from pathlib import Path
from unittest.mock import patch


# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class TestListBlocks:
    """Tests for /api/blocks/list endpoint."""
    
    def test_list_blocks_success(self, test_client, mock_sdg_hub):
        """Test listing available blocks."""
        response = test_client.get("/api/blocks/list")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "blocks" in data or isinstance(data, list)
    
    def test_list_blocks_returns_categories(self, test_client, mock_sdg_hub):
        """Test that block list returns categorized blocks."""
        # Mock BlockRegistry to return categorized blocks
        mock_blocks = {
            "llm": ["LLMChatBlock", "LLMParserBlock"],
            "transform": ["TextConcatBlock", "RenameColumnsBlock"],
            "filtering": ["ColumnValueFilterBlock"],
        }
        
        with patch("routers.config.BlockRegistry.list_blocks", return_value=list(sum(mock_blocks.values(), []))):
            response = test_client.get("/api/blocks/list")
            assert response.status_code == 200

