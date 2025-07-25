# SPDX-License-Identifier: Apache-2.0
"""Tests for flow registry."""

# Standard
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

# Third Party
import pytest
import yaml

# Local
from src.sdg_hub.flow.registry import FlowRegistry


class TestFlowRegistry:
    """Test FlowRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear registry state
        FlowRegistry._entries.clear()
        FlowRegistry._search_paths.clear()
        
        # Create temporary directory for test flows
        self.temp_dir = tempfile.mkdtemp()
        self.test_flow_path = Path(self.temp_dir) / "test_flows"
        self.test_flow_path.mkdir()

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
        
        # Clear registry state
        FlowRegistry._entries.clear()
        FlowRegistry._search_paths.clear()

    def create_test_flow(self, name, filename=None, **metadata_kwargs):
        """Create a test flow file."""
        if filename is None:
            filename = f"{name.lower().replace(' ', '_')}.yaml"
        
        flow_config = {
            "metadata": {
                "name": name,
                "description": f"Test flow: {name}",
                "version": "1.0.0",
                "author": "Test Author",
                "tags": ["test"],
                **metadata_kwargs
            },
            "blocks": [
                {
                    "block_type": "LLMChatBlock",
                    "block_config": {
                        "block_name": "test_block",
                        "input_cols": "input",
                        "output_cols": "output",
                        "model": "test/model"
                    }
                }
            ]
        }
        
        flow_path = self.test_flow_path / filename
        with open(flow_path, 'w') as f:
            yaml.dump(flow_config, f)
        
        return str(flow_path)

    def test_register_search_path(self):
        """Test registering search paths."""
        path = "/test/path"
        FlowRegistry.register_search_path(path)
        assert path in FlowRegistry._search_paths
        
        # Should not add duplicates
        FlowRegistry.register_search_path(path)
        assert FlowRegistry._search_paths.count(path) == 1

    def test_discover_flows(self):
        """Test flow discovery."""
        # Create test flows
        self.create_test_flow("Test Flow 1")
        self.create_test_flow("Test Flow 2", author="Another Author")
        
        # Create non-flow file (should be ignored)
        (self.test_flow_path / "not_a_flow.yaml").write_text("not a flow")
        
        # Register search path and discover
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)
        
        # Should have found the flows
        flows = FlowRegistry.list_flows()
        assert len(flows) == 2
        assert "Test Flow 1" in flows
        assert "Test Flow 2" in flows

    def test_discover_flows_recursive(self):
        """Test recursive flow discovery."""
        # Create nested directory structure
        nested_dir = self.test_flow_path / "nested"
        nested_dir.mkdir()
        
        self.create_test_flow("Main Flow")
        
        # Create flow in nested directory
        nested_config = {
            "metadata": {
                "name": "Nested Flow",
                "description": "A nested flow",
                "version": "1.0.0"
            },
            "blocks": []
        }
        
        with open(nested_dir / "nested_flow.yaml", 'w') as f:
            yaml.dump(nested_config, f)
        
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)
        
        flows = FlowRegistry.list_flows()
        assert len(flows) == 2
        assert "Main Flow" in flows
        assert "Nested Flow" in flows

    def test_discover_flows_invalid_yaml(self):
        """Test discovery with invalid YAML files."""
        # Create invalid YAML file
        (self.test_flow_path / "invalid.yaml").write_text("invalid: yaml: content:")
        
        # Create valid flow
        self.create_test_flow("Valid Flow")
        
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)
        
        # Should only find the valid flow
        flows = FlowRegistry.list_flows()
        assert len(flows) == 1
        assert "Valid Flow" in flows

    def test_discover_flows_missing_metadata(self):
        """Test discovery with files missing metadata."""
        # Create file without metadata
        config_without_metadata = {"blocks": []}
        with open(self.test_flow_path / "no_metadata.yaml", 'w') as f:
            yaml.dump(config_without_metadata, f)
        
        # Create valid flow
        self.create_test_flow("Valid Flow")
        
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)
        
        # Should only find the valid flow
        flows = FlowRegistry.list_flows()
        assert len(flows) == 1
        assert "Valid Flow" in flows

    def test_discover_flows_nonexistent_path(self):
        """Test discovery with non-existent path."""
        # Register non-existent path
        FlowRegistry.register_search_path("/nonexistent/path")
        
        # Should not crash
        FlowRegistry._discover_flows(force_refresh=True)
        flows = FlowRegistry.list_flows()
        assert len(flows) == 0

    def test_get_flow_path(self):
        """Test getting flow path."""
        # Create test flow
        flow_path = self.create_test_flow("Test Flow")
        
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)
        
        # Should return the correct path
        retrieved_path = FlowRegistry.get_flow_path("Test Flow")
        assert retrieved_path == flow_path
        
        # Non-existent flow should return None
        assert FlowRegistry.get_flow_path("Nonexistent Flow") is None

    def test_get_flow_metadata(self):
        """Test getting flow metadata."""
        # Create test flow
        self.create_test_flow("Test Flow", author="Test Author", tags=["test", "example"])
        
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)
        
        # Should return metadata
        metadata = FlowRegistry.get_flow_metadata("Test Flow")
        assert metadata is not None
        assert metadata.name == "Test Flow"
        assert metadata.author == "Test Author"
        assert "test" in metadata.tags
        assert "example" in metadata.tags
        
        # Non-existent flow should return None
        assert FlowRegistry.get_flow_metadata("Nonexistent Flow") is None

    def test_list_flows(self):
        """Test listing flows."""
        # Initially empty
        assert FlowRegistry.list_flows() == []
        
        # Create test flows
        self.create_test_flow("Flow A")
        self.create_test_flow("Flow B")
        
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)
        
        flows = FlowRegistry.list_flows()
        assert len(flows) == 2
        assert "Flow A" in flows
        assert "Flow B" in flows

    def test_search_flows_by_tag(self):
        """Test searching flows by tag."""
        # Create flows with different tags
        self.create_test_flow("QA Flow", tags=["qa", "question-answering"])
        self.create_test_flow("Summary Flow", tags=["summarization", "nlp"])
        self.create_test_flow("Mixed Flow", tags=["qa", "summarization"])
        
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)
        
        # Search by tag
        qa_flows = FlowRegistry.search_flows(tag="qa")
        assert len(qa_flows) == 2
        assert "QA Flow" in qa_flows
        assert "Mixed Flow" in qa_flows
        
        nlp_flows = FlowRegistry.search_flows(tag="nlp")
        assert len(nlp_flows) == 1
        assert "Summary Flow" in nlp_flows
        
        # Non-existent tag
        assert FlowRegistry.search_flows(tag="nonexistent") == []

    def test_search_flows_by_author(self):
        """Test searching flows by author."""
        # Create flows with different authors
        self.create_test_flow("Flow 1", author="Alice Smith")
        self.create_test_flow("Flow 2", author="Bob Johnson")
        self.create_test_flow("Flow 3", author="Alice Johnson")
        
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)
        
        # Search by author (case-insensitive partial match)
        alice_flows = FlowRegistry.search_flows(author="alice")
        assert len(alice_flows) == 2
        assert "Flow 1" in alice_flows
        assert "Flow 3" in alice_flows
        
        johnson_flows = FlowRegistry.search_flows(author="Johnson")
        assert len(johnson_flows) == 2
        assert "Flow 2" in johnson_flows
        assert "Flow 3" in johnson_flows
        
        # Non-existent author
        assert FlowRegistry.search_flows(author="nonexistent") == []

    def test_search_flows_combined(self):
        """Test searching flows with combined criteria."""
        # Create flows
        self.create_test_flow("Flow 1", author="Alice", tags=["qa"])
        self.create_test_flow("Flow 2", author="Bob", tags=["qa"])
        self.create_test_flow("Flow 3", author="Alice", tags=["summarization"])
        
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)
        
        # Search with both criteria
        results = FlowRegistry.search_flows(tag="qa", author="Alice")
        assert len(results) == 1
        assert "Flow 1" in results

    def test_get_flows_by_category(self):
        """Test getting flows organized by category."""
        # Create flows with different primary tags
        self.create_test_flow("QA Flow", tags=["question-answering", "nlp"])
        self.create_test_flow("Summary Flow", tags=["summarization", "text-processing"])
        self.create_test_flow("Another QA Flow", tags=["question-answering"])
        self.create_test_flow("Uncategorized Flow", tags=[])
        
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)
        
        categories = FlowRegistry.get_flows_by_category()
        
        # Should have categories based on primary tags
        assert "question-answering" in categories
        assert "summarization" in categories
        assert "uncategorized" in categories
        
        # Check flows in each category
        assert len(categories["question-answering"]) == 2
        assert "QA Flow" in categories["question-answering"]
        assert "Another QA Flow" in categories["question-answering"]
        
        assert len(categories["summarization"]) == 1
        assert "Summary Flow" in categories["summarization"]
        
        assert len(categories["uncategorized"]) == 1
        assert "Uncategorized Flow" in categories["uncategorized"]

    def test_caching_behavior(self):
        """Test that discovery results are cached."""
        # Create test flow
        self.create_test_flow("Test Flow")
        
        FlowRegistry.register_search_path(str(self.test_flow_path))
        
        # First discovery
        FlowRegistry.discover_flows()
        flows1 = FlowRegistry.list_flows()
        
        # Add another flow
        self.create_test_flow("Another Flow")
        
        # Second discovery without force_refresh - should use cache
        FlowRegistry.discover_flows()
        flows2 = FlowRegistry.list_flows()
        
        # Should be the same (cached)
        assert flows1 == flows2
        assert len(flows2) == 1
        
        # Force refresh should pick up new flow
        FlowRegistry._discover_flows(force_refresh=True)
        flows3 = FlowRegistry.list_flows()
        
        assert len(flows3) == 2
        assert "Test Flow" in flows3
        assert "Another Flow" in flows3

    def test_multiple_search_paths(self):
        """Test discovery with multiple search paths."""
        # Create two directories with flows
        dir1 = self.test_flow_path / "dir1"
        dir2 = self.test_flow_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        
        # Create flows in each directory
        flow1_config = {
            "metadata": {"name": "Flow 1", "description": "First flow"},
            "blocks": []
        }
        flow2_config = {
            "metadata": {"name": "Flow 2", "description": "Second flow"},
            "blocks": []
        }
        
        with open(dir1 / "flow1.yaml", 'w') as f:
            yaml.dump(flow1_config, f)
        
        with open(dir2 / "flow2.yaml", 'w') as f:
            yaml.dump(flow2_config, f)
        
        # Register both paths
        FlowRegistry.register_search_path(str(dir1))
        FlowRegistry.register_search_path(str(dir2))
        
        FlowRegistry._discover_flows(force_refresh=True)
        
        # Should find flows from both directories
        flows = FlowRegistry.list_flows()
        assert len(flows) == 2
        assert "Flow 1" in flows
        assert "Flow 2" in flows