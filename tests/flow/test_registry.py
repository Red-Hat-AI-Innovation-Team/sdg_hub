# SPDX-License-Identifier: Apache-2.0
"""Tests for flow registry."""

# Standard
from pathlib import Path
import tempfile

# First Party
# Local
from sdg_hub import FlowRegistry

# Third Party
import yaml


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
        # Standard
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
                **metadata_kwargs,
            },
            "blocks": [
                {
                    "block_type": "LLMChatBlock",
                    "block_config": {
                        "block_name": "test_block",
                        "input_cols": "input",
                        "output_cols": "output",
                        "model": "test/model",
                    },
                }
            ],
        }

        flow_path = self.test_flow_path / filename
        with open(flow_path, "w") as f:
            yaml.dump(flow_config, f)

        return str(flow_path)

    def test_id_persistence(self):
        """Test that flow IDs are consistent only when saved to YAML."""
        # Create flow without id
        flow_path = self.create_test_flow("Test Flow")
        FlowRegistry.register_search_path(str(self.test_flow_path))

        # First discovery - should generate and save id
        FlowRegistry._discover_flows(force_refresh=True)
        metadata1 = FlowRegistry.get_flow_metadata("Test Flow")
        first_id = metadata1.id

        # Verify id was saved to YAML
        with open(flow_path, "r") as f:
            flow_config = yaml.safe_load(f)
            assert "id" in flow_config["metadata"]
            assert flow_config["metadata"]["id"] == first_id

        # Clear registry and rediscover - should load same id from YAML
        FlowRegistry._entries.clear()
        FlowRegistry._discover_flows(force_refresh=True)
        metadata2 = FlowRegistry.get_flow_metadata("Test Flow")
        assert metadata2.id == first_id  # Should be same as saved in YAML

        # Create new flow without saving id to YAML
        self.create_test_flow("Test Flow 2")

        # Multiple discoveries without saving to YAML should generate different IDs
        FlowRegistry._discover_flows(force_refresh=True)
        id1 = FlowRegistry.get_flow_metadata("Test Flow").id

        FlowRegistry._entries.clear()
        FlowRegistry._discover_flows(force_refresh=True)
        id2 = FlowRegistry.get_flow_metadata("Test Flow 2").id

        assert id1 != id2  # IDs should be different since they weren't saved

    def test_id_yaml_update(self):
        """Test that id is written to YAML during discovery."""
        # Create flow without id
        flow_path = self.create_test_flow("Test Flow")

        # Verify no id in original YAML
        with open(flow_path, "r") as f:
            original_config = yaml.safe_load(f)
            assert "id" not in original_config["metadata"]

        # Discover flows - should generate and save id
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)

        # Verify id was written to YAML
        with open(flow_path, "r") as f:
            updated_config = yaml.safe_load(f)
            assert "id" in updated_config["metadata"]
            saved_id = updated_config["metadata"]["id"]

        # Clear registry and rediscover - should use saved id
        FlowRegistry._entries.clear()
        FlowRegistry._discover_flows(force_refresh=True)
        metadata = FlowRegistry.get_flow_metadata("Test Flow")
        assert metadata.id == saved_id

    def test_custom_id_preservation(self):
        """Test that custom flow IDs are preserved."""
        # Create flow with custom id
        custom_id = "custom-test-id"
        self.create_test_flow("Test Flow", id=custom_id)

        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)

        # Verify custom id is preserved
        metadata = FlowRegistry.get_flow_metadata("Test Flow")
        assert metadata.id == custom_id

        # Clear registry and rediscover - should still use custom id
        FlowRegistry._entries.clear()
        FlowRegistry._discover_flows(force_refresh=True)
        metadata = FlowRegistry.get_flow_metadata("Test Flow")
        assert metadata.id == custom_id

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
        flow_names = [f["name"] for f in flows]
        assert "Test Flow 1" in flow_names
        assert "Test Flow 2" in flow_names
        # Verify each flow has an ID
        for flow in flows:
            assert "id" in flow
            assert flow["id"]  # ID should not be empty

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
                "version": "1.0.0",
            },
            "blocks": [],
        }

        with open(nested_dir / "nested_flow.yaml", "w") as f:
            yaml.dump(nested_config, f)

        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)

        flows = FlowRegistry.list_flows()
        assert len(flows) == 2
        flow_names = [f["name"] for f in flows]
        assert "Main Flow" in flow_names
        assert "Nested Flow" in flow_names
        # Verify each flow has an ID
        for flow in flows:
            assert "id" in flow
            assert flow["id"]  # ID should not be empty

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
        assert "Valid Flow" in flows[0]["name"]

    def test_discover_flows_missing_metadata(self):
        """Test discovery with files missing metadata."""
        # Create file without metadata
        config_without_metadata = {"blocks": []}
        with open(self.test_flow_path / "no_metadata.yaml", "w") as f:
            yaml.dump(config_without_metadata, f)

        # Create valid flow
        self.create_test_flow("Valid Flow")

        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)

        # Should only find the valid flow
        flows = FlowRegistry.list_flows()
        assert len(flows) == 1
        assert "Valid Flow" in flows[0]["name"]

    def test_discover_flows_nonexistent_path(self):
        """Test discovery with non-existent path."""
        # Register non-existent path
        FlowRegistry.register_search_path("/nonexistent/path")

        # Should not crash
        FlowRegistry._discover_flows(force_refresh=True)
        flows = FlowRegistry.list_flows()
        assert len(flows) == 0

    def test_get_flow_path(self):
        """Test getting flow path by both ID and name."""
        # Create test flow
        flow_path = self.create_test_flow("Test Flow")

        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)

        # Get the flow's ID from metadata
        metadata = FlowRegistry.get_flow_metadata("Test Flow")
        id = metadata.id

        # Should find by id (preferred)
        retrieved_path = FlowRegistry.get_flow_path(id)
        assert retrieved_path == flow_path

        # Should also find by name (backward compatibility)
        retrieved_path_by_name = FlowRegistry.get_flow_path("Test Flow")
        assert retrieved_path_by_name == flow_path

        # Non-existent identifier should return None
        assert FlowRegistry.get_flow_path("Nonexistent Flow") is None
        assert FlowRegistry.get_flow_path("nonexistent-id") is None

        # Create another flow to test uniqueness
        another_flow_path = self.create_test_flow("Another Flow")
        FlowRegistry._discover_flows(force_refresh=True)

        # Each flow should be uniquely identifiable by either name or ID
        another_metadata = FlowRegistry.get_flow_metadata("Another Flow")
        another_id = another_metadata.id

        assert FlowRegistry.get_flow_path(another_id) == another_flow_path
        assert FlowRegistry.get_flow_path("Another Flow") == another_flow_path

        # Original flow should still be accessible
        assert FlowRegistry.get_flow_path(id) == flow_path
        assert FlowRegistry.get_flow_path("Test Flow") == flow_path

    def test_get_flow_metadata(self):
        """Test getting flow metadata."""
        # Create test flow
        self.create_test_flow(
            "Test Flow", author="Test Author", tags=["test", "example"]
        )

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
        flow_names = [f["name"] for f in flows]
        assert "Flow A" in flow_names
        assert "Flow B" in flow_names
        # Verify each flow has an ID
        for flow in flows:
            assert flow["id"]  # ID should not be empty

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
        qa_flow_names = [f["name"] for f in qa_flows]
        assert "QA Flow" in qa_flow_names
        assert "Mixed Flow" in qa_flow_names
        # Verify IDs
        for flow in qa_flows:
            assert flow["id"]

        nlp_flows = FlowRegistry.search_flows(tag="nlp")
        assert len(nlp_flows) == 1
        assert nlp_flows[0]["name"] == "Summary Flow"
        assert nlp_flows[0]["id"]  # Should have an ID

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
        alice_flow_names = [f["name"] for f in alice_flows]
        assert "Flow 1" in alice_flow_names
        assert "Flow 3" in alice_flow_names
        # Verify IDs
        for flow in alice_flows:
            assert flow["id"]

        johnson_flows = FlowRegistry.search_flows(author="Johnson")
        assert len(johnson_flows) == 2
        johnson_flow_names = [f["name"] for f in johnson_flows]
        assert "Flow 2" in johnson_flow_names
        assert "Flow 3" in johnson_flow_names
        # Verify IDs
        for flow in johnson_flows:
            assert flow["id"]

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
        assert results[0]["name"] == "Flow 1"
        assert results[0]["id"]  # Should have an ID

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
        qa_flow_names = [f["name"] for f in categories["question-answering"]]
        assert "QA Flow" in qa_flow_names
        assert "Another QA Flow" in qa_flow_names
        # Verify IDs
        for flow in categories["question-answering"]:
            assert flow["id"]

        assert len(categories["summarization"]) == 1
        assert categories["summarization"][0]["name"] == "Summary Flow"
        assert categories["summarization"][0]["id"]  # Should have an ID

        assert len(categories["uncategorized"]) == 1
        assert categories["uncategorized"][0]["name"] == "Uncategorized Flow"
        assert categories["uncategorized"][0]["id"]  # Should have an ID

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
        assert flows2[0]["name"] == "Test Flow"
        assert flows2[0]["id"]  # Should have an ID

        # Force refresh should pick up new flow
        FlowRegistry._discover_flows(force_refresh=True)
        flows3 = FlowRegistry.list_flows()

        assert len(flows3) == 2
        flow_names = [f["name"] for f in flows3]
        assert "Test Flow" in flow_names
        assert "Another Flow" in flow_names
        # Verify IDs
        for flow in flows3:
            assert flow["id"]

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
            "blocks": [],
        }
        flow2_config = {
            "metadata": {"name": "Flow 2", "description": "Second flow"},
            "blocks": [],
        }

        with open(dir1 / "flow1.yaml", "w") as f:
            yaml.dump(flow1_config, f)

        with open(dir2 / "flow2.yaml", "w") as f:
            yaml.dump(flow2_config, f)

        # Register both paths
        FlowRegistry.register_search_path(str(dir1))
        FlowRegistry.register_search_path(str(dir2))

        FlowRegistry._discover_flows(force_refresh=True)

        # Should find flows from both directories
        flows = FlowRegistry.list_flows()
        assert len(flows) == 2
        flow_names = [f["name"] for f in flows]
        assert "Flow 1" in flow_names
        assert "Flow 2" in flow_names
        # Verify IDs
        for flow in flows:
            assert flow["id"]  # Should have an ID

    def test_get_flow_path_safe_success(self):
        """Test get_flow_path_safe with existing flow."""
        # Create test flow
        flow_path = self.create_test_flow("Test Flow")
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)

        # Get flow metadata to find its ID
        metadata = FlowRegistry.get_flow_metadata("Test Flow")
        flow_id = metadata.id

        # Should return path successfully
        result_path = FlowRegistry.get_flow_path_safe(flow_id)
        assert result_path == flow_path

        # Should also work with flow name (backward compatibility)
        result_path_by_name = FlowRegistry.get_flow_path_safe("Test Flow")
        assert result_path_by_name == flow_path

    def test_get_flow_path_safe_not_found_with_available_flows(self):
        """Test get_flow_path_safe with non-existent flow when flows are available."""
        # Third Party
        import pytest

        # Create some flows in registry
        self.create_test_flow("Flow 1")
        self.create_test_flow("Flow 2")
        FlowRegistry.register_search_path(str(self.test_flow_path))
        FlowRegistry._discover_flows(force_refresh=True)

        # Try to get non-existent flow
        with pytest.raises(ValueError) as exc_info:
            FlowRegistry.get_flow_path_safe("nonexistent-flow")

        # Verify error message format
        error_msg = str(exc_info.value)
        assert "Flow 'nonexistent-flow' not found" in error_msg
        assert "Available flows:" in error_msg
        assert "ID:" in error_msg
        assert "Name:" in error_msg
        # Should contain the actual flow names
        assert "Flow 1" in error_msg
        assert "Flow 2" in error_msg

    def test_get_flow_path_safe_not_found_empty_registry(self):
        """Test get_flow_path_safe with empty registry."""
        # Third Party
        # Standard
        from unittest.mock import patch

        import pytest

        # Use patch to prevent auto-discovery during initialization
        with patch("sdg_hub.core.flow.registry.Path") as mock_path:
            # Make the built-in flows directory appear non-existent
            mock_path.return_value.parent.return_value.__truediv__.return_value.exists.return_value = False

            # Clear registry state
            FlowRegistry._entries.clear()
            FlowRegistry._search_paths.clear()
            FlowRegistry._initialized = False

            with pytest.raises(ValueError) as exc_info:
                FlowRegistry.get_flow_path_safe("any-flow")

            error_msg = str(exc_info.value)
            assert "Flow 'any-flow' not found" in error_msg
            assert "No flows are currently registered" in error_msg
            assert "FlowRegistry.discover_flows()" in error_msg
