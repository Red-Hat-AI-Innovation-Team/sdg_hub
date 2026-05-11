# SPDX-License-Identifier: Apache-2.0
"""Flow-specific test fixtures."""

# Standard
from pathlib import Path
from unittest.mock import Mock

# Third Party
import pytest
import yaml

# First Party
from sdg_hub import FlowMetadata
from sdg_hub.core.flow.metadata import ModelCompatibility, ModelOption


@pytest.fixture
def sample_metadata():
    """Create sample flow metadata."""
    return FlowMetadata(
        name="Test Flow",
        description="A test flow for testing",
        version="1.0.0",
        author="Test Author",
        recommended_models=[
            ModelOption(name="test-model", compatibility=ModelCompatibility.REQUIRED),
            ModelOption(
                name="fallback-model", compatibility=ModelCompatibility.COMPATIBLE
            ),
        ],
        tags=["test", "example"],
        estimated_cost="low",
        estimated_duration="1-2 minutes",
    )


@pytest.fixture
def sample_flow_yaml(temp_dir):
    """Create a sample flow YAML file."""

    def _create_flow_yaml(name="Test Flow", **kwargs):
        flow_config = {
            "metadata": {
                "name": name,
                "description": f"Test flow: {name}",
                "version": "1.0.0",
                "author": "Test Author",
                "recommended_models": [
                    {"name": "test-model", "compatibility": "required"}
                ],
                "tags": ["test"],
                **kwargs,
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

        file_path = Path(temp_dir) / f"{name.lower().replace(' ', '_')}.yaml"
        with open(file_path, "w") as f:
            yaml.dump(flow_config, f)

        return str(file_path)

    return _create_flow_yaml


@pytest.fixture
def flow_registry_setup(temp_dir):
    """Set up flow registry for testing."""
    # First Party
    from src.sdg_hub.flow.registry import FlowRegistry

    # Clear existing state
    FlowRegistry._entries.clear()
    FlowRegistry._search_paths.clear()

    # Create test flows directory
    flows_dir = Path(temp_dir) / "test_flows"
    flows_dir.mkdir()

    yield flows_dir

    # Cleanup
    FlowRegistry._entries.clear()
    FlowRegistry._search_paths.clear()


@pytest.fixture
def mock_block_registry():
    """Mock the BlockRegistry for testing."""
    # Standard
    from unittest.mock import patch

    with patch("src.sdg_hub.flow.base.BlockRegistry") as mock_registry:
        # Mock the get method to return a mock block class
        def mock_get(block_type):
            mock_class = Mock()
            mock_instance = Mock()
            mock_instance.block_name = "test_block"
            mock_instance.__class__.__name__ = block_type
            mock_class.return_value = mock_instance
            return mock_class

        mock_registry._get.side_effect = mock_get
        mock_registry.list_blocks.return_value = ["LLMChatBlock", "MockBlock"]

        yield mock_registry
