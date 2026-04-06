# SPDX-License-Identifier: Apache-2.0
"""Tests for agent_config.py and serialization.py agent-config paths.

Covers the uncovered branches identified in issue #646:
- The extra=="allow" branch in set_agent_config()
- reset_agent_config() edge cases
- from_yaml sets flow._agent_config_set based on agent block presence
"""

from unittest.mock import Mock, patch
import logging

import pytest
import yaml

from sdg_hub import Flow, FlowMetadata
from sdg_hub.core.flow.metadata import RecommendedModels
from tests.flow.conftest import MockBlock


@pytest.fixture()
def test_metadata():
    """Create sample FlowMetadata for agent config tests."""
    return FlowMetadata(
        name="Test Flow",
        description="A test flow",
        version="1.0.0",
        author="Test Author",
        recommended_models=RecommendedModels(
            default="test-model", compatible=["alt-model"], experimental=[]
        ),
        tags=["test"],
    )


def _build_flow_yaml(name, description, block_type, block_name):
    """Build a flow YAML config dict with the given block definition."""
    return {
        "metadata": {
            "name": name,
            "description": description,
            "version": "1.0.0",
            "recommended_models": {
                "default": "test-model",
                "compatible": [],
                "experimental": [],
            },
        },
        "blocks": [
            {
                "block_type": block_type,
                "block_config": {
                    "block_name": block_name,
                    "input_cols": "input",
                    "output_cols": "output",
                },
            }
        ],
    }


def _create_mock_block(name="test_block"):
    """Create a MockBlock with default input/output columns."""
    return MockBlock(block_name=name, input_cols=["input"], output_cols=["output"])


def _create_mock_agent_block(name="agent_block"):
    """Create a MockBlock configured as an agent block with block_type='agent'.

    Uses setattr for agent attributes rather than Pydantic field declarations.
    This mirrors how test_base.py's create_mock_agent_block works and is
    sufficient for the hasattr-based checks in set_agent_config(). If the
    production code switches to model_fields introspection, these helpers
    would need to be updated to use declared Pydantic fields instead.
    """
    block = MockBlock(block_name=name, input_cols=["input"], output_cols=["output"])
    block.block_type = "agent"
    block.agent_framework = "langflow"
    block.agent_url = "http://localhost:7860"
    block.agent_api_key = None
    return block


class TestAgentConfigCoverage:
    """Tests targeting uncovered branches in agent_config.py."""

    # --- Gap 1: extra == "allow" branch in set_agent_config ---

    def test_set_agent_config_extra_allow_branch(self, test_metadata):
        """Passing an unknown kwarg triggers the elif extra=='allow' branch."""
        agent_block = _create_mock_agent_block("agent1")
        flow = Flow(blocks=[agent_block], metadata=test_metadata)

        assert agent_block.model_config.get("extra") == "allow", (
            "Precondition: block must allow extra fields for this branch"
        )
        assert not hasattr(agent_block, "custom_timeout")

        flow.set_agent_config(
            agent_url="http://new:7860",
            custom_timeout=60,
        )

        # agent_url hit the hasattr branch (already exists)
        assert flow.blocks[0].agent_url == "http://new:7860"
        # custom_timeout hit the elif extra=="allow" branch
        assert flow.blocks[0].custom_timeout == 60

    def test_set_agent_config_extra_allow_sensitive_redaction(
        self, test_metadata, caplog
    ):
        """Sensitive params via the extra=='allow' branch are redacted in logs."""
        agent_block = _create_mock_agent_block("agent1")
        flow = Flow(blocks=[agent_block], metadata=test_metadata)

        assert agent_block.model_config.get("extra") == "allow", (
            "Precondition: block must allow extra fields for this branch"
        )
        assert not hasattr(agent_block, "secret")

        with caplog.at_level(logging.DEBUG, logger="sdg_hub.core.flow.agent_config"):
            flow.set_agent_config(
                agent_url="http://new:7860",
                secret="super-secret-value",
            )

        assert flow.blocks[0].secret == "super-secret-value"

        # The secret value must not appear in any log message
        all_log_text = " ".join(record.message for record in caplog.records)
        assert "super-secret-value" not in all_log_text
        # Verify redaction actually happened (not just silent skip)
        assert "secret set (redacted)" in all_log_text

    # --- Gap 2: reset_agent_config() edge cases ---

    def test_reset_agent_config_no_agent_blocks(self, test_metadata):
        """Reset on a flow with no agent blocks is a silent no-op."""
        regular_block = _create_mock_block("regular")
        flow = Flow(blocks=[regular_block], metadata=test_metadata)

        assert flow.is_agent_config_set()

        flow.reset_agent_config()

        # Should still report as set (no agent blocks = nothing to reset)
        assert flow.is_agent_config_set()

    def test_reset_agent_config_never_configured(self, test_metadata):
        """Reset on a flow with agent blocks that was never configured."""
        agent_block = _create_mock_agent_block("agent1")
        flow = Flow(blocks=[agent_block], metadata=test_metadata)

        assert not flow.is_agent_config_set()

        flow.reset_agent_config()

        # Should remain not set
        assert not flow.is_agent_config_set()

    # --- Gap 3: from_yaml sets flow._agent_config_set based on agent blocks ---

    def test_from_yaml_with_agent_blocks_sets_flag_false(self, tmp_path):
        """Loading a YAML with agent blocks sets flow._agent_config_set=False."""
        flow_config = _build_flow_yaml(
            "Agent Flow", "Flow with agent block", "AgentBlock", "my_agent"
        )

        yaml_path = tmp_path / "agent_flow.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(flow_config, f)

        with patch("sdg_hub.core.flow.serialization.BlockRegistry._get") as mock_get:
            mock_block_class = Mock()
            mock_instance = _create_mock_agent_block("my_agent")
            mock_block_class.return_value = mock_instance
            mock_get.return_value = mock_block_class

            flow = Flow.from_yaml(str(yaml_path))

            mock_get.assert_called_once_with("AgentBlock")
            mock_block_class.assert_called_once()
            kwargs = mock_block_class.call_args.kwargs
            assert kwargs["block_name"] == "my_agent"
            assert kwargs["input_cols"] == "input"
            assert kwargs["output_cols"] == "output"

        assert flow._agent_config_set is False
        assert not flow.is_agent_config_set()

    def test_from_yaml_without_agent_blocks_sets_flag_true(self, tmp_path):
        """Loading a YAML with only regular blocks sets flow._agent_config_set=True."""
        flow_config = _build_flow_yaml(
            "Regular Flow", "Flow without agent blocks", "MockBlock", "regular_block"
        )

        yaml_path = tmp_path / "regular_flow.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(flow_config, f)

        with patch("sdg_hub.core.flow.serialization.BlockRegistry._get") as mock_get:
            mock_block_class = Mock()
            mock_instance = _create_mock_block("regular_block")
            mock_block_class.return_value = mock_instance
            mock_get.return_value = mock_block_class

            flow = Flow.from_yaml(str(yaml_path))

            mock_get.assert_called_once_with("MockBlock")
            mock_block_class.assert_called_once()
            kwargs = mock_block_class.call_args.kwargs
            assert kwargs["block_name"] == "regular_block"
            assert kwargs["input_cols"] == "input"
            assert kwargs["output_cols"] == "output"

        assert flow._agent_config_set is True
        assert flow.is_agent_config_set()
