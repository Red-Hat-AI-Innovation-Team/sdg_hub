# SPDX-License-Identifier: Apache-2.0
"""Tests for flow migration functionality."""

# Standard
from pathlib import Path
from unittest.mock import MagicMock
import tempfile

# First Party
from sdg_hub.core.flow.migration import FlowMigration

# Third Party
import yaml


class TestFlowMigration:
    """Test cases for FlowMigration utility class."""

    def test_is_old_format_detects_list(self):
        """Test that old format (list) is correctly detected."""
        old_config = [
            {"block_type": "LLMBlock", "block_config": {"block_name": "test"}}
        ]
        assert FlowMigration.is_old_format(old_config) is True

    def test_is_old_format_detects_new_format(self):
        """Test that new format (dict with metadata/blocks) is correctly detected."""
        new_config = {
            "metadata": {"name": "test"},
            "blocks": [{"block_type": "LLMBlock"}],
        }
        assert FlowMigration.is_old_format(new_config) is False

    def test_is_old_format_handles_edge_cases(self):
        """Test edge cases for format detection."""
        # Empty list (old format)
        assert FlowMigration.is_old_format([]) is True

        # Empty dict (assume new format)
        assert FlowMigration.is_old_format({}) is False

        # Dict without metadata/blocks but with block-like structure
        dict_with_blocks = {"some_key": {"block_type": "TestBlock"}}
        assert FlowMigration.is_old_format(dict_with_blocks) is True

    def test_migrate_basic_old_format(self):
        """Test basic migration of old format to new format."""
        old_config = [
            {
                "block_type": "LLMBlock",
                "block_config": {
                    "block_name": "test_block",
                    "config_path": "test.yaml",
                    "output_cols": ["output"],
                },
                "gen_kwargs": {"temperature": 0.7, "max_tokens": 100},
            }
        ]

        new_config, runtime_params = FlowMigration.migrate_to_new_format(
            old_config, "/test/flow.yaml"
        )

        # Check structure
        assert "metadata" in new_config
        assert "blocks" in new_config
        assert len(new_config["blocks"]) == 1

        # Check metadata
        metadata = new_config["metadata"]
        assert metadata["name"] == "flow"
        assert metadata["version"] == "1.0.0"
        assert "migrated" in metadata["tags"]

        # Check runtime params extraction
        assert "test_block" in runtime_params
        assert runtime_params["test_block"]["temperature"] == 0.7
        assert runtime_params["test_block"]["max_tokens"] == 100

        # Check gen_kwargs removed from block config
        block = new_config["blocks"][0]
        assert "gen_kwargs" not in block

    def test_migrate_removes_unsupported_fields(self):
        """Test that unsupported fields are removed during migration."""
        old_config = [
            {
                "block_type": "FilterByValueBlock",
                "block_config": {
                    "block_name": "filter_block",
                    "filter_column": "score",
                    "filter_value": 1.0,
                },
                "drop_columns": ["temp_col"],
                "drop_duplicates": ["id"],
                "batch_kwargs": {"num_procs": 4},
            }
        ]

        new_config, runtime_params = FlowMigration.migrate_to_new_format(
            old_config, "/test/flow.yaml"
        )

        block = new_config["blocks"][0]
        assert "drop_columns" not in block
        assert "drop_duplicates" not in block
        assert "batch_kwargs" not in block

    def test_migrate_converts_operator_strings(self):
        """Test that operator strings are converted for FilterByValueBlock."""
        old_config = [
            {
                "block_type": "FilterByValueBlock",
                "block_config": {
                    "block_name": "filter_block",
                    "filter_column": "score",
                    "filter_value": 1.0,
                    "operation": "operator.eq",
                },
            }
        ]

        new_config, runtime_params = FlowMigration.migrate_to_new_format(
            old_config, "/test/flow.yaml"
        )

        block = new_config["blocks"][0]
        assert block["block_config"]["operation"] == "eq"

    def test_migrate_preserves_parser_kwargs(self):
        """Test that parser_kwargs are preserved for LLMBlock."""
        old_config = [
            {
                "block_type": "LLMBlock",
                "block_config": {
                    "block_name": "llm_block",
                    "config_path": "test.yaml",
                    "output_cols": ["output"],
                    "parser_kwargs": {
                        "parser_name": "custom",
                        "parsing_pattern": "test_pattern",
                    },
                },
            }
        ]

        new_config, runtime_params = FlowMigration.migrate_to_new_format(
            old_config, "/test/flow.yaml"
        )

        block = new_config["blocks"][0]
        parser_kwargs = block["block_config"]["parser_kwargs"]
        assert parser_kwargs["parser_name"] == "custom"
        assert parser_kwargs["parsing_pattern"] == "test_pattern"

    def test_migrate_handles_multiple_blocks(self):
        """Test migration with multiple blocks of different types."""
        old_config = [
            {
                "block_type": "LLMBlock",
                "block_config": {"block_name": "llm1"},
                "gen_kwargs": {"temperature": 0.5},
            },
            {
                "block_type": "FilterByValueBlock",
                "block_config": {"block_name": "filter1", "operation": "operator.ge"},
                "drop_columns": ["temp"],
            },
            {"block_type": "DuplicateColumns", "block_config": {"block_name": "dup1"}},
        ]

        new_config, runtime_params = FlowMigration.migrate_to_new_format(
            old_config, "/test/complex_flow.yaml"
        )

        assert len(new_config["blocks"]) == 3
        assert "llm1" in runtime_params
        assert runtime_params["llm1"]["temperature"] == 0.5

        # Check operator conversion
        filter_block = new_config["blocks"][1]
        assert filter_block["block_config"]["operation"] == "ge"

        # Check drop_columns removed
        assert "drop_columns" not in filter_block

    def test_migrate_handles_malformed_blocks(self):
        """Test that malformed blocks are handled gracefully."""
        old_config = [
            {"block_type": "LLMBlock", "block_config": {"block_name": "good_block"}},
            "invalid_block_config",  # This should be handled gracefully
            {
                "block_type": "FilterByValueBlock",
                "block_config": {"block_name": "another_good_block"},
            },
        ]

        new_config, runtime_params = FlowMigration.migrate_to_new_format(
            old_config, "/test/flow.yaml"
        )

        # Should still have 3 blocks (including the malformed one as fallback)
        assert len(new_config["blocks"]) == 3
        # Good blocks should still be processed
        assert new_config["blocks"][0]["block_config"]["block_name"] == "good_block"
        assert (
            new_config["blocks"][2]["block_config"]["block_name"]
            == "another_good_block"
        )

    def test_generate_default_metadata(self):
        """Test default metadata generation."""
        metadata = FlowMigration._generate_default_metadata("test_flow")

        assert metadata["name"] == "test_flow"
        assert metadata["version"] == "1.0.0"
        assert metadata["author"] == "SDG_Hub"
        assert "migrated" in metadata["tags"]
        assert "recommended_models" in metadata
        assert len(metadata["recommended_models"]) > 0
        # Check id generation
        assert metadata["id"] is not None

    def test_migrate_block_config_edge_cases(self):
        """Test edge cases in block config migration."""
        # Test with non-dict input
        result_config, result_params = FlowMigration._migrate_block_config("not_a_dict")
        assert result_config == "not_a_dict"
        assert result_params == {}

        # Test with empty dict
        result_config, result_params = FlowMigration._migrate_block_config({})
        assert result_config == {}
        assert result_params == {}

        # Test with dict missing expected fields
        config = {"some_field": "some_value"}
        result_config, result_params = FlowMigration._migrate_block_config(config)
        assert result_config == config
        assert result_params == {}


class TestFlowMigrationIntegration:
    """Integration tests for flow migration with actual Flow class."""

    def test_flow_from_yaml_with_old_format(self):
        """Test loading an old format YAML through Flow.from_yaml."""
        # First Party
        from sdg_hub import Flow

        old_flow_config = [
            {
                "block_type": "DuplicateColumns",
                "block_config": {
                    "block_name": "test_duplicate",
                    "columns_map": {"input": "output"},
                },
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(old_flow_config, f)
            temp_path = f.name

        try:
            # First load - should migrate and generate id
            flow1 = Flow.from_yaml(temp_path)
            assert flow1.metadata.name == Path(temp_path).stem
            assert "migrated" in flow1.metadata.tags
            assert len(flow1.blocks) == 1
            assert flow1.blocks[0].block_name == "test_duplicate"
            first_id = flow1.metadata.id
            assert first_id  # Should have generated an ID

            # Verify id was saved to YAML during migration
            with open(temp_path, "r") as f:
                migrated_config = yaml.safe_load(f)
                assert "metadata" in migrated_config
                assert "id" in migrated_config["metadata"]
                assert migrated_config["metadata"]["id"] == first_id

            # Load again - should use same id since it's now in new format
            flow2 = Flow.from_yaml(temp_path)
            assert flow2.metadata.id == first_id  # Should use same ID

            # Create another old format flow - should get different ID when migrated
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f2:
                yaml.dump(old_flow_config, f2)
                temp_path2 = f2.name

            try:
                # When migrated, should get different ID
                flow3 = Flow.from_yaml(temp_path2)
                assert flow3.metadata.id != first_id  # Should get different ID

                # Verify the new ID was saved to YAML during migration
                with open(temp_path2, "r") as f:
                    migrated_config2 = yaml.safe_load(f)
                    assert "metadata" in migrated_config2
                    assert "id" in migrated_config2["metadata"]
                    assert migrated_config2["metadata"]["id"] == flow3.metadata.id
            finally:
                Path(temp_path2).unlink()

            # Create two identical old format flows to verify they get different IDs when migrated
            old_config = [
                {
                    "block_type": "DuplicateColumns",
                    "block_config": {
                        "block_name": "test_duplicate2",
                        "columns_map": {"input": "output"},
                    },
                }
            ]

            # Create first temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f2:
                yaml.dump(old_config, f2)
                temp_path2 = f2.name

            # Create second temp file with same content
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f3:
                yaml.dump(old_config, f3)
                temp_path3 = f3.name

            try:
                # Load both files - should get different IDs since they're separate migrations
                flow3 = Flow.from_yaml(temp_path2)
                id1 = flow3.metadata.id

                flow4 = Flow.from_yaml(temp_path3)
                id2 = flow4.metadata.id

                assert id1 != id2  # Should get different IDs for different files

                # Verify both IDs were saved to their respective YAMLs
                with open(temp_path2, "r") as f:
                    config2 = yaml.safe_load(f)
                    assert config2["metadata"]["id"] == id1

                with open(temp_path3, "r") as f:
                    config3 = yaml.safe_load(f)
                    assert config3["metadata"]["id"] == id2

            finally:
                Path(temp_path2).unlink()
                Path(temp_path3).unlink()

        finally:
            Path(temp_path).unlink()

    def test_flow_from_yaml_with_llm_client_injection(self):
        """Test that client is properly injected for LLMBlocks."""
        # First Party
        from sdg_hub import Flow

        # Create a simple config file for LLMBlock
        config_content = """
system: "You are a helpful assistant."
introduction: "Generate text"
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as config_f:
            config_f.write(config_content)
            config_path = config_f.name

        old_flow_config = [
            {
                "block_type": "LLMBlock",
                "block_config": {
                    "block_name": "test_llm",
                    "config_path": config_path,
                    "output_cols": ["output"],
                    "model_id": "test-model",
                },
                "gen_kwargs": {"temperature": 0.7},
            }
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as flow_f:
            yaml.dump(old_flow_config, flow_f)
            flow_path = flow_f.name

        try:
            mock_client = MagicMock()

            # Should successfully load with client injection
            flow = Flow.from_yaml(flow_path, client=mock_client)

            assert len(flow.blocks) == 1
            assert flow.blocks[0].block_name == "test_llm"

            # Check that runtime params were extracted
            assert hasattr(flow, "_migrated_runtime_params")
            assert "test_llm" in flow._migrated_runtime_params
            assert flow._migrated_runtime_params["test_llm"]["temperature"] == 0.7

        finally:
            Path(flow_path).unlink()
            Path(config_path).unlink()

    def test_flow_from_yaml_new_format_ignores_client(self):
        """Test that new format YAMLs ignore the client parameter."""
        # First Party
        from sdg_hub import Flow

        new_flow_config = {
            "metadata": {
                "name": "test_flow",
                "version": "1.0.0",
                "description": "Test flow",
            },
            "blocks": [
                {
                    "block_type": "DuplicateColumnsBlock",
                    "block_config": {
                        "block_name": "test_block",
                        "input_cols": {"input": "output"},
                    },
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(new_flow_config, f)
            temp_path = f.name

        try:
            mock_client = MagicMock()

            # Should load successfully and ignore client
            flow = Flow.from_yaml(temp_path, client=mock_client)

            assert flow.metadata.name == "test_flow"
            assert not hasattr(flow, "_llm_client") or flow._llm_client is None
            assert len(flow.blocks) == 1

        finally:
            Path(temp_path).unlink()
