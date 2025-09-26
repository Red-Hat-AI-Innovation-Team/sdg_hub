# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the flow system."""

# Standard
from pathlib import Path
from unittest.mock import Mock, patch

# Third Party
from datasets import Dataset

# First Party
from sdg_hub import Flow, FlowMetadata, FlowRegistry
import pytest
import yaml


class TestFlowIntegration:
    """Integration tests for the complete flow system."""

    def test_end_to_end_flow_creation_and_execution(self, temp_dir, mock_block):
        """Test complete end-to-end flow creation and execution."""
        # Create a flow YAML file
        flow_config = {
            "metadata": {
                "name": "Integration Test Flow",
                "description": "End-to-end test flow",
                "version": "1.0.0",
                "author": "Test Suite",
                "recommended_models": {
                    "default": "test-model",
                    "compatible": [],
                    "experimental": [],
                },
                "tags": ["integration", "test"],
                "estimated_cost": "low",
                "estimated_duration": "1 minute",
                "dataset_requirements": {
                    "required_columns": ["input"],
                    "min_samples": 1,
                },
            },
            "parameters": {
                "temperature": {
                    "default": 0.7,
                    "description": "Model temperature",
                    "type_hint": "float",
                    "required": False,
                }
            },
            "blocks": [
                {
                    "block_type": "LLMChatBlock",
                    "block_config": {
                        "block_name": "chat_block",
                        "input_cols": "input",
                        "output_cols": "response",
                    },
                },
                {
                    "block_type": "ProcessorBlock",
                    "block_config": {
                        "block_name": "processor",
                        "input_cols": "response",
                        "output_cols": "processed_response",
                    },
                },
            ],
        }

        yaml_path = Path(temp_dir) / "integration_test.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(flow_config, f)

        # Mock the block registry
        with patch("sdg_hub.core.flow.base.BlockRegistry") as mock_registry:

            def mock_get(block_type):
                mock_class = Mock()
                if block_type == "LLMChatBlock":
                    mock_instance = mock_block("chat_block", ["input"], ["response"])
                else:
                    mock_instance = mock_block(
                        "processor", ["response"], ["processed_response"]
                    )
                mock_class.return_value = mock_instance
                return mock_class

            mock_registry._get.side_effect = mock_get

            # Load the flow
            flow = Flow.from_yaml(str(yaml_path))

            # Configure model since we have LLM blocks
            flow.set_model_config(
                model="test-model", api_base="http://localhost:8000/v1", api_key="EMPTY"
            )

            # Verify flow structure
            assert flow.metadata.name == "Integration Test Flow"
            assert len(flow.blocks) == 2
            assert flow.blocks[0].block_name == "chat_block"
            assert flow.blocks[1].block_name == "processor"
            assert len(flow.parameters) == 1
            assert flow.parameters["temperature"].default == 0.7

            # Create test dataset
            dataset = Dataset.from_dict(
                {"input": ["Hello world", "How are you?", "Test input"]}
            )

            # Validate dataset
            errors = flow.validate_dataset(dataset)
            assert errors == []

            # Perform dry run
            dry_run_result = flow.dry_run(dataset, sample_size=2)

            assert dry_run_result["execution_successful"] is True
            assert dry_run_result["sample_size"] == 2
            assert dry_run_result["original_dataset_size"] == 3
            assert len(dry_run_result["blocks_executed"]) == 2
            assert dry_run_result["blocks_executed"][0]["block_name"] == "chat_block"
            assert dry_run_result["blocks_executed"][1]["block_name"] == "processor"

            # Execute full flow
            runtime_params = {"chat_block": {"temperature": 0.5}}

            result = flow.generate(dataset, runtime_params=runtime_params)

            # Verify results
            assert len(result) == 3
            assert "processed_response" in result.column_names
            assert all(result["processed_response"])

    def test_flow_registry_discovery_and_loading(self, temp_dir):
        """Test flow registry discovery and loading."""
        # Create multiple test flows
        flows_dir = Path(temp_dir) / "test_flows"
        flows_dir.mkdir()

        # Create flow configurations
        flow_configs = [
            {
                "name": "QA Flow",
                "tags": ["qa", "question-answering"],
                "author": "QA Team",
                "config": {
                    "metadata": {
                        "name": "QA Flow",
                        "description": "Question answering flow",
                        "tags": ["qa", "question-answering"],
                        "author": "QA Team",
                        "recommended_models": {
                            "default": "gpt-4",
                            "compatible": [],
                            "experimental": [],
                        },
                    },
                    "blocks": [
                        {
                            "block_type": "LLMChatBlock",
                            "block_config": {
                                "block_name": "qa_block",
                                "input_cols": "question",
                                "output_cols": "answer",
                            },
                        }
                    ],
                },
            },
            {
                "name": "Summary Flow",
                "tags": ["summarization", "nlp"],
                "author": "NLP Team",
                "config": {
                    "metadata": {
                        "name": "Summary Flow",
                        "description": "Text summarization flow",
                        "tags": ["summarization", "nlp"],
                        "author": "NLP Team",
                        "recommended_models": {
                            "default": "claude-3",
                            "compatible": [],
                            "experimental": [],
                        },
                    },
                    "blocks": [
                        {
                            "block_type": "SummaryBlock",
                            "block_config": {
                                "block_name": "summary_block",
                                "input_cols": "text",
                                "output_cols": "summary",
                            },
                        }
                    ],
                },
            },
        ]

        # Write flow files
        for flow_info in flow_configs:
            flow_path = (
                flows_dir / f"{flow_info['name'].lower().replace(' ', '_')}.yaml"
            )
            with open(flow_path, "w") as f:
                yaml.dump(flow_info["config"], f)

        # Clear registry and add search path
        FlowRegistry._entries.clear()
        FlowRegistry._search_paths.clear()
        FlowRegistry._initialized = False  # Prevent auto-discovery of built-in flows
        FlowRegistry.register_search_path(str(flows_dir))

        # Manually discover flows in test directory only (bypass auto-discovery)
        FlowRegistry._discover_flows_in_directory(str(flows_dir))

        # Test listing flows
        flows = FlowRegistry.list_flows()
        flow_names = [flow["name"] for flow in flows]
        assert len(flows) == 2
        assert "QA Flow" in flow_names
        assert "Summary Flow" in flow_names

        # Validate flow id presence and validity
        assert all("id" in flow for flow in flows), "All flows should have id"
        assert all(flow["id"] for flow in flows), "All flow ids should be non-empty"
        assert all(
            isinstance(flow["id"], str) for flow in flows
        ), "All flow ids should be strings"

        # Test searching by tag
        qa_flows = FlowRegistry.search_flows(tag="qa")
        qa_flow_names = [flow["name"] for flow in qa_flows]
        assert len(qa_flows) == 1
        assert "QA Flow" in qa_flow_names

        # Validate flow id presence and validity for QA flows
        assert all("id" in flow for flow in qa_flows), "QA flows should have id"
        assert all(flow["id"] for flow in qa_flows), "QA flow ids should be non-empty"

        nlp_flows = FlowRegistry.search_flows(tag="nlp")
        nlp_flow_names = [flow["name"] for flow in nlp_flows]
        assert len(nlp_flows) == 1
        assert "Summary Flow" in nlp_flow_names

        # Test searching by author
        qa_team_flows = FlowRegistry.search_flows(author="QA Team")
        qa_team_flow_names = [flow["name"] for flow in qa_team_flows]
        assert len(qa_team_flows) == 1
        assert "QA Flow" in qa_team_flow_names

        # Test getting flow metadata
        qa_metadata = FlowRegistry.get_flow_metadata("QA Flow")
        assert qa_metadata is not None
        assert qa_metadata.name == "QA Flow"
        assert qa_metadata.author == "QA Team"
        assert "qa" in qa_metadata.tags

        # Test getting flow path
        qa_path = FlowRegistry.get_flow_path("QA Flow")
        assert qa_path is not None
        assert Path(qa_path).exists()

        # Test categories
        categories = FlowRegistry.get_flows_by_category()
        assert "qa" in categories
        assert "summarization" in categories
        assert "QA Flow" in categories["qa"][0]["name"]
        assert "Summary Flow" in categories["summarization"][0]["name"]

    def test_model_compatibility_system(self):
        """Test the model compatibility system with new simplified format."""
        # First Party
        from sdg_hub.core.flow.metadata import RecommendedModels

        # Create metadata with the new simplified format
        recommended_models = RecommendedModels(
            default="default-model",
            compatible=["compatible-model1", "compatible-model2"],
            experimental=["experimental-model"],
        )

        metadata = FlowMetadata(
            name="Compatibility Test Flow", recommended_models=recommended_models
        )

        # Test model selection with different availability scenarios

        # Scenario 1: Default model available
        available_models = ["default-model", "compatible-model1", "experimental-model"]
        best_model = metadata.get_best_model(available_models)
        assert best_model == "default-model"  # Should pick default first

        # Scenario 2: Default not available, compatible available
        without_default = [
            "compatible-model1",
            "compatible-model2",
            "experimental-model",
        ]
        best_model = metadata.get_best_model(without_default)
        assert best_model == "compatible-model1"  # Should pick first compatible

        # Scenario 3: Only experimental model available
        only_experimental = ["experimental-model"]
        best_model = metadata.get_best_model(only_experimental)
        assert (
            best_model == "experimental-model"
        )  # Should pick experimental if available

        # Scenario 4: No compatible models available
        no_compatible = ["some-other-model"]
        best_model = metadata.get_best_model(no_compatible)
        assert best_model is None  # Should return None

        # Scenario 5: No availability constraint (should return default)
        best_model = metadata.get_best_model()
        assert best_model == "default-model"  # Should return default

    def test_complex_flow_with_parameters(self, temp_dir, mock_block):
        """Test a complex flow with parameters and validation."""
        # Create a complex flow configuration
        flow_config = {
            "metadata": {
                "name": "Complex Parameter Flow",
                "description": "Flow with complex parameter handling",
                "version": "2.0.0",
                "dataset_requirements": {
                    "required_columns": ["input", "context"],
                    "min_samples": 2,
                    "max_samples": 1000,
                },
            },
            "parameters": {
                "global_temperature": {
                    "default": 0.7,
                    "description": "Global temperature setting",
                    "type_hint": "float",
                    "required": True,
                    "constraints": {"min": 0.0, "max": 1.0},
                },
                "max_tokens": {
                    "default": 512,
                    "description": "Maximum tokens per response",
                    "type_hint": "int",
                    "required": False,
                },
            },
            "blocks": [
                {
                    "block_type": "ContextProcessor",
                    "block_config": {
                        "block_name": "context_processor",
                        "input_cols": ["input", "context"],
                        "output_cols": "processed_context",
                    },
                },
                {
                    "block_type": "LLMChatBlock",
                    "block_config": {
                        "block_name": "llm_generator",
                        "input_cols": "processed_context",
                        "output_cols": "response",
                    },
                },
            ],
        }

        yaml_path = Path(temp_dir) / "complex_flow.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(flow_config, f)

        # Mock the block registry
        with patch("sdg_hub.core.flow.base.BlockRegistry") as mock_registry:

            def mock_get(block_type):
                mock_class = Mock()
                if block_type == "ContextProcessor":
                    mock_instance = mock_block(
                        "context_processor", ["input", "context"], ["processed_context"]
                    )
                else:
                    mock_instance = mock_block(
                        "llm_generator", ["processed_context"], ["response"]
                    )
                mock_class.return_value = mock_instance
                return mock_class

            mock_registry._get.side_effect = mock_get

            # Load the flow
            flow = Flow.from_yaml(str(yaml_path))

            # Verify parameter handling
            assert len(flow.parameters) == 2
            assert flow.parameters["global_temperature"].required is True
            assert flow.parameters["max_tokens"].required is False

            # Test dataset validation
            # Valid dataset
            valid_dataset = Dataset.from_dict(
                {
                    "input": ["input1", "input2", "input3"],
                    "context": ["context1", "context2", "context3"],
                }
            )
            errors = flow.validate_dataset(valid_dataset)
            assert errors == []

            # Invalid dataset - missing required column
            invalid_dataset = Dataset.from_dict(
                {
                    "input": ["input1", "input2"]
                    # Missing context column
                }
            )
            errors = flow.validate_dataset(invalid_dataset)
            assert len(errors) > 0
            assert any("Missing required columns" in error for error in errors)

            # Invalid dataset - too few samples
            small_dataset = Dataset.from_dict(
                {"input": ["input1"], "context": ["context1"]}
            )
            errors = flow.validate_dataset(small_dataset)
            assert len(errors) > 0
            assert any("minimum required: 2" in error for error in errors)

            # Test execution with runtime parameters
            runtime_params = {
                "context_processor": {"processing_mode": "detailed"},
                "llm_generator": {"temperature": 0.3, "max_tokens": 256},
            }

            # Dry run
            dry_run_result = flow.dry_run(
                valid_dataset, sample_size=2, runtime_params=runtime_params
            )

            assert dry_run_result["execution_successful"] is True
            assert len(dry_run_result["blocks_executed"]) == 2

            # Check that runtime parameters were passed
            context_block_params = dry_run_result["blocks_executed"][0][
                "parameters_used"
            ]
            assert context_block_params["processing_mode"] == "detailed"

            llm_block_params = dry_run_result["blocks_executed"][1]["parameters_used"]
            assert llm_block_params["temperature"] == 0.3
            assert llm_block_params["max_tokens"] == 256

    def test_flow_serialization_and_deserialization(self, temp_dir, mock_block):
        """Test flow serialization to YAML and back."""
        # Create a flow programmatically
        metadata = FlowMetadata(
            name="Serialization Test Flow",
            description="Test flow for serialization",
            version="1.0.0",
            author="Test Author",
            recommended_models={
                "default": "test-model",
                "compatible": [],
                "experimental": [],
            },
            tags=["serialization", "test"],
        )

        # Create mock blocks
        block1 = mock_block("block1", ["input"], ["intermediate"])
        block2 = mock_block("block2", ["intermediate"], ["output"])

        flow = Flow(blocks=[block1, block2], metadata=metadata)

        # Serialize to YAML
        yaml_path = Path(temp_dir) / "serialized_flow.yaml"
        flow.to_yaml(str(yaml_path))

        # Verify the file exists and contains expected content
        assert yaml_path.exists()

        with open(yaml_path) as f:
            content = f.read()

        assert "Serialization Test Flow" in content
        assert "1.0.0" in content
        assert "MockBlock" in content
        assert "block1" in content
        assert "block2" in content

        # Test flow info
        info = flow.get_info()
        assert info["metadata"]["name"] == "Serialization Test Flow"
        assert info["total_blocks"] == 2
        assert info["block_names"] == ["block1", "block2"]

    def test_error_handling_and_edge_cases(self, temp_dir):
        """Test error handling and edge cases."""
        # Test with malformed YAML
        malformed_yaml = Path(temp_dir) / "malformed.yaml"
        with open(malformed_yaml, "w") as f:
            f.write("invalid: yaml: content:")

        with pytest.raises(Exception):  # Should raise FlowValidationError
            Flow.from_yaml(str(malformed_yaml))

        # Test with missing required sections
        incomplete_flow = {
            "metadata": {"name": "Incomplete Flow"}
            # Missing blocks section
        }

        incomplete_yaml = Path(temp_dir) / "incomplete.yaml"
        with open(incomplete_yaml, "w") as f:
            yaml.dump(incomplete_flow, f)

        with pytest.raises(Exception):  # Should raise FlowValidationError
            Flow.from_yaml(str(incomplete_yaml))

        # Test with empty flow
        empty_flow = Flow(blocks=[], metadata=FlowMetadata(name="Empty Flow"))
        dataset = Dataset.from_dict({"input": ["test"]})

        # Should fail on generate
        with pytest.raises(Exception):
            empty_flow.generate(dataset)

        # Should fail on dry_run
        with pytest.raises(Exception):
            empty_flow.dry_run(dataset)
