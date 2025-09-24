# SPDX-License-Identifier: Apache-2.0
"""Tests for the base Flow class."""

# Standard
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

# Third Party
from datasets import Dataset
from pydantic import ValidationError

# First Party
from sdg_hub import Flow, FlowMetadata, FlowParameter
from sdg_hub.core.flow.metadata import DatasetRequirements
from sdg_hub.core.utils.error_handling import EmptyDatasetError, FlowValidationError
import pytest
import yaml


class TestFlow:
    """Test Flow class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # First Party
        from sdg_hub.core.flow.metadata import RecommendedModels

        self.test_metadata = FlowMetadata(
            name="Test Flow",
            description="A test flow",
            version="1.0.0",
            author="Test Author",
            recommended_models=RecommendedModels(
                default="test-model", compatible=["alt-model"], experimental=[]
            ),
            tags=["test"],
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        # Standard
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_mock_block(self, name="test_block", input_cols=None, output_cols=None):
        """Create a mock block for testing."""
        # First Party
        from tests.flow.conftest import MockBlock

        return MockBlock(
            block_name=name,
            input_cols=input_cols or ["input"],
            output_cols=output_cols or ["output"],
        )

    def test_flow_creation_empty(self):
        """Test creating an empty flow."""
        flow = Flow(blocks=[], metadata=self.test_metadata)
        assert len(flow.blocks) == 0
        assert flow.metadata.name == "Test Flow"
        assert flow.parameters == {}

    def test_flow_creation_with_blocks(self):
        """Test creating a flow with blocks."""
        block1 = self.create_mock_block("block1")
        block2 = self.create_mock_block("block2")

        flow = Flow(blocks=[block1, block2], metadata=self.test_metadata)
        assert len(flow.blocks) == 2
        assert flow.blocks[0].block_name == "block1"
        assert flow.blocks[1].block_name == "block2"

    def test_flow_creation_with_parameters(self):
        """Test creating a flow with parameters."""
        param1 = FlowParameter(default="value1", description="Test param 1")
        param2 = FlowParameter(default=42, description="Test param 2", required=True)

        flow = Flow(
            blocks=[],
            metadata=self.test_metadata,
            parameters={"param1": param1, "param2": param2},
        )

        assert len(flow.parameters) == 2
        assert flow.parameters["param1"].default == "value1"
        assert flow.parameters["param2"].default == 42
        assert flow.parameters["param2"].required is True

    def test_validate_blocks_invalid_type(self):
        """Test block validation with invalid block type."""
        with pytest.raises(ValidationError) as exc_info:
            Flow(blocks=["not a block"], metadata=self.test_metadata)

        assert "instance of BaseBlock" in str(exc_info.value)

    def test_validate_block_names_unique(self):
        """Test validation of unique block names."""
        block1 = self.create_mock_block("duplicate_name")
        block2 = self.create_mock_block("duplicate_name")

        with pytest.raises(ValidationError) as exc_info:
            Flow(blocks=[block1, block2], metadata=self.test_metadata)

        assert "Duplicate block name" in str(exc_info.value)

    def test_validate_parameters_invalid_name(self):
        """Test parameter validation with invalid name."""
        param = FlowParameter(default="value")

        with pytest.raises(ValidationError) as exc_info:
            Flow(
                blocks=[],
                metadata=self.test_metadata,
                parameters={"": param},  # Empty name
            )

        assert "non-empty string" in str(exc_info.value)

    def test_validate_parameters_invalid_type(self):
        """Test parameter validation with invalid parameter type."""
        with pytest.raises(ValidationError) as exc_info:
            Flow(
                blocks=[],
                metadata=self.test_metadata,
                parameters={"param": "not a FlowParameter"},
            )

        assert "instance of FlowParameter" in str(exc_info.value)

    def test_from_yaml_valid_file(self):
        """Test loading flow from valid YAML file."""
        flow_config = {
            "metadata": {
                "name": "YAML Flow",
                "description": "Flow from YAML",
                "version": "1.0.0",
                "recommended_models": {
                    "default": "test-model",
                    "compatible": ["alt-model"],
                    "experimental": [],
                },
            },
            "blocks": [
                {
                    "block_type": "LLMChatBlock",
                    "block_config": {
                        "block_name": "test_block",
                        "input_cols": "input",
                        "output_cols": "output",
                    },
                }
            ],
        }

        yaml_path = Path(self.temp_dir) / "test_flow.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(flow_config, f)

        # Mock the block creation
        with patch("sdg_hub.core.flow.base.BlockRegistry._get") as mock_get:
            mock_block_class = Mock()
            mock_block_instance = self.create_mock_block("test_block")
            mock_block_class.return_value = mock_block_instance
            mock_get.return_value = mock_block_class

            flow = Flow.from_yaml(str(yaml_path))

            assert flow.metadata.name == "YAML Flow"
            assert len(flow.blocks) == 1
            assert flow.blocks[0].block_name == "test_block"

    def test_from_yaml_file_not_found(self):
        """Test loading flow from non-existent file."""
        with pytest.raises(FileNotFoundError):
            Flow.from_yaml("/nonexistent/path.yaml")

    def test_from_yaml_invalid_yaml(self):
        """Test loading flow from invalid YAML file."""
        yaml_path = Path(self.temp_dir) / "invalid.yaml"
        with open(yaml_path, "w") as f:
            f.write("invalid: yaml: content:")

        with pytest.raises(FlowValidationError) as exc_info:
            Flow.from_yaml(str(yaml_path))

        assert "Invalid YAML" in str(exc_info.value)

    def test_from_yaml_new_format(self):
        """Test loading flow with new recommended_models format."""
        flow_config = {
            "metadata": {
                "name": "New Format Flow",
                "recommended_models": {
                    "default": "meta-llama/Llama-3.3-70B-Instruct",
                    "compatible": ["microsoft/phi-4"],
                    "experimental": [],
                },
            },
            "blocks": [
                {
                    "block_type": "LLMChatBlock",
                    "block_config": {
                        "block_name": "test_block",
                        "input_cols": "input",
                        "output_cols": "output",
                    },
                }
            ],
        }

        yaml_path = Path(self.temp_dir) / "new_format.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(flow_config, f)

        # Mock the block creation
        with patch("sdg_hub.core.flow.base.BlockRegistry._get") as mock_get:
            mock_block_class = Mock()
            mock_block_instance = self.create_mock_block("test_block")
            mock_block_class.return_value = mock_block_instance
            mock_get.return_value = mock_block_class

            flow = Flow.from_yaml(str(yaml_path))

            # Should have new format
            assert (
                flow.metadata.recommended_models.default
                == "meta-llama/Llama-3.3-70B-Instruct"
            )
            assert flow.metadata.recommended_models.compatible == ["microsoft/phi-4"]

    def test_generate_empty_flow(self):
        """Test generating with empty flow."""
        flow = Flow(blocks=[], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"]})

        with pytest.raises(FlowValidationError) as exc_info:
            flow.generate(dataset)

        assert "empty flow" in str(exc_info.value)

    def test_generate_empty_dataset(self):
        """Test generating with empty dataset."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        empty_dataset = Dataset.from_dict({"input": []})

        with pytest.raises(EmptyDatasetError) as exc_info:
            flow.generate(empty_dataset)

        assert "empty" in str(exc_info.value)

    def test_generate_with_dataset_requirements(self):
        """Test generating with dataset requirements."""
        requirements = DatasetRequirements(required_columns=["input"], min_samples=2)
        metadata = FlowMetadata(name="Test Flow", dataset_requirements=requirements)

        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=metadata)

        # Valid dataset
        Dataset.from_dict({"input": ["test1", "test2"]})
        # This would work if we had real blocks

        # Invalid dataset - missing column
        invalid_dataset = Dataset.from_dict({"wrong_col": ["test1", "test2"]})
        with pytest.raises(FlowValidationError) as exc_info:
            flow.generate(invalid_dataset)

        assert "validation failed" in str(exc_info.value)

    def test_generate_success(self):
        """Test successful generation."""
        block = self.create_mock_block("test_block", output_cols=["output"])
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2"]})

        result = flow.generate(dataset)

        assert len(result) == 2
        assert "output" in result.column_names
        assert result["output"] == ["test_block_output_0", "test_block_output_1"]

    def test_generate_with_runtime_params(self):
        """Test generation with runtime parameters."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"]})

        runtime_params = {"test_block": {"temperature": 0.5, "max_tokens": 100}}

        # Runtime parameters are passed to the block but we can't easily test them
        # in this mock setup. The test just verifies the flow runs without error.
        result = flow.generate(dataset, runtime_params=runtime_params)

        assert len(result) == 1
        assert "output" in result.column_names

    def test_validate_dataset_empty(self):
        """Test dataset validation with empty dataset."""
        flow = Flow(blocks=[], metadata=self.test_metadata)
        empty_dataset = Dataset.from_dict({"input": []})

        errors = flow.validate_dataset(empty_dataset)
        assert len(errors) == 1
        assert "empty" in errors[0]

    def test_validate_dataset_with_requirements(self):
        """Test dataset validation with requirements."""
        requirements = DatasetRequirements(
            required_columns=["input", "label"], min_samples=5
        )
        metadata = FlowMetadata(name="Test Flow", dataset_requirements=requirements)
        flow = Flow(blocks=[], metadata=metadata)

        # Valid dataset
        valid_dataset = Dataset.from_dict(
            {"input": ["test"] * 5, "label": ["label"] * 5}
        )
        errors = flow.validate_dataset(valid_dataset)
        assert errors == []

        # Invalid dataset
        invalid_dataset = Dataset.from_dict(
            {
                "input": ["test"] * 3,  # Too few samples
                # Missing label column
            }
        )
        errors = flow.validate_dataset(invalid_dataset)
        assert len(errors) == 2

    def test_dry_run_empty_flow(self):
        """Test dry run with empty flow."""
        flow = Flow(blocks=[], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"]})

        with pytest.raises(FlowValidationError) as exc_info:
            flow.dry_run(dataset)

        assert "empty flow" in str(exc_info.value)

    def test_dry_run_empty_dataset(self):
        """Test dry run with empty dataset."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        empty_dataset = Dataset.from_dict({"input": []})

        with pytest.raises(EmptyDatasetError):
            flow.dry_run(empty_dataset)

    def test_dry_run_success(self):
        """Test successful dry run."""
        block = self.create_mock_block("test_block", output_cols=["output"])
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2", "test3"]})

        result = flow.dry_run(dataset, sample_size=2)

        assert result["flow_name"] == "Test Flow"
        assert result["sample_size"] == 2
        assert result["original_dataset_size"] == 3
        assert result["execution_successful"] is True
        assert len(result["blocks_executed"]) == 1
        assert result["blocks_executed"][0]["block_name"] == "test_block"
        assert result["blocks_executed"][0]["input_rows"] == 2
        assert result["blocks_executed"][0]["output_rows"] == 2

    def test_dry_run_sample_size_adjustment(self):
        """Test dry run with sample size larger than dataset."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2"]})

        result = flow.dry_run(dataset, sample_size=5)

        # Should use actual dataset size
        assert result["sample_size"] == 2

    def test_dry_run_with_runtime_params(self):
        """Test dry run with runtime parameters."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"]})

        runtime_params = {"test_block": {"temperature": 0.3}}

        result = flow.dry_run(dataset, runtime_params=runtime_params)

        assert result["blocks_executed"][0]["parameters_used"]["temperature"] == 0.3

    def test_add_block_success(self):
        """Test successfully adding a block."""
        flow = Flow(blocks=[], metadata=self.test_metadata)
        new_block = self.create_mock_block("new_block")

        new_flow = flow.add_block(new_block)

        assert len(new_flow.blocks) == 1
        assert new_flow.blocks[0].block_name == "new_block"
        assert len(flow.blocks) == 0  # Original flow unchanged

    def test_add_block_duplicate_name(self):
        """Test adding block with duplicate name."""
        existing_block = self.create_mock_block("existing")
        flow = Flow(blocks=[existing_block], metadata=self.test_metadata)

        duplicate_block = self.create_mock_block("existing")

        with pytest.raises(ValueError) as exc_info:
            flow.add_block(duplicate_block)

        assert "already exists" in str(exc_info.value)

    def test_add_block_invalid_type(self):
        """Test adding invalid block type."""
        flow = Flow(blocks=[], metadata=self.test_metadata)

        with pytest.raises(ValueError) as exc_info:
            flow.add_block("not a block")

        assert "BaseBlock instance" in str(exc_info.value)

    def test_get_info(self):
        """Test getting flow information."""
        block = self.create_mock_block(
            "test_block", input_cols=["input"], output_cols=["output"]
        )
        param = FlowParameter(default="test_value", description="Test parameter")

        flow = Flow(
            blocks=[block],
            metadata=self.test_metadata,
            parameters={"test_param": param},
        )

        info = flow.get_info()

        assert info["metadata"]["name"] == "Test Flow"
        assert info["total_blocks"] == 1
        assert info["block_names"] == ["test_block"]
        assert len(info["parameters"]) == 1
        assert info["parameters"]["test_param"]["default"] == "test_value"
        assert len(info["blocks"]) == 1
        assert info["blocks"][0]["block_name"] == "test_block"

    def test_to_yaml(self):
        """Test saving flow to YAML."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)

        output_path = Path(self.temp_dir) / "output.yaml"
        flow.to_yaml(str(output_path))

        assert output_path.exists()

        # Load and verify content - check structure without full parsing
        with open(output_path) as f:
            content = f.read()

        assert "Test Flow" in content
        assert "MockBlock" in content
        assert "test_block" in content

    def test_string_representations(self):
        """Test string representations of flow."""
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)

        # Test __repr__
        repr_str = repr(flow)
        assert "Test Flow" in repr_str
        assert "1.0.0" in repr_str
        assert "blocks=1" in repr_str

        # Test __str__
        str_str = str(flow)
        assert "Test Flow" in str_str
        assert "v1.0.0" in str_str
        assert "test_block" in str_str
        assert "Test Author" in str_str

    def test_len(self):
        """Test flow length."""
        flow = Flow(blocks=[], metadata=self.test_metadata)
        assert len(flow) == 0

        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        assert len(flow) == 1

    def create_mock_llm_block(
        self,
        name="llm_block",
        model="test-model",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
    ):
        """Create a mock LLM block with model attributes."""
        # First Party
        from tests.flow.conftest import MockBlock

        block = MockBlock(block_name=name, input_cols=["input"], output_cols=["output"])
        # Add LLM-related attributes
        block.model = model
        block.api_base = api_base
        block.api_key = api_key
        block.temperature = 0.0
        block.max_tokens = 1024
        return block

    def test_detect_llm_blocks_by_model_attribute(self):
        """Test detecting LLM blocks by model attribute."""
        regular_block = self.create_mock_block("regular_block")
        llm_block = self.create_mock_llm_block("llm_block", model="test-model")

        flow = Flow(blocks=[regular_block, llm_block], metadata=self.test_metadata)

        detected_blocks = flow._detect_llm_blocks()

        assert len(detected_blocks) == 1
        assert "llm_block" in detected_blocks
        assert "regular_block" not in detected_blocks

    def test_detect_llm_blocks_by_api_base_attribute(self):
        """Test detecting LLM blocks by api_base attribute."""
        regular_block = self.create_mock_block("regular_block")
        llm_block = self.create_mock_llm_block("llm_block", model=None)
        llm_block.model = None  # Remove model but keep api_base

        flow = Flow(blocks=[regular_block, llm_block], metadata=self.test_metadata)

        detected_blocks = flow._detect_llm_blocks()

        assert len(detected_blocks) == 1
        assert "llm_block" in detected_blocks

    def test_detect_llm_blocks_by_api_key_attribute(self):
        """Test detecting LLM blocks by api_key attribute."""
        regular_block = self.create_mock_block("regular_block")
        llm_block = self.create_mock_llm_block("llm_block", model=None, api_base=None)
        llm_block.model = None
        llm_block.api_base = None
        # Only api_key remains

        flow = Flow(blocks=[regular_block, llm_block], metadata=self.test_metadata)

        detected_blocks = flow._detect_llm_blocks()

        assert len(detected_blocks) == 1
        assert "llm_block" in detected_blocks

    def test_detect_llm_blocks_none_found(self):
        """Test detecting LLM blocks when none exist."""
        regular_block1 = self.create_mock_block("regular_block1")
        regular_block2 = self.create_mock_block("regular_block2")

        flow = Flow(
            blocks=[regular_block1, regular_block2], metadata=self.test_metadata
        )

        detected_blocks = flow._detect_llm_blocks()

        assert len(detected_blocks) == 0

    def test_detect_llm_blocks_multiple(self):
        """Test detecting multiple LLM blocks."""
        regular_block = self.create_mock_block("regular_block")
        llm_block1 = self.create_mock_llm_block("llm_block1", model="model1")
        llm_block2 = self.create_mock_llm_block("llm_block2", model="model2")
        llm_block3 = self.create_mock_llm_block(
            "llm_block3", model=None, api_base="http://localhost:8001/v1"
        )
        llm_block3.model = None  # Only has api_base

        flow = Flow(
            blocks=[regular_block, llm_block1, llm_block2, llm_block3],
            metadata=self.test_metadata,
        )

        detected_blocks = flow._detect_llm_blocks()

        assert len(detected_blocks) == 3
        assert "llm_block1" in detected_blocks
        assert "llm_block2" in detected_blocks
        assert "llm_block3" in detected_blocks
        assert "regular_block" not in detected_blocks

    def test_set_model_config_all_llm_blocks(self):
        """Test set_model_config with auto-detection of all LLM blocks."""
        regular_block = self.create_mock_block("regular_block")
        llm_block1 = self.create_mock_llm_block("llm_block1", model="old-model1")
        llm_block2 = self.create_mock_llm_block("llm_block2", model="old-model2")

        flow = Flow(
            blocks=[regular_block, llm_block1, llm_block2], metadata=self.test_metadata
        )

        # Configure model for all LLM blocks
        flow.set_model_config(
            model="new-model",
            api_base="http://localhost:8101/v1",
            api_key="NEW_KEY",
            temperature=0.7,
            max_tokens=2048,
        )

        # Check that LLM blocks were modified
        assert flow.blocks[1].model == "new-model"  # llm_block1
        assert flow.blocks[1].api_base == "http://localhost:8101/v1"
        assert flow.blocks[1].api_key == "NEW_KEY"
        assert flow.blocks[1].temperature == 0.7
        assert flow.blocks[1].max_tokens == 2048

        assert flow.blocks[2].model == "new-model"  # llm_block2
        assert flow.blocks[2].api_base == "http://localhost:8101/v1"

        # Check that regular block was not modified (doesn't have these attributes)
        assert not hasattr(flow.blocks[0], "model")

    def test_set_model_config_specific_blocks(self):
        """Test set_model_config with specific block targeting."""
        llm_block1 = self.create_mock_llm_block("llm_block1", model="old-model1")
        llm_block2 = self.create_mock_llm_block("llm_block2", model="old-model2")
        llm_block3 = self.create_mock_llm_block("llm_block3", model="old-model3")

        flow = Flow(
            blocks=[llm_block1, llm_block2, llm_block3], metadata=self.test_metadata
        )

        # Configure only specific blocks
        flow.set_model_config(
            model="new-model",
            api_base="http://localhost:8101/v1",
            blocks=["llm_block1", "llm_block3"],
        )

        # Check that only specified blocks were modified
        assert flow.blocks[0].model == "new-model"  # llm_block1
        assert flow.blocks[0].api_base == "http://localhost:8101/v1"

        assert flow.blocks[1].model == "old-model2"  # llm_block2 unchanged
        assert flow.blocks[1].api_base == "http://localhost:8000/v1"  # unchanged

        assert flow.blocks[2].model == "new-model"  # llm_block3
        assert flow.blocks[2].api_base == "http://localhost:8101/v1"

    def test_set_model_config_partial_parameters(self):
        """Test set_model_config with only some parameters."""
        llm_block = self.create_mock_llm_block(
            "llm_block",
            model="old-model",
            api_base="http://localhost:8000/v1",
            api_key="OLD_KEY",
        )
        llm_block.temperature = 0.0
        llm_block.max_tokens = 1024

        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)

        # Configure only model and temperature
        flow.set_model_config(model="new-model", temperature=0.8)

        # Check that only specified parameters were changed
        assert flow.blocks[0].model == "new-model"
        assert flow.blocks[0].temperature == 0.8

        # Other parameters should remain unchanged
        assert flow.blocks[0].api_base == "http://localhost:8000/v1"
        assert flow.blocks[0].api_key == "OLD_KEY"
        assert flow.blocks[0].max_tokens == 1024

    def test_set_model_config_with_kwargs(self):
        """Test set_model_config with additional kwargs."""
        llm_block = self.create_mock_llm_block("llm_block")
        llm_block.top_p = 1.0
        llm_block.frequency_penalty = 0.0

        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)

        # Configure with additional parameters via kwargs
        flow.set_model_config(model="new-model", top_p=0.9, frequency_penalty=0.1)

        assert flow.blocks[0].model == "new-model"
        assert flow.blocks[0].top_p == 0.9
        assert flow.blocks[0].frequency_penalty == 0.1

    def test_set_model_config_no_parameters(self):
        """Test set_model_config with no parameters raises error."""
        llm_block = self.create_mock_llm_block("llm_block")
        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)

        with pytest.raises(ValueError) as exc_info:
            flow.set_model_config()

        assert "At least one configuration parameter must be provided" in str(
            exc_info.value
        )

    def test_set_model_config_invalid_block_names(self):
        """Test set_model_config with invalid block names raises error."""
        llm_block = self.create_mock_llm_block("llm_block")
        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)

        with pytest.raises(ValueError) as exc_info:
            flow.set_model_config(
                model="new-model", blocks=["nonexistent_block", "another_missing_block"]
            )

        assert "Specified blocks not found in flow" in str(exc_info.value)
        assert "nonexistent_block" in str(exc_info.value)
        assert "another_missing_block" in str(exc_info.value)

    def test_set_model_config_no_llm_blocks_detected(self):
        """Test set_model_config when no LLM blocks are detected."""
        regular_block1 = self.create_mock_block("regular_block1")
        regular_block2 = self.create_mock_block("regular_block2")

        flow = Flow(
            blocks=[regular_block1, regular_block2], metadata=self.test_metadata
        )

        # Should not raise error but log warning
        flow.set_model_config(model="new-model")

        # Blocks should remain unchanged
        assert not hasattr(flow.blocks[0], "model")
        assert not hasattr(flow.blocks[1], "model")

    def test_set_model_config_missing_attributes_warning(self):
        """Test set_model_config logs warning for missing attributes."""
        # Create a block that has model but not other attributes
        llm_block = self.create_mock_llm_block("llm_block")
        delattr(llm_block, "api_base")  # Remove api_base attribute

        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)

        # This should work for model but log warning for api_base
        flow.set_model_config(model="new-model", api_base="http://localhost:8101/v1")

        # Model should be changed
        assert flow.blocks[0].model == "new-model"
        # api_base should not be set since attribute doesn't exist

    def test_set_model_config_preserves_unspecified_attributes(self):
        """Test that set_model_config preserves attributes not specified."""
        llm_block = self.create_mock_llm_block(
            "llm_block",
            model="original-model",
            api_base="http://localhost:8000/v1",
            api_key="ORIGINAL_KEY",
        )
        llm_block.temperature = 0.5
        llm_block.max_tokens = 1024
        llm_block.custom_param = "custom_value"

        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)

        # Configure only model
        flow.set_model_config(model="new-model")

        # Only model should change
        assert flow.blocks[0].model == "new-model"

        # Everything else should remain the same
        assert flow.blocks[0].api_base == "http://localhost:8000/v1"
        assert flow.blocks[0].api_key == "ORIGINAL_KEY"
        assert flow.blocks[0].temperature == 0.5
        assert flow.blocks[0].max_tokens == 1024
        assert flow.blocks[0].custom_param == "custom_value"

    def test_generate_requires_model_config_for_llm_flows(self):
        """Test that generate() requires set_model_config() for flows with LLM blocks."""
        llm_block = self.create_mock_llm_block(
            "llm_block", model=None
        )  # No model configured
        llm_block.model = None  # Ensure no model set

        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"]})

        # Should fail because model config not set
        with pytest.raises(FlowValidationError) as exc_info:
            flow.generate(dataset)

        assert "Model configuration required before generate()" in str(exc_info.value)
        assert "llm_block" in str(exc_info.value)
        assert "Call flow.set_model_config() first" in str(exc_info.value)

    def test_generate_allows_execution_after_model_config(self):
        """Test that generate() works after set_model_config() is called."""
        llm_block = self.create_mock_llm_block("llm_block", model=None)
        llm_block.model = None

        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"]})

        # Configure model first
        flow.set_model_config(
            model="test-model", api_base="http://localhost:8000/v1", api_key="EMPTY"
        )

        # Now generate should work
        result = flow.generate(dataset)
        assert len(result) == 1

    def test_generate_works_for_non_llm_flows(self):
        """Test that generate() works without model config for flows without LLM blocks."""
        regular_block = self.create_mock_block("regular_block")
        flow = Flow(blocks=[regular_block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"]})

        # Should work without set_model_config() because no LLM blocks
        result = flow.generate(dataset)
        assert len(result) == 1

    def test_generate_requires_model_config_even_for_llm_blocks_with_attributes(self):
        """Test that generate() requires set_model_config() even if blocks have LLM attributes."""
        # Create an LLM block that has model attributes but no values
        llm_block = self.create_mock_llm_block(
            "llm_block", model=None, api_base=None, api_key=None
        )
        llm_block.model = None
        llm_block.api_base = None
        llm_block.api_key = None

        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test"]})

        # Should fail because model config not set (new approach)
        with pytest.raises(FlowValidationError) as exc_info:
            flow.generate(dataset)

        assert "Model configuration required before generate()" in str(exc_info.value)

        # After calling set_model_config(), it should work
        flow.set_model_config(
            model="test-model", api_base="http://localhost:8000/v1", api_key="EMPTY"
        )
        result = flow.generate(dataset)
        assert len(result) == 1

    def test_is_model_config_required(self):
        """Test is_model_config_required() method."""
        # Flow without LLM blocks
        regular_block = self.create_mock_block("regular_block")
        flow = Flow(blocks=[regular_block], metadata=self.test_metadata)
        assert not flow.is_model_config_required()

        # Flow with LLM blocks
        llm_block = self.create_mock_llm_block("llm_block")
        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)
        assert flow.is_model_config_required()

    def test_is_model_config_set(self):
        """Test is_model_config_set() method."""
        llm_block = self.create_mock_llm_block("llm_block", model=None)
        llm_block.model = None

        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)

        # Should be False initially
        assert not flow.is_model_config_set()

        # Should be True after calling set_model_config()
        flow.set_model_config(model="test-model")
        assert flow.is_model_config_set()

    def test_reset_model_config(self):
        """Test reset_model_config() method."""
        llm_block = self.create_mock_llm_block("llm_block", model=None)
        llm_block.model = None

        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)

        # Configure model
        flow.set_model_config(model="test-model")
        assert flow.is_model_config_set()

        # Reset should make it False again
        flow.reset_model_config()
        assert not flow.is_model_config_set()

    def test_get_default_model_with_new_format(self):
        """Test get_default_model() with new simplified format."""
        # First Party
        from sdg_hub.core.flow.metadata import RecommendedModels

        recommended_models = RecommendedModels(
            default="meta-llama/Llama-3.3-70B-Instruct",
            compatible=["microsoft/phi-4", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
            experimental=[],
        )

        metadata = FlowMetadata(name="Test Flow", recommended_models=recommended_models)

        flow = Flow(blocks=[], metadata=metadata)

        assert flow.get_default_model() == "meta-llama/Llama-3.3-70B-Instruct"

    def test_get_default_model_no_recommendations(self):
        """Test get_default_model() when no models are recommended."""
        metadata = FlowMetadata(name="Test Flow")
        flow = Flow(blocks=[], metadata=metadata)

        assert flow.get_default_model() is None

    def test_get_model_recommendations(self):
        """Test get_model_recommendations() method."""
        # First Party
        from sdg_hub.core.flow.metadata import RecommendedModels

        recommended_models = RecommendedModels(
            default="meta-llama/Llama-3.3-70B-Instruct",
            compatible=["microsoft/phi-4", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
            experimental=["experimental-model"],
        )

        metadata = FlowMetadata(name="Test Flow", recommended_models=recommended_models)

        flow = Flow(blocks=[], metadata=metadata)
        recommendations = flow.get_model_recommendations()

        assert recommendations["default"] == "meta-llama/Llama-3.3-70B-Instruct"
        assert recommendations["compatible"] == [
            "microsoft/phi-4",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ]
        assert recommendations["experimental"] == ["experimental-model"]

    def test_get_model_recommendations_no_models(self):
        """Test get_model_recommendations() when no models are specified."""
        metadata = FlowMetadata(name="Test Flow")
        flow = Flow(blocks=[], metadata=metadata)

        recommendations = flow.get_model_recommendations()

        assert recommendations["default"] is None
        assert recommendations["compatible"] == []
        assert recommendations["experimental"] == []

    def test_generate_with_checkpointing_disabled(self):
        """Test generation without checkpointing."""
        block = self.create_mock_block("test_block", output_cols=["output"])
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2"]})

        # Should work the same as before when no checkpoint_dir provided
        result = flow.generate(dataset)

        assert len(result) == 2
        assert "output" in result.column_names

    def test_generate_with_checkpointing_no_existing_data(self):
        """Test generation with checkpointing when no existing checkpoints."""
        block = self.create_mock_block("test_block", output_cols=["output"])
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2"]})

        checkpoint_dir = Path(self.temp_dir) / "checkpoints"

        result = flow.generate(dataset, checkpoint_dir=str(checkpoint_dir))

        assert len(result) == 2
        assert "output" in result.column_names

        # Should have created checkpoint files
        assert checkpoint_dir.exists()
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 1

        # Should have metadata file
        metadata_file = checkpoint_dir / "flow_metadata.json"
        assert metadata_file.exists()

    def test_generate_with_checkpointing_and_save_freq(self):
        """Test generation with checkpointing and save frequency."""
        block = self.create_mock_block("test_block", output_cols=["output"])
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict(
            {"input": ["test1", "test2", "test3", "test4", "test5"]}
        )

        checkpoint_dir = Path(self.temp_dir) / "checkpoints_freq"
        save_freq = 2

        result = flow.generate(
            dataset, checkpoint_dir=str(checkpoint_dir), save_freq=save_freq
        )

        assert len(result) == 5
        assert "output" in result.column_names

        # Should have created multiple checkpoint files (3 chunks: 2+2+1)
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 3

    def test_generate_with_existing_checkpoints(self):
        """Test generation resuming from existing checkpoints."""
        block = self.create_mock_block("test_block", output_cols=["output"])
        flow = Flow(blocks=[block], metadata=self.test_metadata)

        checkpoint_dir = Path(self.temp_dir) / "existing_checkpoints"
        checkpoint_dir.mkdir(parents=True)

        # Pre-create some checkpoint data manually
        # First Party
        from sdg_hub.core.flow.checkpointer import FlowCheckpointer

        checkpointer = FlowCheckpointer(
            checkpoint_dir=str(checkpoint_dir),
            save_freq=2,  # Need save_freq to trigger checkpoint save
            flow_id=flow.metadata.id,
        )

        # Simulate some completed samples
        completed_data = Dataset.from_dict(
            {"input": ["test1", "test2"], "output": ["existing1", "existing2"]}
        )
        checkpointer.add_completed_samples(completed_data)

        # Now run flow with larger input dataset
        full_dataset = Dataset.from_dict(
            {"input": ["test1", "test2", "test3", "test4"]}
        )

        result = flow.generate(full_dataset, checkpoint_dir=str(checkpoint_dir))

        # Should have 4 samples total: 2 existing + 2 newly processed
        assert len(result) == 4

        # Should include both existing and new outputs
        assert "existing1" in result["output"]
        assert "existing2" in result["output"]
        assert "test_block_output_0" in result["output"]  # New outputs
        assert "test_block_output_1" in result["output"]

    def test_generate_all_samples_already_completed(self):
        """Test generation when all samples are already completed."""
        block = self.create_mock_block("test_block", output_cols=["output"])
        flow = Flow(blocks=[block], metadata=self.test_metadata)

        checkpoint_dir = Path(self.temp_dir) / "all_completed"
        checkpoint_dir.mkdir(parents=True)

        # Pre-create checkpoint data for all input samples
        # First Party
        from sdg_hub.core.flow.checkpointer import FlowCheckpointer

        checkpointer = FlowCheckpointer(
            checkpoint_dir=str(checkpoint_dir),
            save_freq=2,  # Need save_freq to trigger checkpoint save
            flow_id=flow.metadata.id,
        )

        completed_data = Dataset.from_dict(
            {"input": ["test1", "test2"], "output": ["existing1", "existing2"]}
        )
        checkpointer.add_completed_samples(completed_data)

        # Run flow with same input dataset
        input_dataset = Dataset.from_dict({"input": ["test1", "test2"]})

        result = flow.generate(input_dataset, checkpoint_dir=str(checkpoint_dir))

        # Should just return existing results without processing
        assert len(result) == 2
        assert result["output"] == ["existing1", "existing2"]

    def test_generate_with_runtime_params_and_checkpointing(self):
        """Test generation with both runtime params and checkpointing."""
        block = self.create_mock_block("test_block", output_cols=["output"])
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2"]})

        checkpoint_dir = Path(self.temp_dir) / "checkpoints_with_params"
        runtime_params = {"test_block": {"temperature": 0.7, "max_tokens": 150}}

        result = flow.generate(
            dataset,
            runtime_params=runtime_params,
            checkpoint_dir=str(checkpoint_dir),
            save_freq=1,
        )

        assert len(result) == 2
        assert "output" in result.column_names

        # Checkpointing should still work with runtime params
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 2  # save_freq=1 means each sample gets saved

    def test_checkpointing_with_multiple_blocks(self):
        """Test checkpointing with multiple blocks in the flow."""
        block1 = self.create_mock_block("block1", output_cols=["intermediate"])
        block2 = self.create_mock_block(
            "block2", input_cols=["intermediate"], output_cols=["final"]
        )

        flow = Flow(blocks=[block1, block2], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2"]})

        checkpoint_dir = Path(self.temp_dir) / "multi_block_checkpoints"

        result = flow.generate(dataset, checkpoint_dir=str(checkpoint_dir), save_freq=1)

        # Should have processed through both blocks
        assert len(result) == 2
        assert "final" in result.column_names

        # Should save final results only (after all blocks completed)
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 2

        # Verify checkpoint content includes results from all blocks
        # Standard
        import json

        with open(checkpoint_files[0], "r") as f:
            checkpoint_data = json.loads(f.readline())
            assert "final" in checkpoint_data

    def test_generate_with_log_dir(self):
        """Test generation with log_dir parameter for dual logging."""
        block = self.create_mock_block("test_block", output_cols=["output"])
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2"]})

        log_dir = Path(self.temp_dir) / "test_logs"

        # Ensure INFO level logging for this test regardless of LOG_LEVEL env var
        with patch.dict("os.environ", {"LOG_LEVEL": "INFO"}):
            result = flow.generate(dataset, log_dir=str(log_dir))

        assert len(result) == 2
        assert "output" in result.column_names

        # Should have created log directory and log file
        assert log_dir.exists()
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) == 1

        # Check log file content
        log_file = log_files[0]
        with open(log_file, "r", encoding="utf-8") as f:
            log_content = f.read()

        # Should contain flow execution logs
        assert "Starting flow 'Test Flow'" in log_content
        assert "completed successfully" in log_content
        assert "test_block" in log_content

        # Log filename should include flow name and timestamp
        assert "test_flow_" in log_file.name
        assert log_file.name.endswith(".log")

    def test_generate_without_log_dir(self):
        """Test generation without log_dir parameter (original behavior)."""
        block = self.create_mock_block("test_block", output_cols=["output"])
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2"]})

        # Should work without log_dir (backward compatibility)
        result = flow.generate(dataset)

        assert len(result) == 2
        assert "output" in result.column_names

    def test_generate_with_log_dir_creates_directory(self):
        """Test that log_dir is created if it doesn't exist."""
        block = self.create_mock_block("test_block", output_cols=["output"])
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1"]})

        # Use a nested path that doesn't exist
        log_dir = Path(self.temp_dir) / "nested" / "log" / "directory"
        assert not log_dir.exists()

        result = flow.generate(dataset, log_dir=str(log_dir))

        assert len(result) == 1
        # Directory should be created
        assert log_dir.exists()
        # Log file should be created
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) == 1

    def test_generate_with_log_dir_and_checkpointing(self):
        """Test generation with both log_dir and checkpointing."""
        block = self.create_mock_block("test_block", output_cols=["output"])
        flow = Flow(blocks=[block], metadata=self.test_metadata)
        dataset = Dataset.from_dict({"input": ["test1", "test2"]})

        log_dir = Path(self.temp_dir) / "logs_with_checkpoints"
        checkpoint_dir = Path(self.temp_dir) / "checkpoints_with_logs"

        result = flow.generate(
            dataset,
            log_dir=str(log_dir),
            checkpoint_dir=str(checkpoint_dir),
            save_freq=1,
        )

        assert len(result) == 2

        # Both log and checkpoint directories should exist
        assert log_dir.exists()
        assert checkpoint_dir.exists()

        # Should have log files
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) == 1

        # Should have checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.jsonl"))
        assert len(checkpoint_files) == 2

    def test_create_block_from_config_block_not_found(self):
        """Test _create_block_from_config when block type is not found in registry."""
        # Standard
        from pathlib import Path

        # Mock BlockRegistry to simulate available blocks
        with (
            patch("sdg_hub.core.flow.base.BlockRegistry._get") as mock_get,
            patch(
                "sdg_hub.core.flow.base.BlockRegistry.list_blocks"
            ) as mock_list_blocks,
        ):
            # Configure mocks
            mock_get.side_effect = KeyError("Block 'nonexistent_block' not found")
            mock_list_blocks.return_value = [
                "llm_chat",
                "prompt_builder",
                "text_concat",
                "rename_columns",
                "column_value_filter",
            ]

            # Create block config with nonexistent block type
            block_config = {
                "block_type": "nonexistent_block",
                "block_config": {"param": "value"},
            }
            yaml_dir = Path(self.temp_dir)

            # Test that FlowValidationError is raised with helpful message
            with pytest.raises(FlowValidationError) as exc_info:
                Flow._create_block_from_config(block_config, yaml_dir)

            error_message = str(exc_info.value)

            # Verify error message contains expected information
            assert (
                "Block type 'nonexistent_block' not found in registry" in error_message
            )
            assert "Available blocks:" in error_message

            # Verify all available blocks are listed in the error message
            assert "llm_chat" in error_message
            assert "prompt_builder" in error_message
            assert "text_concat" in error_message
            assert "rename_columns" in error_message
            assert "column_value_filter" in error_message

            # Verify the blocks are flattened from all categories
            mock_list_blocks.assert_called_once_with()
            mock_get.assert_called_once_with("nonexistent_block")

    def test_generate_with_max_concurrency_limit(self):
        """Test that max_concurrency limits concurrent requests."""
        # Standard
        from unittest.mock import patch
        import asyncio

        active, max_concurrent = [0], [0]

        async def mock_acompletion(*args, **kwargs):
            active[0] += 1
            max_concurrent[0] = max(max_concurrent[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1
            # Return mock response in LiteLLM format
            from types import SimpleNamespace

            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="response"))]
            )

        # Use real LLMChatBlock
        # First Party
        from sdg_hub.core.blocks.llm.llm_chat_block import LLMChatBlock

        messages_data = [[{"role": "user", "content": f"test {i}"}] for i in range(10)]
        dataset = Dataset.from_dict({"messages": messages_data})

        llm_block = LLMChatBlock(
            block_name="llm_block",
            input_cols="messages",
            output_cols="output",
            model="test/model",
            async_mode=True,
        )

        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)
        flow._model_config_set = True

        # Mock LiteLLM's acompletion function
        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.acompletion",
            side_effect=mock_acompletion,
        ):
            flow.generate(dataset, max_concurrency=3)

        assert max_concurrent[0] <= 3

    def test_generate_without_concurrency_limit(self):
        """Test that without max_concurrency, all requests run concurrently."""
        # Standard
        import asyncio

        active, max_concurrent = [0], [0]

        async def mock_acreate(*args, **kwargs):
            active[0] += 1
            max_concurrent[0] = max(max_concurrent[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1
            return "response"

        # Use real LLMChatBlock instead of MockBlock
        # First Party
        from sdg_hub.core.blocks.llm.llm_chat_block import LLMChatBlock

        # Create dataset with messages format (required for LLMChatBlock)
        messages_data = [[{"role": "user", "content": f"test {i}"}] for i in range(5)]
        dataset = Dataset.from_dict({"messages": messages_data})

        llm_block = LLMChatBlock(
            block_name="llm_block",
            input_cols="messages",
            output_cols="output",
            model="test/model",
            async_mode=True,
        )

        flow = Flow(blocks=[llm_block], metadata=self.test_metadata)
        flow._model_config_set = True

        # Mock LiteLLM's acompletion function
        from types import SimpleNamespace
        from unittest.mock import patch

        async def mock_acompletion(*args, **kwargs):
            active[0] += 1
            max_concurrent[0] = max(max_concurrent[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1
            # Return mock response in LiteLLM format
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="response"))]
            )

        with patch(
            "sdg_hub.core.blocks.llm.llm_chat_block.acompletion",
            side_effect=mock_acompletion,
        ):
            flow.generate(dataset)  # No max_concurrency limit

        assert max_concurrent[0] == 5  # All 5 should run concurrently

    def test_generate_max_concurrency_validation(self):
        """Test that max_concurrency parameter validation works correctly."""
        # Third Party
        from datasets import Dataset

        # First Party
        from sdg_hub.core.utils.error_handling import FlowValidationError

        dataset = Dataset.from_dict(
            {"messages": [[{"role": "user", "content": "test"}]]}
        )
        flow = Flow(blocks=[], metadata=self.test_metadata)

        # Test invalid type
        with pytest.raises(FlowValidationError, match="max_concurrency must be an int"):
            flow.generate(dataset, max_concurrency=3.5)

        # Test zero value
        with pytest.raises(
            FlowValidationError, match="max_concurrency must be greater than 0"
        ):
            flow.generate(dataset, max_concurrency=0)

        # Test negative value
        with pytest.raises(
            FlowValidationError, match="max_concurrency must be greater than 0"
        ):
            flow.generate(dataset, max_concurrency=-1)
