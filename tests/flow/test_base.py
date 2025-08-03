# SPDX-License-Identifier: Apache-2.0
"""Tests for the base Flow class."""

# Standard
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Third Party
import pytest
import yaml
from datasets import Dataset
from pydantic import ValidationError

# Local
from sdg_hub import Flow
from sdg_hub import FlowMetadata, FlowParameter
from sdg_hub.core.flow.metadata import ModelOption, ModelCompatibility, DatasetRequirements
from sdg_hub.core.utils.error_handling import FlowValidationError, EmptyDatasetError


class TestFlow:
    """Test Flow class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_metadata = FlowMetadata(
            name="Test Flow",
            description="A test flow",
            version="1.0.0",
            author="Test Author",
            recommended_models=[
                ModelOption(name="test-model", compatibility=ModelCompatibility.REQUIRED)
            ],
            tags=["test"]
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_mock_block(self, name="test_block", input_cols=None, output_cols=None):
        """Create a mock block for testing."""
        from tests.flow.conftest import MockBlock
        return MockBlock(
            block_name=name,
            input_cols=input_cols or ["input"],
            output_cols=output_cols or ["output"]
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
            parameters={"param1": param1, "param2": param2}
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
                parameters={"": param}  # Empty name
            )
        
        assert "non-empty string" in str(exc_info.value)

    def test_validate_parameters_invalid_type(self):
        """Test parameter validation with invalid parameter type."""
        with pytest.raises(ValidationError) as exc_info:
            Flow(
                blocks=[],
                metadata=self.test_metadata,
                parameters={"param": "not a FlowParameter"}
            )
        
        assert "instance of FlowParameter" in str(exc_info.value)

    def test_from_yaml_valid_file(self):
        """Test loading flow from valid YAML file."""
        flow_config = {
            "metadata": {
                "name": "YAML Flow",
                "description": "Flow from YAML",
                "version": "1.0.0",
                "recommended_models": [
                    {
                        "name": "test-model",
                        "compatibility": "required"
                    }
                ]
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
        
        yaml_path = Path(self.temp_dir) / "test_flow.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(flow_config, f)
        
        # Mock the block creation
        with patch('sdg_hub.core.flow.base.BlockRegistry.get') as mock_get:
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
        with open(yaml_path, 'w') as f:
            f.write("invalid: yaml: content:")
        
        with pytest.raises(FlowValidationError) as exc_info:
            Flow.from_yaml(str(yaml_path))
        
        assert "Invalid YAML" in str(exc_info.value)

    def test_from_yaml_backward_compatibility(self):
        """Test backward compatibility with old recommended_model format."""
        flow_config = {
            "metadata": {
                "name": "Old Format Flow",
                "recommended_model": "old-model"  # Old format
            },
            "blocks": [
                {
                    "block_type": "LLMChatBlock",
                    "block_config": {
                        "block_name": "test_block",
                        "input_cols": "input",
                        "output_cols": "output"
                    }
                }
            ]
        }
        
        yaml_path = Path(self.temp_dir) / "old_format.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(flow_config, f)
        
        # Mock the block creation
        with patch('sdg_hub.core.flow.base.BlockRegistry.get') as mock_get:
            mock_block_class = Mock()
            mock_block_instance = self.create_mock_block("test_block")
            mock_block_class.return_value = mock_block_instance
            mock_get.return_value = mock_block_class
            
            flow = Flow.from_yaml(str(yaml_path))
            
            # Should convert to new format
            assert len(flow.metadata.recommended_models) == 1
            assert flow.metadata.recommended_models[0].name == "old-model"
            assert flow.metadata.recommended_models[0].compatibility == ModelCompatibility.RECOMMENDED

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
        requirements = DatasetRequirements(
            required_columns=["input"],
            min_samples=2
        )
        metadata = FlowMetadata(
            name="Test Flow",
            dataset_requirements=requirements
        )
        
        block = self.create_mock_block("test_block")
        flow = Flow(blocks=[block], metadata=metadata)
        
        # Valid dataset
        valid_dataset = Dataset.from_dict({"input": ["test1", "test2"]})
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
        
        runtime_params = {
            "test_block": {
                "temperature": 0.5,
                "max_tokens": 100
            }
        }
        
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
            required_columns=["input", "label"],
            min_samples=5
        )
        metadata = FlowMetadata(
            name="Test Flow",
            dataset_requirements=requirements
        )
        flow = Flow(blocks=[], metadata=metadata)
        
        # Valid dataset
        valid_dataset = Dataset.from_dict({
            "input": ["test"] * 5,
            "label": ["label"] * 5
        })
        errors = flow.validate_dataset(valid_dataset)
        assert errors == []
        
        # Invalid dataset
        invalid_dataset = Dataset.from_dict({
            "input": ["test"] * 3,  # Too few samples
            # Missing label column
        })
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
        
        runtime_params = {
            "test_block": {
                "temperature": 0.3
            }
        }
        
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
        block = self.create_mock_block("test_block", input_cols=["input"], output_cols=["output"])
        param = FlowParameter(default="test_value", description="Test parameter")
        
        flow = Flow(
            blocks=[block],
            metadata=self.test_metadata,
            parameters={"test_param": param}
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
        with open(output_path, 'r') as f:
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