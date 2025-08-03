# SPDX-License-Identifier: Apache-2.0
"""Tests for flow metadata models."""

# Standard
import pytest
from datetime import datetime

# Third Party
from pydantic import ValidationError

# Local
from sdg_hub import FlowMetadata, FlowParameter
from sdg_hub.core.flow.metadata import (
    ModelOption,
    ModelCompatibility,
    DatasetRequirements,
)


class TestModelOption:
    """Test ModelOption class."""

    def test_model_option_creation(self):
        """Test creating a ModelOption."""
        model = ModelOption(
            name="gpt-4o-mini",
            compatibility=ModelCompatibility.REQUIRED
        )
        assert model.name == "gpt-4o-mini"
        assert model.compatibility == ModelCompatibility.REQUIRED

    def test_model_option_default_compatibility(self):
        """Test default compatibility level."""
        model = ModelOption(name="gpt-3.5-turbo")
        assert model.compatibility == ModelCompatibility.COMPATIBLE

    def test_model_option_validation(self):
        """Test model option validation."""
        # Empty name should fail
        with pytest.raises(ValidationError):
            ModelOption(name="")

        # Whitespace-only name should fail
        with pytest.raises(ValidationError):
            ModelOption(name="   ")

    def test_model_option_name_strip(self):
        """Test name stripping."""
        model = ModelOption(name="  gpt-4o-mini  ")
        assert model.name == "gpt-4o-mini"


class TestFlowParameter:
    """Test FlowParameter class."""

    def test_flow_parameter_creation(self):
        """Test creating a FlowParameter."""
        param = FlowParameter(
            default="test",
            description="Test parameter",
            type_hint="str",
            required=True
        )
        assert param.default == "test"
        assert param.description == "Test parameter"
        assert param.type_hint == "str"
        assert param.required is True

    def test_flow_parameter_defaults(self):
        """Test parameter defaults."""
        param = FlowParameter(default="test")
        assert param.description == ""
        assert param.type_hint == "Any"
        assert param.required is False
        assert param.constraints == {}

    def test_flow_parameter_required_validation(self):
        """Test required parameter validation."""
        # Required parameter with None default should fail
        with pytest.raises(ValidationError):
            FlowParameter(default=None, required=True)

        # Required parameter with non-None default should pass
        param = FlowParameter(default="value", required=True)
        assert param.default == "value"
        assert param.required is True


class TestDatasetRequirements:
    """Test DatasetRequirements class."""

    def test_dataset_requirements_creation(self):
        """Test creating DatasetRequirements."""
        req = DatasetRequirements(
            required_columns=["text", "label"],
            optional_columns=["metadata"],
            min_samples=10,
            max_samples=1000,
            description="Test requirements"
        )
        assert req.required_columns == ["text", "label"]
        assert req.optional_columns == ["metadata"]
        assert req.min_samples == 10
        assert req.max_samples == 1000
        assert req.description == "Test requirements"

    def test_dataset_requirements_defaults(self):
        """Test default values."""
        req = DatasetRequirements()
        assert req.required_columns == []
        assert req.optional_columns == []
        assert req.min_samples == 1
        assert req.max_samples is None
        assert req.column_types == {}
        assert req.description == ""

    def test_dataset_requirements_validation(self):
        """Test dataset requirements validation."""
        # max_samples < min_samples should fail
        with pytest.raises(ValidationError):
            DatasetRequirements(min_samples=100, max_samples=50)

        # Valid ranges should pass
        req = DatasetRequirements(min_samples=10, max_samples=100)
        assert req.min_samples == 10
        assert req.max_samples == 100

    def test_column_name_validation(self):
        """Test column name validation."""
        req = DatasetRequirements(
            required_columns=["  text  ", "", "label", "   "],
            optional_columns=["metadata", "  ", "extra"]
        )
        # Empty and whitespace-only columns should be filtered out
        assert req.required_columns == ["text", "label"]
        assert req.optional_columns == ["metadata", "extra"]

    def test_validate_dataset(self):
        """Test dataset validation."""
        req = DatasetRequirements(
            required_columns=["text", "label"],
            min_samples=5
        )

        # Valid dataset
        errors = req.validate_dataset(["text", "label", "extra"], 10)
        assert errors == []

        # Missing required columns
        errors = req.validate_dataset(["text"], 10)
        assert len(errors) == 1
        assert "Missing required columns" in errors[0]

        # Too few samples
        errors = req.validate_dataset(["text", "label"], 3)
        assert len(errors) == 1
        assert "3 samples, minimum required: 5" in errors[0]

        # Multiple errors
        errors = req.validate_dataset(["text"], 3)
        assert len(errors) == 2


class TestFlowMetadata:
    """Test FlowMetadata class."""

    def test_flow_metadata_creation(self):
        """Test creating FlowMetadata."""
        models = [
            ModelOption(name="gpt-4o-mini", compatibility=ModelCompatibility.REQUIRED),
            ModelOption(name="gpt-3.5-turbo", compatibility=ModelCompatibility.RECOMMENDED)
        ]
        
        metadata = FlowMetadata(
            name="Test Flow",
            description="A test flow",
            version="1.0.0",
            author="Test Author",
            recommended_models=models,
            tags=["test", "example"],
            estimated_cost="low",
            estimated_duration="1-2 minutes"
        )
        
        assert metadata.name == "Test Flow"
        assert metadata.description == "A test flow"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert len(metadata.recommended_models) == 2
        assert metadata.tags == ["test", "example"]
        assert metadata.estimated_cost == "low"
        assert metadata.estimated_duration == "1-2 minutes"

    def test_flow_metadata_defaults(self):
        """Test default values."""
        metadata = FlowMetadata(name="Test Flow")
        assert metadata.description == ""
        assert metadata.version == "1.0.0"
        assert metadata.author == ""
        assert metadata.recommended_models == []
        assert metadata.tags == []
        assert metadata.license == "Apache-2.0"
        assert metadata.estimated_cost == "medium"
        assert metadata.estimated_duration == ""

    def test_flow_metadata_validation(self):
        """Test metadata validation."""
        # Empty name should fail
        with pytest.raises(ValidationError):
            FlowMetadata(name="")

        # Invalid version should fail
        with pytest.raises(ValidationError):
            FlowMetadata(name="Test", version="invalid")

        # Invalid cost should fail
        with pytest.raises(ValidationError):
            FlowMetadata(name="Test", estimated_cost="invalid")

    def test_tags_validation(self):
        """Test tags validation and cleaning."""
        metadata = FlowMetadata(
            name="Test Flow",
            tags=["  Test  ", "EXAMPLE", "", "   ", "demo"]
        )
        # Tags should be cleaned, lowercased, and empty ones removed
        assert metadata.tags == ["test", "example", "demo"]

    def test_recommended_models_validation(self):
        """Test recommended models validation."""
        # Duplicate models should fail
        with pytest.raises(ValidationError):
            FlowMetadata(
                name="Test Flow",
                recommended_models=[
                    ModelOption(name="gpt-4o-mini"),
                    ModelOption(name="gpt-4o-mini")  # Duplicate
                ]
            )

    def test_recommended_models_sorting(self):
        """Test recommended models are sorted by compatibility."""
        models = [
            ModelOption(name="model3", compatibility=ModelCompatibility.COMPATIBLE),
            ModelOption(name="model1", compatibility=ModelCompatibility.REQUIRED),
            ModelOption(name="model2", compatibility=ModelCompatibility.RECOMMENDED)
        ]
        
        metadata = FlowMetadata(name="Test Flow", recommended_models=models)
        
        # Should be sorted by compatibility priority
        assert metadata.recommended_models[0].name == "model1"  # REQUIRED
        assert metadata.recommended_models[1].name == "model2"  # RECOMMENDED
        assert metadata.recommended_models[2].name == "model3"  # COMPATIBLE

    def test_update_timestamp(self):
        """Test timestamp updating."""
        metadata = FlowMetadata(name="Test Flow")
        original_time = metadata.updated_at
        
        # Small delay to ensure different timestamp
        import time
        time.sleep(0.01)
        
        metadata.update_timestamp()
        assert metadata.updated_at != original_time

    def test_get_best_model(self):
        """Test getting the best model."""
        models = [
            ModelOption(name="gpt-4o", compatibility=ModelCompatibility.REQUIRED),
            ModelOption(name="gpt-4o-mini", compatibility=ModelCompatibility.RECOMMENDED),
            ModelOption(name="gpt-3.5-turbo", compatibility=ModelCompatibility.COMPATIBLE)
        ]
        
        metadata = FlowMetadata(name="Test Flow", recommended_models=models)
        
        # No availability list - should return first (highest priority)
        best = metadata.get_best_model()
        assert best.name == "gpt-4o"
        
        # With availability list - should return first available
        available = ["gpt-4o-mini", "gpt-3.5-turbo"]
        best = metadata.get_best_model(available)
        assert best.name == "gpt-4o-mini"
        
        # No compatible models available
        available = ["claude-3-haiku"]
        best = metadata.get_best_model(available)
        assert best is None
        
        # No recommended models
        empty_metadata = FlowMetadata(name="Empty Flow")
        best = empty_metadata.get_best_model()
        assert best is None

    def test_timestamps_auto_generation(self):
        """Test automatic timestamp generation."""
        metadata = FlowMetadata(name="Test Flow")
        
        # Should have created_at and updated_at
        assert metadata.created_at != ""
        assert metadata.updated_at != ""
        
        # Should be valid ISO format
        datetime.fromisoformat(metadata.created_at)
        datetime.fromisoformat(metadata.updated_at)

    def test_dataset_requirements_integration(self):
        """Test dataset requirements integration."""
        req = DatasetRequirements(
            required_columns=["text"],
            min_samples=5
        )
        
        metadata = FlowMetadata(
            name="Test Flow",
            dataset_requirements=req
        )
        
        assert metadata.dataset_requirements is not None
        assert metadata.dataset_requirements.required_columns == ["text"]
        assert metadata.dataset_requirements.min_samples == 5