# SPDX-License-Identifier: Apache-2.0
"""Tests for flow metadata models."""

# Standard
from datetime import datetime

# Third Party
from pydantic import ValidationError

# First Party
from sdg_hub import FlowMetadata, FlowParameter
from sdg_hub.core.flow.metadata import (
    DatasetRequirements,
    ModelCompatibility,
    ModelOption,
    RecommendedModels,
)
import pytest


class TestModelOption:
    """Test ModelOption class."""

    def test_model_option_creation(self):
        """Test creating a ModelOption."""
        model = ModelOption(
            name="gpt-4o-mini", compatibility=ModelCompatibility.REQUIRED
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


class TestRecommendedModels:
    """Test RecommendedModels class."""

    def test_recommended_models_creation(self):
        """Test creating RecommendedModels."""
        models = RecommendedModels(
            default="meta-llama/Llama-3.3-70B-Instruct",
            compatible=["microsoft/phi-4", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
            experimental=["experimental-model"],
        )
        assert models.default == "meta-llama/Llama-3.3-70B-Instruct"
        assert models.compatible == [
            "microsoft/phi-4",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ]
        assert models.experimental == ["experimental-model"]

    def test_recommended_models_defaults(self):
        """Test default values."""
        models = RecommendedModels(default="test-model")
        assert models.default == "test-model"
        assert models.compatible == []
        assert models.experimental == []

    def test_recommended_models_validation(self):
        """Test validation of model names."""
        # Empty default should fail
        with pytest.raises(ValidationError):
            RecommendedModels(default="")

        # Whitespace-only default should fail
        with pytest.raises(ValidationError):
            RecommendedModels(default="   ")

    def test_recommended_models_name_strip(self):
        """Test name stripping in all lists."""
        models = RecommendedModels(
            default="  default-model  ",
            compatible=["  model1  ", "  model2  "],
            experimental=["  exp-model  "],
        )
        assert models.default == "default-model"
        assert models.compatible == ["model1", "model2"]
        assert models.experimental == ["exp-model"]

    def test_recommended_models_empty_name_filtering(self):
        """Test filtering of empty names from lists."""
        models = RecommendedModels(
            default="default-model",
            compatible=["model1", "", "   ", "model2"],
            experimental=["", "exp-model", "   "],
        )
        assert models.compatible == ["model1", "model2"]
        assert models.experimental == ["exp-model"]

    def test_get_all_models(self):
        """Test get_all_models() method."""
        models = RecommendedModels(
            default="default-model",
            compatible=["compat1", "compat2"],
            experimental=["exp1"],
        )
        all_models = models.get_all_models()
        assert all_models == ["default-model", "compat1", "compat2", "exp1"]

    def test_get_best_model_no_available_list(self):
        """Test get_best_model() without available list."""
        models = RecommendedModels(
            default="default-model", compatible=["compat1"], experimental=["exp1"]
        )
        assert models.get_best_model() == "default-model"

    def test_get_best_model_with_available_list(self):
        """Test get_best_model() with availability checking."""
        models = RecommendedModels(
            default="default-model",
            compatible=["compat1", "compat2"],
            experimental=["exp1"],
        )

        # Default is available
        available = ["default-model", "compat1", "other-model"]
        assert models.get_best_model(available) == "default-model"

        # Only compatible available
        available = ["compat2", "other-model"]
        assert models.get_best_model(available) == "compat2"

        # Only experimental available
        available = ["exp1", "other-model"]
        assert models.get_best_model(available) == "exp1"

        # None available
        available = ["other-model", "another-model"]
        assert models.get_best_model(available) is None


class TestFlowParameter:
    """Test FlowParameter class."""

    def test_flow_parameter_creation(self):
        """Test creating a FlowParameter."""
        param = FlowParameter(
            default="test", description="Test parameter", type_hint="str", required=True
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
            description="Test requirements",
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
            optional_columns=["metadata", "  ", "extra"],
        )
        # Empty and whitespace-only columns should be filtered out
        assert req.required_columns == ["text", "label"]
        assert req.optional_columns == ["metadata", "extra"]

    def test_validate_dataset(self):
        """Test dataset validation."""
        req = DatasetRequirements(required_columns=["text", "label"], min_samples=5)

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
        models = RecommendedModels(
            default="meta-llama/Llama-3.3-70B-Instruct",
            compatible=["microsoft/phi-4", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
            experimental=["experimental-model"],
        )

        metadata = FlowMetadata(
            name="Test Flow",
            description="A test flow",
            version="1.0.0",
            author="Test Author",
            recommended_models=models,
            tags=["test", "example"],
            estimated_cost="low",
            estimated_duration="1-2 minutes",
        )

        assert metadata.name == "Test Flow"
        assert metadata.description == "A test flow"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert (
            metadata.recommended_models.default == "meta-llama/Llama-3.3-70B-Instruct"
        )
        assert metadata.tags == ["test", "example"]
        assert metadata.estimated_cost == "low"
        assert metadata.estimated_duration == "1-2 minutes"

    def test_flow_metadata_defaults(self):
        """Test default values."""
        metadata = FlowMetadata(name="Test Flow")
        assert metadata.description == ""
        assert metadata.version == "1.0.0"
        assert metadata.author == ""
        assert metadata.recommended_models is None
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
            name="Test Flow", tags=["  Test  ", "EXAMPLE", "", "   ", "demo"]
        )
        # Tags should be cleaned, lowercased, and empty ones removed
        assert metadata.tags == ["test", "example", "demo"]

    def test_recommended_models_validation(self):
        """Test recommended models validation."""
        # Valid models should pass
        models = RecommendedModels(
            default="test-model",
            compatible=["model1", "model2"],
            experimental=["exp-model"],
        )
        metadata = FlowMetadata(name="Test Flow", recommended_models=models)
        assert metadata.recommended_models.default == "test-model"

    def test_recommended_models_new_format(self):
        """Test new simplified recommended models format."""
        models = RecommendedModels(
            default="meta-llama/Llama-3.3-70B-Instruct",
            compatible=["microsoft/phi-4", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
            experimental=[],
        )

        metadata = FlowMetadata(name="Test Flow", recommended_models=models)

        # Should maintain the structure
        assert (
            metadata.recommended_models.default == "meta-llama/Llama-3.3-70B-Instruct"
        )
        assert metadata.recommended_models.compatible == [
            "microsoft/phi-4",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ]
        assert metadata.recommended_models.experimental == []

    def test_update_timestamp(self):
        """Test timestamp updating."""
        metadata = FlowMetadata(name="Test Flow")
        original_time = metadata.updated_at

        # Small delay to ensure different timestamp
        # Standard
        import time

        time.sleep(0.01)

        metadata.update_timestamp()
        assert metadata.updated_at != original_time

    def test_get_best_model(self):
        """Test getting the best model with new format."""
        models = RecommendedModels(
            default="meta-llama/Llama-3.3-70B-Instruct",
            compatible=["microsoft/phi-4", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
            experimental=["experimental-model"],
        )

        metadata = FlowMetadata(name="Test Flow", recommended_models=models)

        # No availability list - should return default
        best = metadata.get_best_model()
        assert best == "meta-llama/Llama-3.3-70B-Instruct"

        # With availability list - should return first available by priority
        available = ["microsoft/phi-4", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
        best = metadata.get_best_model(available)
        assert best == "microsoft/phi-4"

        # Only experimental available
        available = ["experimental-model"]
        best = metadata.get_best_model(available)
        assert best == "experimental-model"

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
        req = DatasetRequirements(required_columns=["text"], min_samples=5)

        metadata = FlowMetadata(name="Test Flow", dataset_requirements=req)

        assert metadata.dataset_requirements is not None
        assert metadata.dataset_requirements.required_columns == ["text"]
        assert metadata.dataset_requirements.min_samples == 5

    def test_id_generation(self):
        """Test automatic id generation from name."""
        metadata = FlowMetadata(name="My Complex Flow Name", description="Test flow")
        assert metadata.id is not None

    def test_id_validation(self):
        """Test id validation with random flow ids."""
        # Test custom id is preserved
        metadata = FlowMetadata(name="My Flow", id="custom-id", description="Test flow")
        assert metadata.id == "custom-id"

        # Test that a random id is generated if not provided
        metadata2 = FlowMetadata(name="Another Flow", description="Test flow")
        assert metadata2.id is not None
        assert isinstance(metadata2.id, str)
        assert metadata2.id != ""  # Should not be empty

        # Test invalid id characters (should fail lowercase check first)
        with pytest.raises(ValidationError) as exc_info:
            FlowMetadata(
                name="My Flow",
                id="Invalid ID!",  # Contains invalid characters and uppercase
                description="Test flow",
            )
        # The error message should mention lowercase requirement
        assert "id must be lowercase" in str(exc_info.value)

        # Test uppercase id
        with pytest.raises(ValidationError) as exc_info:
            FlowMetadata(name="My Flow", id="INVALID-CASE", description="Test flow")
        assert "id must be lowercase" in str(exc_info.value)

        # Test valid lowercase but invalid characters
        with pytest.raises(ValidationError) as exc_info:
            FlowMetadata(
                name="My Flow",
                id="invalid id!",  # Lowercase but contains space and exclamation
                description="Test flow",
            )
        assert "id must contain only alphanumeric characters and hyphens" in str(
            exc_info.value
        )
