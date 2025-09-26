# SPDX-License-Identifier: Apache-2.0
"""Flow metadata and parameter definitions."""

# Standard
from datetime import datetime
from enum import Enum
from typing import Any, Optional

# Third Party
from pydantic import BaseModel, Field, field_validator, model_validator

# Local
from ..utils.flow_identifier import get_flow_identifier


class ModelCompatibility(str, Enum):
    """Model compatibility levels."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    COMPATIBLE = "compatible"
    EXPERIMENTAL = "experimental"


class ModelOption(BaseModel):
    """Represents a model option with compatibility level.

    Attributes
    ----------
    name : str
        Model identifier (e.g., "gpt-4", "claude-3-sonnet")
    compatibility : ModelCompatibility
        Compatibility level with the flow
    """

    name: str = Field(..., description="Model identifier")
    compatibility: ModelCompatibility = Field(
        default=ModelCompatibility.COMPATIBLE,
        description="Compatibility level with the flow",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class RecommendedModels(BaseModel):
    """Simplified recommended models structure.

    Attributes
    ----------
    default : str
        The default model to use
    compatible : List[str]
        List of compatible models
    experimental : List[str]
        List of experimental models
    """

    default: str = Field(..., description="Default model to use")
    compatible: list[str] = Field(default_factory=list, description="Compatible models")
    experimental: list[str] = Field(
        default_factory=list, description="Experimental models"
    )

    @field_validator("default")
    @classmethod
    def validate_default(cls, v: str) -> str:
        """Validate default model name is not empty."""
        if not v.strip():
            raise ValueError("Default model name cannot be empty")
        return v.strip()

    @field_validator("compatible", "experimental")
    @classmethod
    def validate_model_lists(cls, v: list[str]) -> list[str]:
        """Validate model lists contain non-empty names."""
        return [model.strip() for model in v if model.strip()]

    def get_all_models(self) -> list[str]:
        """Get all models (default + compatible + experimental)."""
        return [self.default] + self.compatible + self.experimental

    def get_best_model(
        self, available_models: Optional[list[str]] = None
    ) -> Optional[str]:
        """Get the best model based on availability.

        Parameters
        ----------
        available_models : Optional[List[str]]
            List of available model names. If None, returns default.

        Returns
        -------
        Optional[str]
            Best model name or None if no models available.
        """
        if available_models is None:
            return self.default

        # Check in priority order: default, compatible, experimental
        if self.default in available_models:
            return self.default

        for model in self.compatible:
            if model in available_models:
                return model

        for model in self.experimental:
            if model in available_models:
                return model

        return None


class FlowParameter(BaseModel):
    """Represents a runtime parameter for a flow.

    Attributes
    ----------
    default : Any
        Default value for the parameter.
    description : str
        Human-readable description of the parameter.
    type_hint : str
        Type hint as string (e.g., "float", "str").
    required : bool
        Whether this parameter is required at runtime.
    constraints : Dict[str, Any]
        Additional constraints for the parameter.
    """

    default: Any = Field(..., description="Default value for the parameter")
    description: str = Field(default="", description="Human-readable description")
    type_hint: str = Field(default="Any", description="Type hint as string")
    required: bool = Field(default=False, description="Whether parameter is required")
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Additional constraints for the parameter"
    )

    @model_validator(mode="after")
    def validate_required_default(self) -> "FlowParameter":
        """Validate that required parameters have appropriate defaults."""
        if self.required and self.default is None:
            raise ValueError("Required parameters cannot have None as default")
        return self


class DatasetRequirements(BaseModel):
    """Dataset requirements for flow execution.

    Attributes
    ----------
    required_columns : List[str]
        Column names that must be present in the input dataset.
    optional_columns : List[str]
        Column names that are optional but can enhance flow performance.
    min_samples : int
        Minimum number of samples required in the dataset.
    max_samples : Optional[int]
        Maximum number of samples to process (for resource management).
    column_types : Dict[str, str]
        Expected types for specific columns.
    description : str
        Human-readable description of dataset requirements.
    """

    required_columns: list[str] = Field(
        default_factory=list, description="Column names that must be present"
    )
    optional_columns: list[str] = Field(
        default_factory=list,
        description="Optional columns that can enhance performance",
    )
    min_samples: int = Field(
        default=1, ge=1, description="Minimum number of samples required"
    )
    max_samples: Optional[int] = Field(
        default=None, gt=0, description="Maximum number of samples to process"
    )
    column_types: dict[str, str] = Field(
        default_factory=dict, description="Expected types for specific columns"
    )
    description: str = Field(default="", description="Human-readable description")

    @field_validator("required_columns", "optional_columns")
    @classmethod
    def validate_column_names(cls, v: list[str]) -> list[str]:
        """Validate column names are not empty."""
        return [col.strip() for col in v if col.strip()]

    @model_validator(mode="after")
    def validate_sample_limits(self) -> "DatasetRequirements":
        """Validate sample limits are consistent."""
        if self.max_samples is not None and self.max_samples < self.min_samples:
            raise ValueError("max_samples must be greater than or equal to min_samples")
        return self

    def validate_dataset(
        self, dataset_columns: list[str], dataset_size: int
    ) -> list[str]:
        """Validate a dataset against these requirements.

        Parameters
        ----------
        dataset_columns : List[str]
            Column names in the dataset.
        dataset_size : int
            Number of samples in the dataset.

        Returns
        -------
        List[str]
            List of validation error messages. Empty if valid.
        """
        errors = []

        # Check required columns
        missing_columns = [
            col for col in self.required_columns if col not in dataset_columns
        ]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Check minimum samples
        if dataset_size < self.min_samples:
            errors.append(
                f"Dataset has {dataset_size} samples, minimum required: {self.min_samples}"
            )

        return errors


class FlowMetadata(BaseModel):
    """Metadata for flow configuration and open source contributions.

    Attributes
    ----------
    id : str
        Unique identifier for the flow.
    name : str
        Human-readable name of the flow.
    description : str
        Detailed description of what the flow does.
    version : str
        Semantic version (e.g., "1.0.0").
    author : str
        Author or contributor name.
    recommended_models : Optional[RecommendedModels]
        Simplified recommended models structure with default, compatible, and experimental lists.
    tags : List[str]
        Tags for categorization and search.
    created_at : str
        Creation timestamp.
    updated_at : str
        Last update timestamp.
    license : str
        License identifier.
    min_sdg_hub_version : str
        Minimum required SDG Hub version.
    dataset_requirements : Optional[DatasetRequirements]
        Requirements for input datasets.
    estimated_cost : str
        Estimated cost tier for running the flow.
    estimated_duration : str
        Estimated duration for flow execution.
    """

    name: str = Field(..., min_length=1, description="Human-readable name")
    id: str = Field(
        default="", description="Unique identifier for the flow, generated from name"
    )
    description: str = Field(default="", description="Detailed description")
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$",
        description="Semantic version",
    )
    author: str = Field(default="", description="Author or contributor name")
    recommended_models: Optional[RecommendedModels] = Field(
        default=None, description="Simplified recommended models structure"
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags for categorization and search"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Creation timestamp",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Last update timestamp",
    )
    license: str = Field(default="Apache-2.0", description="License identifier")
    min_sdg_hub_version: str = Field(
        default="", description="Minimum required SDG Hub version"
    )
    dataset_requirements: Optional[DatasetRequirements] = Field(
        default=None, description="Requirements for input datasets"
    )
    estimated_cost: str = Field(
        default="medium",
        pattern="^(low|medium|high)$",
        description="Estimated cost tier for running the flow",
    )
    estimated_duration: str = Field(
        default="", description="Estimated duration for flow execution"
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate flow id."""
        # Note: Auto-generation is handled in the model_validator since field_validator
        # doesn't have access to other field values in Pydantic v2

        # Validate id format if provided
        if v:
            # Must be lowercase
            if not v.islower():
                raise ValueError("id must be lowercase")

            # Must contain only alphanumeric characters and hyphens
            if not v.replace("-", "").isalnum():
                raise ValueError(
                    "id must contain only alphanumeric characters and hyphens"
                )

            # Must not start or end with a hyphen
            if v.startswith("-") or v.endswith("-"):
                raise ValueError("id must not start or end with a hyphen")

        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate and clean tags."""
        return [tag.strip().lower() for tag in v if tag.strip()]

    @field_validator("recommended_models")
    @classmethod
    def validate_recommended_models(
        cls, v: Optional[RecommendedModels]
    ) -> Optional[RecommendedModels]:
        """Validate recommended models structure."""
        # Validation is handled within RecommendedModels class
        return v

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()

    @model_validator(mode="after")
    def ensure_id(self) -> "FlowMetadata":
        """Ensure id is set.

        Note: YAML persistence is handled by Flow.from_yaml() and FlowRegistry
        to maintain proper separation of concerns.
        """
        if not self.id and self.name:
            self.id = get_flow_identifier(self.name)

        return self

    def get_best_model(
        self, available_models: Optional[list[str]] = None
    ) -> Optional[str]:
        """Get the best recommended model based on availability.

        Parameters
        ----------
        available_models : Optional[List[str]]
            List of available model names. If None, returns default model.

        Returns
        -------
        Optional[str]
            Best model name or None if no models available.
        """
        if not self.recommended_models:
            return None

        return self.recommended_models.get_best_model(available_models)
