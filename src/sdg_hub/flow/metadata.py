# SPDX-License-Identifier: Apache-2.0
"""Flow metadata and parameter definitions."""

# Standard
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Third Party
from pydantic import BaseModel, Field, field_validator, model_validator


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
    constraints: Dict[str, Any] = Field(
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

    required_columns: List[str] = Field(
        default_factory=list, description="Column names that must be present"
    )
    optional_columns: List[str] = Field(
        default_factory=list,
        description="Optional columns that can enhance performance",
    )
    min_samples: int = Field(
        default=1, ge=1, description="Minimum number of samples required"
    )
    max_samples: Optional[int] = Field(
        default=None, gt=0, description="Maximum number of samples to process"
    )
    column_types: Dict[str, str] = Field(
        default_factory=dict, description="Expected types for specific columns"
    )
    description: str = Field(default="", description="Human-readable description")

    @field_validator("required_columns", "optional_columns")
    @classmethod
    def validate_column_names(cls, v: List[str]) -> List[str]:
        """Validate column names are not empty."""
        return [col.strip() for col in v if col.strip()]

    @model_validator(mode="after")
    def validate_sample_limits(self) -> "DatasetRequirements":
        """Validate sample limits are consistent."""
        if self.max_samples is not None and self.max_samples < self.min_samples:
            raise ValueError("max_samples must be greater than or equal to min_samples")
        return self

    def validate_dataset(
        self, dataset_columns: List[str], dataset_size: int
    ) -> List[str]:
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
    name : str
        Human-readable name of the flow.
    description : str
        Detailed description of what the flow does.
    version : str
        Semantic version (e.g., "1.0.0").
    author : str
        Author or contributor name.
    recommended_models : List[ModelOption]
        Suggested LLM models with compatibility info.
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
    description: str = Field(default="", description="Detailed description")
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$",
        description="Semantic version",
    )
    author: str = Field(default="", description="Author or contributor name")
    recommended_models: List[ModelOption] = Field(
        default_factory=list, description="Suggested LLM models with compatibility info"
    )
    tags: List[str] = Field(
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

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate and clean tags."""
        return [tag.strip().lower() for tag in v if tag.strip()]

    @field_validator("recommended_models")
    @classmethod
    def validate_recommended_models(cls, v: List[ModelOption]) -> List[ModelOption]:
        """Validate recommended models list."""
        if not v:
            return v

        # Check for duplicates
        seen_names = set()
        for model in v:
            if model.name in seen_names:
                raise ValueError(f"Duplicate model name: {model.name}")
            seen_names.add(model.name)

        # Sort by compatibility priority
        priority_order = {
            ModelCompatibility.REQUIRED: 0,
            ModelCompatibility.RECOMMENDED: 1,
            ModelCompatibility.COMPATIBLE: 2,
            ModelCompatibility.EXPERIMENTAL: 3,
        }
        return sorted(v, key=lambda x: priority_order.get(x.compatibility, 999))

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()

    def get_best_model(
        self, available_models: Optional[List[str]] = None
    ) -> Optional[ModelOption]:
        """Get the best recommended model based on availability.

        Parameters
        ----------
        available_models : Optional[List[str]]
            List of available model names. If None, returns highest priority model.

        Returns
        -------
        Optional[ModelOption]
            Best model option or None if no models available.
        """
        if not self.recommended_models:
            return None

        if available_models is None:
            return self.recommended_models[0]

        # Find first model that's available
        for model in self.recommended_models:
            if model.name in available_models:
                return model

        return None
