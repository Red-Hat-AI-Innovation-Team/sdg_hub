# SPDX-License-Identifier: Apache-2.0
"""Flow metadata and parameter definitions."""

# Standard
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional


@dataclass
class FlowParameter:
    """Represents a runtime parameter for a flow.

    Parameters
    ----------
    default : Any
        Default value for the parameter.
    description : str, optional
        Human-readable description of the parameter.
    type_hint : str, optional
        Type hint as string (e.g., "float", "str").
    required : bool, optional
        Whether this parameter is required at runtime.
    """

    default: Any
    description: str = ""
    type_hint: str = "Any"
    required: bool = False

    def __post_init__(self) -> None:
        """Validate parameter configuration."""
        if self.required and self.default is None:
            raise ValueError("Required parameters cannot have None as default")


@dataclass
class DatasetRequirements:
    """Dataset requirements for flow execution.

    Parameters
    ----------
    required_columns : List[str], optional
        Column names that must be present in the input dataset.
    optional_columns : List[str], optional
        Column names that are optional but can enhance flow performance.
    min_samples : int, optional
        Minimum number of samples required in the dataset.
    column_types : dict, optional
        Expected types for specific columns (e.g., {"text": "string", "score": "float"}).
    description : str, optional
        Human-readable description of dataset requirements.
    """

    required_columns: List[str] = field(default_factory=list)
    optional_columns: List[str] = field(default_factory=list)
    min_samples: int = 1
    column_types: dict = field(default_factory=dict)
    description: str = ""

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

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        return {
            "required_columns": self.required_columns,
            "optional_columns": self.optional_columns,
            "min_samples": self.min_samples,
            "column_types": self.column_types,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetRequirements":
        """Create from dictionary."""
        return cls(**data)


@dataclass  # pylint: disable=too-many-instance-attributes
class FlowMetadata:
    """Metadata for flow configuration and open source contributions.

    Parameters
    ----------
    name : str
        Human-readable name of the flow.
    description : str, optional
        Detailed description of what the flow does.
    version : str, optional
        Semantic version (e.g., "1.0.0").
    author : str, optional
        Author or contributor name.
    recommended_model : str, optional
        Suggested LLM model for optimal performance.
    tags : List[str], optional
        Tags for categorization and search.
    created_at : str, optional
        Creation timestamp.
    updated_at : str, optional
        Last update timestamp.
    license : str, optional
        License identifier.
    min_sdg_hub_version : str, optional
        Minimum required SDG Hub version.
    dataset_requirements : DatasetRequirements, optional
        Requirements for input datasets.
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    recommended_model: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    license: str = "Apache-2.0"
    min_sdg_hub_version: str = ""
    dataset_requirements: Optional[DatasetRequirements] = None

    def __post_init__(self) -> None:
        """Set timestamps if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for YAML serialization."""
        result = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "recommended_model": self.recommended_model,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "license": self.license,
            "min_sdg_hub_version": self.min_sdg_hub_version,
        }

        if self.dataset_requirements:
            result["dataset_requirements"] = self.dataset_requirements.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "FlowMetadata":
        """Create metadata from dictionary."""
        # Handle dataset_requirements separately
        data_copy = data.copy()
        dataset_requirements = None

        if "dataset_requirements" in data_copy:
            req_data = data_copy.pop("dataset_requirements")
            if req_data:
                dataset_requirements = DatasetRequirements.from_dict(req_data)

        return cls(dataset_requirements=dataset_requirements, **data_copy)
