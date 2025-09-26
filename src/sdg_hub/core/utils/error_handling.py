"""Custom exception classes for SDG Hub error handling."""

# Standard
from typing import Optional


class SDGHubError(Exception):
    """Base exception class for all SDG Hub errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        """Initialize SDGHubError.

        Parameters
        ----------
        message : str
            The main error message.
        details : str, optional
            Additional details about the error.
        """
        self.message = message
        self.details = details
        full_message = message
        if details:
            full_message = f"{message}\nDetails: {details}"
        super().__init__(full_message)


class FlowRunnerError(SDGHubError):
    """Base exception class for flow runner errors."""

    pass


class DatasetLoadError(FlowRunnerError):
    """Raised when dataset loading fails."""

    pass


class FlowConfigurationError(FlowRunnerError):
    """Raised when flow configuration is invalid."""

    pass


class APIConnectionError(FlowRunnerError):
    """Raised when API connection fails."""

    pass


class DataGenerationError(FlowRunnerError):
    """Raised when data generation fails."""

    pass


class DataSaveError(FlowRunnerError):
    """Raised when saving generated data fails."""

    pass


class BlockError(SDGHubError):
    """Base exception class for block-related errors."""

    pass


class BlockConfigurationError(BlockError):
    """Raised when block configuration is invalid."""

    pass


class BlockExecutionError(BlockError):
    """Raised when block execution fails."""

    pass


class BlockValidationError(BlockError):
    """Base exception class for block validation errors."""

    pass


class MissingColumnError(BlockValidationError):
    """Raised when required input columns are missing from dataset."""

    def __init__(
        self, block_name: str, missing_columns: list[str], available_columns: list[str]
    ):
        """Initialize MissingColumnError.

        Parameters
        ----------
        block_name : str
            Name of the block that failed validation.
        missing_columns : List[str]
            List of missing column names.
        available_columns : List[str]
            List of available column names in the dataset.
        """
        self.block_name = block_name
        self.missing_columns = missing_columns
        self.available_columns = available_columns

        message = (
            f"Block '{block_name}' missing required input columns: {missing_columns}"
        )
        details = f"Available columns: {available_columns}"

        super().__init__(message, details)


class EmptyDatasetError(BlockValidationError):
    """Raised when an empty dataset is provided to a block."""

    def __init__(self, block_name: str):
        """Initialize EmptyDatasetError.

        Parameters
        ----------
        block_name : str
            Name of the block that received the empty dataset.
        """
        self.block_name = block_name

        message = f"Block '{block_name}' received an empty dataset"
        details = "Dataset must contain at least one sample for processing"

        super().__init__(message, details)


class OutputColumnCollisionError(BlockValidationError):
    """Raised when output columns would overwrite existing dataset columns."""

    def __init__(
        self, block_name: str, collision_columns: list[str], existing_columns: list[str]
    ):
        """Initialize OutputColumnCollisionError.

        Parameters
        ----------
        block_name : str
            Name of the block that has column collisions.
        collision_columns : List[str]
            List of output columns that collide with existing columns.
        existing_columns : List[str]
            List of existing column names in the dataset.
        """
        self.block_name = block_name
        self.collision_columns = collision_columns
        self.existing_columns = existing_columns

        message = f"Block '{block_name}' output columns would overwrite existing data: {collision_columns}"
        details = f"Existing columns: {existing_columns}"

        super().__init__(message, details)


class TemplateValidationError(BlockValidationError):
    """Raised when template validation fails due to missing variables."""

    def __init__(
        self,
        block_name: str,
        missing_variables: list[str],
        available_variables: list[str],
    ):
        """Initialize TemplateValidationError.

        Parameters
        ----------
        block_name : str
            Name of the block that failed template validation.
        missing_variables : List[str]
            List of missing template variable names.
        available_variables : List[str]
            List of available template variable names.
        """
        self.block_name = block_name
        self.missing_variables = missing_variables
        self.available_variables = available_variables

        message = f"Block '{block_name}' template validation failed - missing required variables: {missing_variables}"
        details = f"Available variables: {available_variables}"

        super().__init__(message, details)


class FlowError(SDGHubError):
    """Base exception class for flow-related errors."""

    pass


class FlowValidationError(FlowError):
    """Raised when flow validation fails."""

    pass


class FlowExecutionError(FlowError):
    """Raised when flow execution fails."""

    pass
