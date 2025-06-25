"""Custom exception classes for SDG Hub error handling."""


class SDGHubError(Exception):
    """Base exception class for all SDG Hub errors."""

    def __init__(self, message: str, details: str = None):
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


class FlowError(SDGHubError):
    """Base exception class for flow-related errors."""

    pass


class FlowValidationError(FlowError):
    """Raised when flow validation fails."""

    pass


class FlowExecutionError(FlowError):
    """Raised when flow execution fails."""

    pass
