"""Tests for custom error handling classes."""

# First Party
# Third Party
from sdg_hub.core.utils.error_handling import (
    APIConnectionError,
    BlockConfigurationError,
    BlockError,
    BlockExecutionError,
    DataGenerationError,
    DataSaveError,
    DatasetLoadError,
    FlowConfigurationError,
    FlowError,
    FlowExecutionError,
    FlowRunnerError,
    FlowValidationError,
    SDGHubError,
)
import pytest


class TestSDGHubError:
    """Test cases for the base SDGHubError class."""

    def test_basic_initialization(self):
        """Test basic error initialization with just a message."""
        error = SDGHubError("Test error message")
        assert error.message == "Test error message"
        assert error.details is None
        assert str(error) == "Test error message"

    def test_initialization_with_details(self):
        """Test error initialization with message and details."""
        error = SDGHubError("Test error message", "Additional details")
        assert error.message == "Test error message"
        assert error.details == "Additional details"
        assert str(error) == "Test error message\nDetails: Additional details"

    def test_initialization_with_none_details(self):
        """Test error initialization with None details."""
        error = SDGHubError("Test error message", None)
        assert error.message == "Test error message"
        assert error.details is None
        assert str(error) == "Test error message"

    def test_initialization_with_empty_details(self):
        """Test error initialization with empty string details."""
        error = SDGHubError("Test error message", "")
        assert error.message == "Test error message"
        assert error.details == ""
        assert str(error) == "Test error message"

    def test_inheritance_from_exception(self):
        """Test that SDGHubError inherits from Exception."""
        error = SDGHubError("Test message")
        assert isinstance(error, Exception)


class TestFlowRunnerErrors:
    """Test cases for FlowRunner-specific error classes."""

    def test_flow_runner_error_inheritance(self):
        """Test FlowRunnerError inherits from SDGHubError."""
        error = FlowRunnerError("Test message")
        assert isinstance(error, SDGHubError)
        assert isinstance(error, Exception)

    def test_dataset_load_error(self):
        """Test DatasetLoadError functionality."""
        error = DatasetLoadError("Failed to load dataset", "File not found")
        assert isinstance(error, FlowRunnerError)
        assert error.message == "Failed to load dataset"
        assert error.details == "File not found"

    def test_flow_configuration_error(self):
        """Test FlowConfigurationError functionality."""
        error = FlowConfigurationError("Invalid flow config", "YAML syntax error")
        assert isinstance(error, FlowRunnerError)
        assert error.message == "Invalid flow config"
        assert error.details == "YAML syntax error"

    def test_api_connection_error(self):
        """Test APIConnectionError functionality."""
        error = APIConnectionError("Connection failed", "Timeout after 30s")
        assert isinstance(error, FlowRunnerError)
        assert error.message == "Connection failed"
        assert error.details == "Timeout after 30s"

    def test_data_generation_error(self):
        """Test DataGenerationError functionality."""
        error = DataGenerationError("Generation failed", "Block execution error")
        assert isinstance(error, FlowRunnerError)
        assert error.message == "Generation failed"
        assert error.details == "Block execution error"

    def test_data_save_error(self):
        """Test DataSaveError functionality."""
        error = DataSaveError("Save failed", "Permission denied")
        assert isinstance(error, FlowRunnerError)
        assert error.message == "Save failed"
        assert error.details == "Permission denied"


class TestBlockErrors:
    """Test cases for Block-specific error classes."""

    def test_block_error_inheritance(self):
        """Test BlockError inherits from SDGHubError."""
        error = BlockError("Test message")
        assert isinstance(error, SDGHubError)
        assert isinstance(error, Exception)

    def test_block_configuration_error(self):
        """Test BlockConfigurationError functionality."""
        error = BlockConfigurationError(
            "Invalid block config", "Missing required field"
        )
        assert isinstance(error, BlockError)
        assert error.message == "Invalid block config"
        assert error.details == "Missing required field"

    def test_block_execution_error(self):
        """Test BlockExecutionError functionality."""
        error = BlockExecutionError("Block execution failed", "API timeout")
        assert isinstance(error, BlockError)
        assert error.message == "Block execution failed"
        assert error.details == "API timeout"


class TestFlowErrors:
    """Test cases for Flow-specific error classes."""

    def test_flow_error_inheritance(self):
        """Test FlowError inherits from SDGHubError."""
        error = FlowError("Test message")
        assert isinstance(error, SDGHubError)
        assert isinstance(error, Exception)

    def test_flow_validation_error(self):
        """Test FlowValidationError functionality."""
        error = FlowValidationError("Flow validation failed", "Invalid block sequence")
        assert isinstance(error, FlowError)
        assert error.message == "Flow validation failed"
        assert error.details == "Invalid block sequence"

    def test_flow_execution_error(self):
        """Test FlowExecutionError functionality."""
        error = FlowExecutionError("Flow execution failed", "Block dependency error")
        assert isinstance(error, FlowError)
        assert error.message == "Flow execution failed"
        assert error.details == "Block dependency error"


class TestErrorRaising:
    """Test that errors can be properly raised and caught."""

    def test_raise_and_catch_sdghub_error(self):
        """Test raising and catching SDGHubError."""
        with pytest.raises(SDGHubError) as exc_info:
            raise SDGHubError("Test error", "Test details")

        assert exc_info.value.message == "Test error"
        assert exc_info.value.details == "Test details"

    def test_raise_and_catch_flow_runner_error(self):
        """Test raising and catching FlowRunnerError."""
        with pytest.raises(FlowRunnerError) as exc_info:
            raise FlowRunnerError("Flow runner error")

        assert exc_info.value.message == "Flow runner error"
        assert exc_info.value.details is None

    def test_catch_derived_error_as_base(self):
        """Test catching derived error as base class."""
        with pytest.raises(SDGHubError):
            raise DatasetLoadError("Dataset load failed")

    def test_catch_flow_runner_error_as_base(self):
        """Test catching FlowRunner errors as SDGHubError."""
        with pytest.raises(SDGHubError):
            raise APIConnectionError("API error")


class TestErrorChaining:
    """Test error chaining functionality with from clause."""

    def test_error_chaining_with_cause(self):
        """Test error chaining preserves the original exception."""
        original_error = ValueError("Original error")

        try:
            try:
                raise original_error
            except ValueError as e:
                raise DatasetLoadError(
                    "Dataset load failed", "ValueError occurred"
                ) from e
        except DatasetLoadError as sdg_error:
            assert sdg_error.__cause__ is original_error
            assert sdg_error.message == "Dataset load failed"
            assert sdg_error.details == "ValueError occurred"

    def test_multiple_error_levels(self):
        """Test multiple levels of error chaining."""
        try:
            try:
                try:
                    raise FileNotFoundError("File not found")
                except FileNotFoundError as e:
                    raise DatasetLoadError("Failed to load", str(e)) from e
            except DatasetLoadError as e:
                raise FlowRunnerError("Flow execution failed", str(e)) from e
        except FlowRunnerError as final_error:
            assert isinstance(final_error.__cause__, DatasetLoadError)
            assert isinstance(final_error.__cause__.__cause__, FileNotFoundError)


class TestErrorMessages:
    """Test error message formatting and content."""

    def test_error_message_formatting(self):
        """Test that error messages are formatted correctly."""
        error = SDGHubError("Main message", "Detailed information")
        expected = "Main message\nDetails: Detailed information"
        assert str(error) == expected

    def test_multiline_details(self):
        """Test error with multiline details."""
        details = "Line 1\nLine 2\nLine 3"
        error = SDGHubError("Main message", details)
        expected = f"Main message\nDetails: {details}"
        assert str(error) == expected

    def test_error_repr(self):
        """Test error representation."""
        error = SDGHubError("Test message", "Test details")
        # The repr should contain the class name and message
        repr_str = repr(error)
        assert "SDGHubError" in repr_str
        assert "Test message" in repr_str
