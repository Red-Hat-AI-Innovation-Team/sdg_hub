# SPDX-License-Identifier: Apache-2.0
"""Tests for connector exceptions."""


from sdg_hub.core.connectors.exceptions import (
    ConnectorConfigError,
    ConnectorConnectionError,
    ConnectorError,
    ConnectorHTTPError,
    ConnectorResponseError,
    ConnectorTimeoutError,
)
from sdg_hub.core.utils.error_handling import SDGHubError


class TestConnectorError:
    """Test base ConnectorError."""

    def test_inherits_from_sdghub_error(self):
        """Test that ConnectorError inherits from SDGHubError."""
        assert issubclass(ConnectorError, SDGHubError)

    def test_basic_error(self):
        """Test creating a basic connector error."""
        error = ConnectorError("Something went wrong")
        assert str(error) == "Something went wrong"


class TestConnectorConfigError:
    """Test ConnectorConfigError."""

    def test_error_message_format(self):
        """Test error message includes connector name."""
        error = ConnectorConfigError("langflow", "missing url")
        assert "langflow" in str(error)
        assert "configuration error" in str(error)
        assert "missing url" in str(error)

    def test_connector_name_attribute(self):
        """Test connector_name attribute is set."""
        error = ConnectorConfigError("my_connector", "invalid api key")
        assert error.connector_name == "my_connector"


class TestConnectorConnectionError:
    """Test ConnectorConnectionError."""

    def test_error_message_with_url(self):
        """Test error message includes URL."""
        error = ConnectorConnectionError("http://localhost:7860")
        assert "http://localhost:7860" in str(error)
        assert "Failed to connect" in str(error)

    def test_error_message_with_details(self):
        """Test error message includes additional details."""
        error = ConnectorConnectionError("http://localhost:7860", "Connection refused")
        assert "Connection refused" in str(error)

    def test_url_attribute(self):
        """Test url attribute is set."""
        error = ConnectorConnectionError("http://api.example.com")
        assert error.url == "http://api.example.com"


class TestConnectorTimeoutError:
    """Test ConnectorTimeoutError."""

    def test_error_message_basic(self):
        """Test basic timeout error message."""
        error = ConnectorTimeoutError("http://localhost:7860")
        assert "http://localhost:7860" in str(error)
        assert "timed out" in str(error)

    def test_error_message_with_timeout_value(self):
        """Test timeout error includes timeout value."""
        error = ConnectorTimeoutError("http://localhost:7860", 30.0)
        assert "30.0s" in str(error)

    def test_attributes(self):
        """Test url and timeout attributes are set."""
        error = ConnectorTimeoutError("http://api.example.com", 60.0)
        assert error.url == "http://api.example.com"
        assert error.timeout == 60.0


class TestConnectorHTTPError:
    """Test ConnectorHTTPError."""

    def test_error_message_format(self):
        """Test HTTP error message format."""
        error = ConnectorHTTPError("http://localhost:7860", 404)
        assert "HTTP 404" in str(error)
        assert "http://localhost:7860" in str(error)

    def test_error_message_with_body(self):
        """Test HTTP error with response body."""
        error = ConnectorHTTPError(
            "http://localhost:7860", 500, "Internal server error"
        )
        assert "Internal server error" in str(error)

    def test_attributes(self):
        """Test url and status_code attributes are set."""
        error = ConnectorHTTPError("http://api.example.com", 401)
        assert error.url == "http://api.example.com"
        assert error.status_code == 401


class TestConnectorResponseError:
    """Test ConnectorResponseError."""

    def test_error_message_format(self):
        """Test response error message format."""
        error = ConnectorResponseError("Expected dict, got list")
        assert "Invalid response" in str(error)
        assert "Expected dict, got list" in str(error)

    def test_with_response_object(self):
        """Test error with response object for debugging."""
        bad_response = ["unexpected", "list"]
        error = ConnectorResponseError("Expected dict", response=bad_response)
        assert error.response == bad_response
