# SPDX-License-Identifier: Apache-2.0
"""Tests for connector exceptions."""

from sdg_hub.core.connectors.exceptions import ConnectorError, ConnectorHTTPError
from sdg_hub.core.utils.error_handling import SDGHubError


class TestConnectorExceptions:
    """Test connector exceptions."""

    def test_connector_error(self):
        """Test ConnectorError inherits from SDGHubError."""
        assert issubclass(ConnectorError, SDGHubError)
        error = ConnectorError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_http_error(self):
        """Test ConnectorHTTPError captures status code and URL."""
        # With message
        error = ConnectorHTTPError("http://localhost:7860", 500, "Server error")
        assert error.status_code == 500
        assert error.url == "http://localhost:7860"
        assert "HTTP 500" in str(error)
        assert "Server error" in str(error)
        assert issubclass(ConnectorHTTPError, ConnectorError)

        # Without message
        error = ConnectorHTTPError("http://localhost:7860", 404)
        assert "HTTP 404" in str(error)
