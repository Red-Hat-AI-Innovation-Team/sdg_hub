# SPDX-License-Identifier: Apache-2.0
"""Exception classes for the connectors subsystem."""

from typing import Optional

from ..utils.error_handling import SDGHubError


class ConnectorError(SDGHubError):
    """Base exception for all connector-related errors."""

    pass


class ConnectorConfigError(ConnectorError):
    """Raised when connector configuration is invalid."""

    def __init__(self, connector_name: str, message: str):
        """Initialize ConnectorConfigError.

        Parameters
        ----------
        connector_name : str
            Name of the connector with invalid configuration.
        message : str
            Description of the configuration error.
        """
        self.connector_name = connector_name
        super().__init__(f"Connector '{connector_name}' configuration error: {message}")


class ConnectorConnectionError(ConnectorError):
    """Raised when a connection to an external service fails."""

    def __init__(self, url: str, message: Optional[str] = None):
        """Initialize ConnectorConnectionError.

        Parameters
        ----------
        url : str
            The URL that failed to connect.
        message : str, optional
            Additional error details.
        """
        self.url = url
        error_msg = f"Failed to connect to '{url}'"
        if message:
            error_msg = f"{error_msg}: {message}"
        super().__init__(error_msg)


class ConnectorTimeoutError(ConnectorError):
    """Raised when a request to an external service times out."""

    def __init__(self, url: str, timeout: Optional[float] = None):
        """Initialize ConnectorTimeoutError.

        Parameters
        ----------
        url : str
            The URL that timed out.
        timeout : float, optional
            The timeout value in seconds.
        """
        self.url = url
        self.timeout = timeout
        error_msg = f"Request to '{url}' timed out"
        if timeout is not None:
            error_msg = f"{error_msg} after {timeout}s"
        super().__init__(error_msg)


class ConnectorHTTPError(ConnectorError):
    """Raised when an HTTP request returns an error status code."""

    def __init__(self, url: str, status_code: int, message: Optional[str] = None):
        """Initialize ConnectorHTTPError.

        Parameters
        ----------
        url : str
            The URL that returned an error.
        status_code : int
            The HTTP status code.
        message : str, optional
            Additional error details (e.g., response body).
        """
        self.url = url
        self.status_code = status_code
        error_msg = f"HTTP {status_code} error from '{url}'"
        if message:
            error_msg = f"{error_msg}: {message}"
        super().__init__(error_msg)


class ConnectorResponseError(ConnectorError):
    """Raised when the response from an external service is invalid or unexpected."""

    def __init__(self, message: str, response: Optional[object] = None):
        """Initialize ConnectorResponseError.

        Parameters
        ----------
        message : str
            Description of what was wrong with the response.
        response : object, optional
            The actual response received (for debugging).
        """
        self.response = response
        super().__init__(f"Invalid response: {message}")
