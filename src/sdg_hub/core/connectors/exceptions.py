# SPDX-License-Identifier: Apache-2.0
"""Exception classes for the connectors subsystem."""

from typing import Optional

from ..utils.error_handling import SDGHubError


class ConnectorError(SDGHubError):
    """Base exception for all connector-related errors.

    Use this for general connector errors including:
    - Configuration errors
    - Connection failures
    - Timeout errors
    - Response parsing errors
    """

    pass


class ConnectorHTTPError(ConnectorError):
    """Raised when an HTTP request returns an error status code.

    Parameters
    ----------
    url : str
        The URL that returned an error.
    status_code : int
        The HTTP status code.
    message : str, optional
        Additional error details (e.g., response body).
    """

    def __init__(self, url: str, status_code: int, message: Optional[str] = None):
        self.url = url
        self.status_code = status_code
        error_msg = f"HTTP {status_code} error from '{url}'"
        if message:
            error_msg = f"{error_msg}: {message}"
        super().__init__(error_msg)
