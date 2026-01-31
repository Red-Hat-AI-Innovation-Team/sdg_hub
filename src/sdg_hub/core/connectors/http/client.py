# SPDX-License-Identifier: Apache-2.0
"""Unified HTTP client with async-first pattern and tenacity retry."""

from typing import Any, Optional
import asyncio

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import httpx

from ...utils.logger_config import setup_logger
from ..exceptions import (
    ConnectorConnectionError,
    ConnectorHTTPError,
    ConnectorTimeoutError,
)

logger = setup_logger(__name__)


class HttpClient:
    """Unified HTTP client - async-first with tenacity retry.

    This client provides both sync and async HTTP methods with automatic
    retry logic using exponential backoff for transient failures.

    Parameters
    ----------
    timeout : float
        Request timeout in seconds. Default is 120.0.
    max_retries : int
        Maximum number of retry attempts. Default is 3.

    Example
    -------
    >>> client = HttpClient(timeout=60.0, max_retries=3)
    >>> # Async usage
    >>> response = await client.post("https://api.example.com", {"key": "value"}, {})
    >>> # Sync usage
    >>> response = client.post_sync("https://api.example.com", {"key": "value"}, {})
    """

    def __init__(self, timeout: float = 120.0, max_retries: int = 3):
        """Initialize the HTTP client.

        Parameters
        ----------
        timeout : float
            Request timeout in seconds.
        max_retries : int
            Maximum number of retry attempts.
        """
        self.timeout = timeout
        self.max_retries = max_retries

    def _create_retry_decorator(self):
        """Create a retry decorator with current settings."""
        return retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
            reraise=True,
        )

    def _handle_error(self, e: Exception, url: str) -> None:
        """Convert httpx exceptions to connector exceptions.

        Parameters
        ----------
        e : Exception
            The exception to handle.
        url : str
            The URL that caused the error.

        Raises
        ------
        ConnectorTimeoutError
            If the request timed out.
        ConnectorHTTPError
            If an HTTP error occurred.
        ConnectorConnectionError
            For all other connection errors.
        """
        if isinstance(e, httpx.TimeoutException):
            raise ConnectorTimeoutError(url, self.timeout) from e
        elif isinstance(e, httpx.HTTPStatusError):
            response_text = e.response.text[:500] if e.response.text else None
            raise ConnectorHTTPError(url, e.response.status_code, response_text) from e
        elif isinstance(e, httpx.ConnectError):
            raise ConnectorConnectionError(url, str(e)) from e
        else:
            raise ConnectorConnectionError(url, str(e)) from e

    async def _post_async_impl(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Internal async POST implementation.

        Parameters
        ----------
        url : str
            The URL to POST to.
        payload : dict
            The JSON payload to send.
        headers : dict
            HTTP headers to include.

        Returns
        -------
        dict
            The JSON response.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.debug(f"POST request to {url}")
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except (httpx.TimeoutException, httpx.ConnectError):
            # Let these propagate for tenacity to retry
            raise
        except httpx.HTTPStatusError as e:
            # Don't retry HTTP errors - convert and raise immediately
            self._handle_error(e, url)
            raise  # Unreachable but keeps type checker happy
        except Exception as e:
            self._handle_error(e, url)
            raise  # Unreachable but keeps type checker happy

    async def post(
        self,
        url: str,
        payload: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Async POST request with retry logic.

        Parameters
        ----------
        url : str
            The URL to POST to.
        payload : dict
            The JSON payload to send.
        headers : dict, optional
            HTTP headers to include.

        Returns
        -------
        dict
            The JSON response.

        Raises
        ------
        ConnectorTimeoutError
            If all retry attempts time out.
        ConnectorConnectionError
            If connection fails after all retries.
        ConnectorHTTPError
            If an HTTP error status is returned.
        """
        headers = headers or {}
        retry_decorator = self._create_retry_decorator()
        retryable_post = retry_decorator(self._post_async_impl)

        try:
            return await retryable_post(url, payload, headers)
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            # Convert after all retries exhausted
            self._handle_error(e, url)
            raise  # Unreachable

    def post_sync(
        self,
        url: str,
        payload: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Synchronous POST request - wraps async implementation.

        Parameters
        ----------
        url : str
            The URL to POST to.
        payload : dict
            The JSON payload to send.
        headers : dict, optional
            HTTP headers to include.

        Returns
        -------
        dict
            The JSON response.

        Raises
        ------
        ConnectorTimeoutError
            If all retry attempts time out.
        ConnectorConnectionError
            If connection fails after all retries.
        ConnectorHTTPError
            If an HTTP error status is returned.
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # We're in an async context - need to use thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.post(url, payload, headers))
                return future.result()
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(self.post(url, payload, headers))
