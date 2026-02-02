# SPDX-License-Identifier: Apache-2.0
"""HTTP client with tenacity retry."""

from typing import Any, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import httpx

from ...utils.logger_config import setup_logger
from ..exceptions import ConnectorError, ConnectorHTTPError

logger = setup_logger(__name__)


class HttpClient:
    """HTTP client with tenacity retry.

    Parameters
    ----------
    timeout : float
        Request timeout in seconds. Default is 120.0.
    max_retries : int
        Maximum number of retry attempts. Default is 3.

    Example
    -------
    >>> client = HttpClient(timeout=60.0, max_retries=3)
    >>> response = await client.post("https://api.example.com", {"key": "value"}, {})
    """

    def __init__(self, timeout: float = 120.0, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries

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
        ConnectorError
            If connection or timeout fails after all retries.
        ConnectorHTTPError
            If an HTTP error status is returned.
        """
        headers = headers or {}

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),  # 1 initial + retries
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
            reraise=True,
        )
        async def _post_with_retry() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.debug(f"POST request to {url}")
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()

        try:
            return await _post_with_retry()
        except httpx.HTTPStatusError as e:
            response_text = e.response.text[:500] if e.response.text else None
            raise ConnectorHTTPError(url, e.response.status_code, response_text) from e
        except httpx.TimeoutException as e:
            raise ConnectorError(
                f"Request to '{url}' timed out after {self.timeout}s"
            ) from e
        except httpx.ConnectError as e:
            raise ConnectorError(f"Failed to connect to '{url}': {e}") from e

    def post_sync(
        self,
        url: str,
        payload: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Synchronous POST request with retry logic.

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
        ConnectorError
            If connection or timeout fails after all retries.
        ConnectorHTTPError
            If an HTTP error status is returned.
        """
        headers = headers or {}

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),  # 1 initial + retries
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
            reraise=True,
        )
        def _post_with_retry() -> dict[str, Any]:
            with httpx.Client(timeout=self.timeout) as client:
                logger.debug(f"POST request to {url}")
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()

        try:
            return _post_with_retry()
        except httpx.HTTPStatusError as e:
            response_text = e.response.text[:500] if e.response.text else None
            raise ConnectorHTTPError(url, e.response.status_code, response_text) from e
        except httpx.TimeoutException as e:
            raise ConnectorError(
                f"Request to '{url}' timed out after {self.timeout}s"
            ) from e
        except httpx.ConnectError as e:
            raise ConnectorError(f"Failed to connect to '{url}': {e}") from e
