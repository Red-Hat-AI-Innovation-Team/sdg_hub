# SPDX-License-Identifier: Apache-2.0
"""Tests for HttpClient."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from sdg_hub.core.connectors.exceptions import (
    ConnectorConnectionError,
    ConnectorHTTPError,
    ConnectorTimeoutError,
)
from sdg_hub.core.connectors.http.client import HttpClient


class TestHttpClient:
    """Test HttpClient."""

    def test_init_defaults(self):
        """Test default initialization."""
        client = HttpClient()
        assert client.timeout == 120.0
        assert client.max_retries == 3

    def test_init_custom_values(self):
        """Test custom initialization."""
        client = HttpClient(timeout=60.0, max_retries=5)
        assert client.timeout == 60.0
        assert client.max_retries == 5

    @pytest.mark.asyncio
    async def test_post_success(self):
        """Test successful POST request."""
        client = HttpClient()

        mock_response = httpx.Response(
            200,
            json={"result": "success"},
            request=httpx.Request("POST", "http://test.com"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await client.post(
                "http://test.com/api",
                {"data": "test"},
                {"Content-Type": "application/json"},
            )

            assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_post_timeout_error(self):
        """Test timeout error handling."""
        client = HttpClient(timeout=30.0, max_retries=1)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timed out")

            with pytest.raises(ConnectorTimeoutError) as exc_info:
                await client.post(
                    "http://test.com/api",
                    {"data": "test"},
                )

            assert "http://test.com/api" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_post_connection_error(self):
        """Test connection error handling."""
        client = HttpClient(max_retries=1)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(ConnectorConnectionError) as exc_info:
                await client.post(
                    "http://test.com/api",
                    {"data": "test"},
                )

            assert "http://test.com/api" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_post_http_error(self):
        """Test HTTP error handling."""
        client = HttpClient()

        mock_request = httpx.Request("POST", "http://test.com/api")
        mock_response = httpx.Response(
            500,
            text="Internal Server Error",
            request=mock_request,
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            mock_response.raise_for_status = lambda: (_ for _ in ()).throw(
                httpx.HTTPStatusError(
                    "Error", request=mock_request, response=mock_response
                )
            )

            with pytest.raises(ConnectorHTTPError) as exc_info:
                await client.post(
                    "http://test.com/api",
                    {"data": "test"},
                )

            assert exc_info.value.status_code == 500
            assert "http://test.com/api" in str(exc_info.value)

    def test_post_sync_success(self):
        """Test synchronous POST wrapper."""
        client = HttpClient()

        mock_response = httpx.Response(
            200,
            json={"result": "sync_success"},
            request=httpx.Request("POST", "http://test.com"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = client.post_sync(
                "http://test.com/api",
                {"data": "test"},
                {"Content-Type": "application/json"},
            )

            assert result == {"result": "sync_success"}

    def test_post_sync_with_default_headers(self):
        """Test POST with default (None) headers."""
        client = HttpClient()

        mock_response = httpx.Response(
            200,
            json={"result": "ok"},
            request=httpx.Request("POST", "http://test.com"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = client.post_sync("http://test.com/api", {"data": "test"})
            assert result == {"result": "ok"}
