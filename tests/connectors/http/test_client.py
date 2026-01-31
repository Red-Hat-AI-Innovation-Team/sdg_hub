# SPDX-License-Identifier: Apache-2.0
"""Tests for HttpClient."""

from unittest.mock import AsyncMock, patch

from sdg_hub.core.connectors.exceptions import ConnectorError, ConnectorHTTPError
from sdg_hub.core.connectors.http.client import HttpClient
import httpx
import pytest


class TestHttpClient:
    """Test HttpClient."""

    @pytest.mark.asyncio
    async def test_post_success(self):
        """Test successful POST request."""
        client = HttpClient(timeout=60.0, max_retries=3)

        mock_response = httpx.Response(
            200,
            json={"result": "success"},
            request=httpx.Request("POST", "http://test.com"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.post("http://test.com/api", {"data": "test"})
            assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_post_errors(self):
        """Test error handling for timeout, connection, and HTTP errors."""
        client = HttpClient(max_retries=1)

        # Timeout error
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock:
            mock.side_effect = httpx.TimeoutException("timeout")
            with pytest.raises(ConnectorError, match="timed out"):
                await client.post("http://test.com", {})

        # Connection error
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock:
            mock.side_effect = httpx.ConnectError("refused")
            with pytest.raises(ConnectorError, match="Failed to connect"):
                await client.post("http://test.com", {})

        # HTTP error
        mock_req = httpx.Request("POST", "http://test.com")
        mock_resp = httpx.Response(500, text="Error", request=mock_req)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock:
            mock.return_value = mock_resp
            mock_resp.raise_for_status = lambda: (_ for _ in ()).throw(
                httpx.HTTPStatusError("Error", request=mock_req, response=mock_resp)
            )
            with pytest.raises(ConnectorHTTPError) as exc:
                await client.post("http://test.com", {})
            assert exc.value.status_code == 500

    def test_post_sync(self):
        """Test synchronous POST wrapper."""
        client = HttpClient()
        mock_response = httpx.Response(
            200,
            json={"result": "ok"},
            request=httpx.Request("POST", "http://test.com"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            result = client.post_sync("http://test.com", {"data": "test"})
            assert result == {"result": "ok"}
