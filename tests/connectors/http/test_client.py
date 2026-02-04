# SPDX-License-Identifier: Apache-2.0
"""Tests for HttpClient."""

from unittest.mock import AsyncMock, MagicMock, patch

from sdg_hub.core.connectors.exceptions import ConnectorError
from sdg_hub.core.connectors.http.client import HttpClient
import httpx
import pytest


class TestHttpClient:
    """Test HttpClient."""

    def test_init(self):
        """Test initialization with defaults and custom values."""
        client = HttpClient()
        assert client.timeout == 120.0
        assert client.max_retries == 3

        client = HttpClient(timeout=60.0, max_retries=5)
        assert client.timeout == 60.0
        assert client.max_retries == 5

    @pytest.mark.asyncio
    async def test_post_async(self):
        """Test async POST request and error handling."""
        client = HttpClient(max_retries=1)

        # Success case
        mock_response = httpx.Response(
            200,
            json={"result": "success"},
            request=httpx.Request("POST", "http://test.com"),
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            result = await client.post("http://test.com", {"data": "test"})
            assert result == {"result": "success"}

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

    def test_post_sync(self):
        """Test synchronous POST request and error handling."""
        client = HttpClient(max_retries=0)

        # Success case
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_class.return_value = mock_client

            result = client.post_sync("http://test.com", {"data": "test"})
            assert result == {"result": "ok"}

        # Connection error
        with patch("httpx.Client") as mock_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.ConnectError("refused")
            mock_class.return_value = mock_client
            with pytest.raises(ConnectorError, match="Failed to connect"):
                client.post_sync("http://test.com", {})
