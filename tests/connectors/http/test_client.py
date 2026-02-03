# SPDX-License-Identifier: Apache-2.0
"""Tests for HttpClient."""

from unittest.mock import AsyncMock, MagicMock, patch

from sdg_hub.core.connectors.exceptions import ConnectorError, ConnectorHTTPError
from sdg_hub.core.connectors.http.client import HttpClient
import httpx
import pytest


class TestHttpClientInit:
    """Test HttpClient initialization."""

    def test_default_values(self):
        """Test default timeout and max_retries."""
        client = HttpClient()
        assert client.timeout == 120.0
        assert client.max_retries == 3

    def test_custom_values(self):
        """Test custom timeout and max_retries."""
        client = HttpClient(timeout=60.0, max_retries=5)
        assert client.timeout == 60.0
        assert client.max_retries == 5


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

        def raise_status_error():
            raise httpx.HTTPStatusError("Error", request=mock_req, response=mock_resp)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock:
            mock.return_value = mock_resp
            mock_resp.raise_for_status = raise_status_error
            with pytest.raises(ConnectorHTTPError) as exc:
                await client.post("http://test.com", {})
            assert exc.value.status_code == 500

    def test_post_sync(self):
        """Test synchronous POST request."""
        client = HttpClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = client.post_sync("http://test.com", {"data": "test"})
            assert result == {"result": "ok"}
            mock_client.post.assert_called_once()

    def test_post_sync_errors(self):
        """Test error handling for sync POST."""
        client = HttpClient(max_retries=0)

        # Connection error
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.ConnectError("refused")
            mock_client_class.return_value = mock_client

            with pytest.raises(ConnectorError, match="Failed to connect"):
                client.post_sync("http://test.com", {})

        # Timeout error (sync)
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.TimeoutException("timeout")
            mock_client_class.return_value = mock_client

            with pytest.raises(ConnectorError, match="timed out"):
                client.post_sync("http://test.com", {})

        # HTTP error (sync)
        mock_req = httpx.Request("POST", "http://test.com")
        mock_resp = httpx.Response(500, text="Error", request=mock_req)

        def raise_status_error():
            raise httpx.HTTPStatusError("Error", request=mock_req, response=mock_resp)

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_resp.raise_for_status = raise_status_error
            mock_client_class.return_value = mock_client

            with pytest.raises(ConnectorHTTPError) as exc:
                client.post_sync("http://test.com", {})
            assert exc.value.status_code == 500
