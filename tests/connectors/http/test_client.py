# SPDX-License-Identifier: Apache-2.0
"""Tests for HttpClient."""

from unittest.mock import AsyncMock, MagicMock, patch

from sdg_hub.core.connectors.exceptions import ConnectorError, ConnectorHTTPError
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
    async def test_post(self):
        """Test async POST request."""
        client = HttpClient()
        mock_response = httpx.Response(
            200,
            json={"result": "success"},
            request=httpx.Request("POST", "http://test.com"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            result = await client.post("http://test.com", {"data": "test"})
            assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_post_errors(self):
        """Test error handling for async POST."""
        client = HttpClient(max_retries=1)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock:
            mock.side_effect = httpx.TimeoutException("timeout")
            with pytest.raises(ConnectorError, match="timed out"):
                await client.post("http://test.com", {})

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock:
            mock.side_effect = httpx.ConnectError("refused")
            with pytest.raises(ConnectorError, match="Failed to connect"):
                await client.post("http://test.com", {})

        mock_req = httpx.Request("POST", "http://test.com")
        mock_resp = httpx.Response(500, text="Error", request=mock_req)
        mock_resp.raise_for_status = lambda: (_ for _ in ()).throw(
            httpx.HTTPStatusError("Error", request=mock_req, response=mock_resp)
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock:
            mock.return_value = mock_resp
            with pytest.raises(ConnectorHTTPError) as exc:
                await client.post("http://test.com", {})
            assert exc.value.status_code == 500

    def test_post_sync(self):
        """Test synchronous POST request."""
        client = HttpClient()
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

    def test_post_sync_errors(self):
        """Test error handling for sync POST."""
        client = HttpClient(max_retries=0)

        def make_mock_client(side_effect):
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = side_effect
            return mock_client

        with patch("httpx.Client") as mock_class:
            mock_class.return_value = make_mock_client(httpx.ConnectError("refused"))
            with pytest.raises(ConnectorError, match="Failed to connect"):
                client.post_sync("http://test.com", {})

        with patch("httpx.Client") as mock_class:
            mock_class.return_value = make_mock_client(httpx.TimeoutException("timeout"))
            with pytest.raises(ConnectorError, match="timed out"):
                client.post_sync("http://test.com", {})

        mock_req = httpx.Request("POST", "http://test.com")
        mock_resp = httpx.Response(500, text="Error", request=mock_req)
        mock_resp.raise_for_status = lambda: (_ for _ in ()).throw(
            httpx.HTTPStatusError("Error", request=mock_req, response=mock_resp)
        )

        with patch("httpx.Client") as mock_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_class.return_value = mock_client

            with pytest.raises(ConnectorHTTPError) as exc:
                client.post_sync("http://test.com", {})
            assert exc.value.status_code == 500
