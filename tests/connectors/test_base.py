# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseConnector and ConnectorConfig."""

from sdg_hub.core.connectors.base import BaseConnector, ConnectorConfig
import pytest


class TestConnectorConfig:
    """Test ConnectorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConnectorConfig()
        assert config.url is None
        assert config.api_key is None
        assert config.timeout == 120.0
        assert config.max_retries == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ConnectorConfig(
            url="http://localhost:7860",
            api_key="secret",
            timeout=60.0,
            max_retries=5,
        )
        assert config.url == "http://localhost:7860"
        assert config.api_key == "secret"
        assert config.timeout == 60.0
        assert config.max_retries == 5

    def test_timeout_must_be_positive(self):
        """Test that timeout must be positive."""
        with pytest.raises(ValueError):
            ConnectorConfig(timeout=0)

        with pytest.raises(ValueError):
            ConnectorConfig(timeout=-1)

    def test_max_retries_must_be_non_negative(self):
        """Test that max_retries must be non-negative."""
        with pytest.raises(ValueError):
            ConnectorConfig(max_retries=-1)

        # Zero is valid
        config = ConnectorConfig(max_retries=0)
        assert config.max_retries == 0

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed."""
        config = ConnectorConfig(
            url="http://localhost",
            custom_field="custom_value",
        )
        assert config.custom_field == "custom_value"


class ConcreteConnector(BaseConnector):
    """Concrete connector implementation for testing."""

    def _initialize_client(self) -> None:
        self._client = "initialized_client"

    def execute(self, request):
        return {"result": request.get("input", "default")}


class TestBaseConnector:
    """Test BaseConnector."""

    def test_create_connector(self):
        """Test creating a connector."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = ConcreteConnector(config=config)
        assert connector.config.url == "http://localhost:7860"
        assert not connector.is_ready

    def test_warm_up(self):
        """Test warm_up initializes client."""
        config = ConnectorConfig()
        connector = ConcreteConnector(config=config)

        assert not connector.is_ready
        assert connector._client is None

        connector.warm_up()

        assert connector.is_ready
        assert connector._client == "initialized_client"

    def test_warm_up_idempotent(self):
        """Test that multiple warm_up calls are idempotent."""
        config = ConnectorConfig()
        connector = ConcreteConnector(config=config)

        connector.warm_up()
        connector.warm_up()
        connector.warm_up()

        assert connector.is_ready

    def test_close(self):
        """Test close releases resources."""
        config = ConnectorConfig()
        connector = ConcreteConnector(config=config)

        connector.warm_up()
        assert connector.is_ready

        connector.close()
        assert not connector.is_ready
        assert connector._client is None

    def test_context_manager(self):
        """Test context manager usage."""
        config = ConnectorConfig()

        with ConcreteConnector(config=config) as connector:
            assert connector.is_ready
            result = connector.execute({"input": "test"})
            assert result == {"result": "test"}

        assert not connector.is_ready

    def test_execute(self):
        """Test execute method."""
        config = ConnectorConfig()
        connector = ConcreteConnector(config=config)
        connector.warm_up()

        result = connector.execute({"input": "hello"})
        assert result == {"result": "hello"}

    @pytest.mark.asyncio
    async def test_aexecute(self):
        """Test async execute wraps sync execute."""
        config = ConnectorConfig()
        connector = ConcreteConnector(config=config)
        connector.warm_up()

        result = await connector.aexecute({"input": "async_test"})
        assert result == {"result": "async_test"}

    def test_capability_flags_defaults(self):
        """Test default capability flags."""
        assert ConcreteConnector.supports_async is False
        assert ConcreteConnector.supports_streaming is False
        assert ConcreteConnector.supports_batch is False

    def test_capability_flags_can_be_set(self):
        """Test capability flags can be set by subclasses."""

        class AsyncConnector(BaseConnector):
            supports_async = True
            supports_streaming = True

            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        assert AsyncConnector.supports_async is True
        assert AsyncConnector.supports_streaming is True
        assert AsyncConnector.supports_batch is False
