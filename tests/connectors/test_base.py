# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseConnector and ConnectorConfig."""

import pytest

from sdg_hub.core.connectors.base import BaseConnector, ConnectorConfig


class ConcreteConnector(BaseConnector):
    """Concrete connector for testing."""

    def execute(self, request):
        return {"result": request.get("input", "default")}


class TestConnectorConfig:
    """Test ConnectorConfig."""

    def test_config_with_defaults_and_custom_values(self):
        """Test default and custom configuration."""
        # Defaults
        config = ConnectorConfig()
        assert config.url is None
        assert config.timeout == 120.0
        assert config.max_retries == 3

        # Custom
        config = ConnectorConfig(url="http://localhost", timeout=60.0, max_retries=5)
        assert config.url == "http://localhost"
        assert config.timeout == 60.0

    def test_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError):
            ConnectorConfig(timeout=0)
        with pytest.raises(ValueError):
            ConnectorConfig(max_retries=-1)


class TestBaseConnector:
    """Test BaseConnector."""

    def test_execute(self):
        """Test execute method."""
        connector = ConcreteConnector(config=ConnectorConfig())
        result = connector.execute({"input": "hello"})
        assert result == {"result": "hello"}

    @pytest.mark.asyncio
    async def test_aexecute(self):
        """Test async execute wraps sync."""
        connector = ConcreteConnector(config=ConnectorConfig())
        result = await connector.aexecute({"input": "async_test"})
        assert result == {"result": "async_test"}
