# SPDX-License-Identifier: Apache-2.0
"""Tests for ConnectorRegistry."""

from unittest.mock import patch

import pytest

from sdg_hub.core.connectors.base import BaseConnector, ConnectorConfig
from sdg_hub.core.connectors.registry import ConnectorMetadata, ConnectorRegistry


class MockConnector(BaseConnector):
    """Mock connector for testing."""

    def _initialize_client(self) -> None:
        self._client = "mock_client"

    def execute(self, request):
        return {"result": "mock"}


class TestConnectorMetadata:
    """Test ConnectorMetadata dataclass."""

    def test_valid_metadata(self):
        """Test creating valid metadata."""
        metadata = ConnectorMetadata(
            name="test_connector",
            connector_class=MockConnector,
            category="test",
            description="A test connector",
        )
        assert metadata.name == "test_connector"
        assert metadata.connector_class == MockConnector
        assert metadata.category == "test"
        assert metadata.description == "A test connector"
        assert not metadata.supports_async
        assert not metadata.supports_streaming
        assert not metadata.supports_batch

    def test_metadata_with_capabilities(self):
        """Test metadata with capability flags."""
        metadata = ConnectorMetadata(
            name="async_connector",
            connector_class=MockConnector,
            category="agent",
            supports_async=True,
            supports_streaming=True,
        )
        assert metadata.supports_async
        assert metadata.supports_streaming
        assert not metadata.supports_batch

    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Connector name cannot be empty"):
            ConnectorMetadata(name="", connector_class=MockConnector, category="test")

    def test_non_class_raises_error(self):
        """Test that non-class raises ValueError."""
        with pytest.raises(ValueError, match="connector_class must be a class"):
            ConnectorMetadata(
                name="test", connector_class="not_a_class", category="test"
            )


class TestConnectorRegistry:
    """Test ConnectorRegistry functionality."""

    def setup_method(self):
        """Save and clear registry state for isolated testing."""
        self._saved_metadata = ConnectorRegistry._metadata.copy()
        self._saved_by_category = {
            k: v.copy() for k, v in ConnectorRegistry._by_category.items()
        }
        ConnectorRegistry.clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        ConnectorRegistry._metadata.clear()
        ConnectorRegistry._metadata.update(self._saved_metadata)
        ConnectorRegistry._by_category.clear()
        ConnectorRegistry._by_category.update(self._saved_by_category)

    def test_register_valid_connector(self):
        """Test registering a valid connector."""

        @ConnectorRegistry.register(
            "test_connector",
            category="test",
            description="A test connector",
        )
        class TestConnector(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        assert "test_connector" in ConnectorRegistry._metadata
        metadata = ConnectorRegistry._metadata["test_connector"]
        assert metadata.name == "test_connector"
        assert metadata.connector_class == TestConnector
        assert metadata.category == "test"

    def test_register_with_capabilities(self):
        """Test registering connector with capability flags."""

        @ConnectorRegistry.register(
            "async_connector",
            category="agent",
            supports_async=True,
            supports_streaming=True,
        )
        class AsyncConnector(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        metadata = ConnectorRegistry._metadata["async_connector"]
        assert metadata.supports_async
        assert metadata.supports_streaming
        assert not metadata.supports_batch

    def test_register_updates_category_index(self):
        """Test that registration updates the category index."""

        @ConnectorRegistry.register("connector1", category="agent")
        class Connector1(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        @ConnectorRegistry.register("connector2", category="agent")
        class Connector2(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        assert "agent" in ConnectorRegistry._by_category
        assert "connector1" in ConnectorRegistry._by_category["agent"]
        assert "connector2" in ConnectorRegistry._by_category["agent"]

    def test_register_invalid_class_raises_error(self):
        """Test that registering non-class raises ValueError."""
        with pytest.raises(ValueError, match="Expected a class"):

            @ConnectorRegistry.register("invalid", category="test")
            def not_a_class():
                pass

    def test_register_non_connector_raises_error(self):
        """Test that non-connector class raises ValueError."""
        with pytest.raises(ValueError, match="must inherit from BaseConnector"):

            @ConnectorRegistry.register("invalid", category="test")
            class NotAConnector:
                pass

    def test_get_existing_connector(self):
        """Test getting an existing connector."""

        @ConnectorRegistry.register("my_connector", category="test")
        class MyConnector(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        retrieved = ConnectorRegistry.get("my_connector")
        assert retrieved == MyConnector

    def test_get_non_existent_connector_raises_error(self):
        """Test that getting non-existent connector raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            ConnectorRegistry.get("non_existent")

        error_msg = str(exc_info.value)
        assert "non_existent" in error_msg
        assert "not found" in error_msg

    def test_get_with_suggestions(self):
        """Test that error includes suggestions for similar names."""

        @ConnectorRegistry.register("langflow", category="agent")
        class LangflowConnector(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        with pytest.raises(KeyError) as exc_info:
            ConnectorRegistry.get("langflo")  # Missing 'w'

        error_msg = str(exc_info.value)
        assert "Did you mean: langflow" in error_msg

    def test_get_metadata(self):
        """Test getting connector metadata."""

        @ConnectorRegistry.register(
            "my_connector", category="test", description="Test connector"
        )
        class MyConnector(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        metadata = ConnectorRegistry.get_metadata("my_connector")
        assert metadata.name == "my_connector"
        assert metadata.description == "Test connector"

    def test_get_metadata_non_existent_raises_error(self):
        """Test that getting metadata for non-existent connector raises KeyError."""
        with pytest.raises(KeyError):
            ConnectorRegistry.get_metadata("non_existent")

    def test_list_by_category(self):
        """Test listing connectors by category."""

        @ConnectorRegistry.register("conn1", category="agent")
        class Conn1(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        @ConnectorRegistry.register("conn2", category="agent")
        class Conn2(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        @ConnectorRegistry.register("conn3", category="storage")
        class Conn3(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        agent_connectors = ConnectorRegistry.list_by_category("agent")
        assert agent_connectors == ["conn1", "conn2"]

        storage_connectors = ConnectorRegistry.list_by_category("storage")
        assert storage_connectors == ["conn3"]

    def test_list_by_category_empty(self):
        """Test listing connectors for non-existent category."""
        result = ConnectorRegistry.list_by_category("non_existent")
        assert result == []

    def test_categories(self):
        """Test getting all categories."""

        @ConnectorRegistry.register("conn1", category="agent")
        class Conn1(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        @ConnectorRegistry.register("conn2", category="storage")
        class Conn2(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        categories = ConnectorRegistry.categories()
        assert categories == ["agent", "storage"]

    def test_list_all(self):
        """Test listing all connectors."""

        @ConnectorRegistry.register("alpha", category="test")
        class Alpha(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        @ConnectorRegistry.register("beta", category="test")
        class Beta(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        all_connectors = ConnectorRegistry.list_all()
        assert all_connectors == ["alpha", "beta"]

    def test_clear(self):
        """Test clearing the registry."""

        @ConnectorRegistry.register("test", category="test")
        class TestConn(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        assert len(ConnectorRegistry._metadata) == 1
        ConnectorRegistry.clear()
        assert len(ConnectorRegistry._metadata) == 0
        assert len(ConnectorRegistry._by_category) == 0

    def test_discover_empty_registry(self):
        """Test discover with empty registry."""
        with patch("sdg_hub.core.connectors.registry.console") as mock_console:
            ConnectorRegistry.discover()
            mock_console.print.assert_called_once_with(
                "[yellow]No connectors registered yet.[/yellow]"
            )

    def test_discover_with_connectors(self):
        """Test discover with registered connectors."""

        @ConnectorRegistry.register(
            "test_connector",
            category="agent",
            description="Test connector",
            supports_async=True,
        )
        class TestConnector(BaseConnector):
            def _initialize_client(self):
                pass

            def execute(self, request):
                return {}

        with patch("sdg_hub.core.connectors.registry.console") as mock_console:
            ConnectorRegistry.discover()
            # Should be called multiple times (table + summary)
            assert mock_console.print.call_count >= 2
