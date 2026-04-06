# SPDX-License-Identifier: Apache-2.0
"""Tests for ConnectorRegistry."""

import pytest

from sdg_hub.core.connectors.base import BaseConnector
from sdg_hub.core.connectors.exceptions import ConnectorError
from sdg_hub.core.connectors.registry import ConnectorRegistry


class TestConnectorRegistry:
    """Test ConnectorRegistry."""

    def setup_method(self):
        """Save and clear registry state."""
        self._saved = ConnectorRegistry._connectors.copy()
        ConnectorRegistry.clear()

    def teardown_method(self):
        """Restore registry state."""
        ConnectorRegistry._connectors.clear()
        ConnectorRegistry._connectors.update(self._saved)

    def test_register_and_get(self):
        """Test registering and retrieving connectors."""

        @ConnectorRegistry.register("test")
        class TestConnector(BaseConnector):
            def execute(self, request):
                return {}

        assert ConnectorRegistry.get("test") == TestConnector
        assert ConnectorRegistry.list_all() == ["test"]

    def test_register_validates_class(self):
        """Test registration validates connector class."""
        with pytest.raises(ConnectorError, match="Expected a class"):

            @ConnectorRegistry.register("invalid")
            def not_a_class():
                pass

        with pytest.raises(ConnectorError, match="must inherit from BaseConnector"):

            @ConnectorRegistry.register("invalid")
            class NotAConnector:
                pass

    def test_get_unknown_raises_error(self):
        """Test getting unknown connector raises with helpful message."""

        @ConnectorRegistry.register("langflow")
        class LF(BaseConnector):
            def execute(self, request):
                return {}

        with pytest.raises(ConnectorError) as exc_info:
            ConnectorRegistry.get("unknown")

        assert "not found" in str(exc_info.value)
        assert "langflow" in str(exc_info.value)
