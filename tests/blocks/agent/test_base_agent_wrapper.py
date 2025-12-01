# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseAgentWrapper abstract class."""

from typing import Any

import pytest

from sdg_hub.core.blocks.agent.agent_wrapper.base import BaseAgentWrapper


class ConcreteAgentWrapper(BaseAgentWrapper):
    """Concrete implementation for testing."""

    def generate(
        self, messages: list[dict[str, Any]], session_id: str
    ) -> dict[str, Any]:
        """Test implementation."""
        return {"result": "test response", "session_id": session_id}

    def validate_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Test implementation."""
        if not isinstance(response, dict):
            raise ValueError("Response must be a dict")
        return response


class TestBaseAgentWrapper:
    """Test suite for BaseAgentWrapper."""

    def test_initialization_with_defaults(self):
        """Test wrapper initialization with default values."""
        wrapper = ConcreteAgentWrapper(
            agent_framework="test_framework",
            agent_url="http://localhost:8000",
        )

        assert wrapper.agent_framework == "test_framework"
        assert wrapper.agent_url == "http://localhost:8000"
        assert wrapper.agent_api_key is None
        assert wrapper.timeout == 120.0

    def test_initialization_with_custom_values(self):
        """Test wrapper initialization with custom values."""
        wrapper = ConcreteAgentWrapper(
            agent_framework="custom_framework",
            agent_url="http://example.com/api",
            agent_api_key="test-key-123",
            timeout=60.0,
        )

        assert wrapper.agent_framework == "custom_framework"
        assert wrapper.agent_url == "http://example.com/api"
        assert wrapper.agent_api_key == "test-key-123"
        assert wrapper.timeout == 60.0

    def test_generate_method_works(self):
        """Test that concrete implementation of generate() works."""
        wrapper = ConcreteAgentWrapper(
            agent_framework="test", agent_url="http://localhost:8000"
        )

        messages = [{"role": "user", "content": "Hello"}]
        session_id = "test-session-123"

        result = wrapper.generate(messages, session_id)

        assert isinstance(result, dict)
        assert result["result"] == "test response"
        assert result["session_id"] == session_id

    def test_validate_response_method_works(self):
        """Test that concrete implementation of validate_response() works."""
        wrapper = ConcreteAgentWrapper(
            agent_framework="test", agent_url="http://localhost:8000"
        )

        response = {"data": "test"}
        result = wrapper.validate_response(response)

        assert result == response

    def test_validate_response_raises_on_invalid_input(self):
        """Test that validate_response raises on invalid input."""
        wrapper = ConcreteAgentWrapper(
            agent_framework="test", agent_url="http://localhost:8000"
        )

        with pytest.raises(ValueError, match="Response must be a dict"):
            wrapper.validate_response("not a dict")

    def test_cannot_instantiate_base_class_directly(self):
        """Test that BaseAgentWrapper cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseAgentWrapper(
                agent_framework="test",
                agent_url="http://localhost:8000",
            )


class IncompleteWrapper(BaseAgentWrapper):
    """Wrapper that doesn't implement all abstract methods."""

    def generate(
        self, messages: list[dict[str, Any]], session_id: str
    ) -> dict[str, Any]:
        """Implement only generate."""
        return {}


class TestAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_missing_validate_response_fails(self):
        """Test that missing validate_response() causes instantiation to fail."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteWrapper(
                agent_framework="test",
                agent_url="http://localhost:8000",
            )
