# SPDX-License-Identifier: Apache-2.0
"""Tests for LangflowAgentWrapper."""

from unittest.mock import MagicMock, patch

from sdg_hub.core.blocks.agent.agent_wrapper.langflow_agent_wrapper import (
    LangflowAgentWrapper,
)
from sdg_hub.core.utils.error_handling import BlockValidationError
import pytest
import requests


@pytest.fixture
def wrapper():
    """Create a LangflowAgentWrapper instance for testing."""
    return LangflowAgentWrapper(
        agent_framework="langflow",
        agent_url="http://localhost:7860/api/v1/run/test-flow",
        agent_api_key="test-api-key",
        timeout=30.0,
    )


@pytest.fixture
def mock_requests_post_success():
    """Mock successful requests.post()."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"outputs": [{"message": {"text": "Test response"}}]}]
        }
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_requests_post_timeout():
    """Mock requests.post() with timeout."""
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        yield mock_post


@pytest.fixture
def mock_requests_post_http_error():
    """Mock requests.post() with HTTP error."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )
        mock_post.return_value = mock_response
        yield mock_post


class TestLangflowAgentWrapperInit:
    """Test initialization of LangflowAgentWrapper."""

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        wrapper = LangflowAgentWrapper(
            agent_framework="langflow",
            agent_url="http://localhost:8000",
            agent_api_key="my-key",
            timeout=60.0,
        )

        assert wrapper.agent_framework == "langflow"
        assert wrapper.agent_url == "http://localhost:8000"
        assert wrapper.agent_api_key == "my-key"
        assert wrapper.timeout == 60.0

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        wrapper = LangflowAgentWrapper(
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        assert wrapper.agent_api_key is None


class TestExtractUserMessage:
    """Test _extract_user_message() method."""

    def test_extract_user_message_from_list(self, wrapper):
        """Test extracting user message from list of messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, world!"},
        ]

        result = wrapper._extract_user_message(messages)

        assert result == "Hello, world!"

    def test_extract_last_user_message(self, wrapper):
        """Test extracting the last user message when multiple exist."""
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second message"},
        ]

        result = wrapper._extract_user_message(messages)

        assert result == "Second message"

    def test_extract_fallback_to_last_message(self, wrapper):
        """Test fallback to last message when no user message exists."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Assistant message"},
        ]

        result = wrapper._extract_user_message(messages)

        assert result == "Assistant message"

    def test_extract_raises_on_empty_list(self, wrapper):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Messages list is empty"):
            wrapper._extract_user_message([])

    def test_extract_raises_on_no_content(self, wrapper):
        """Test that messages without content raise ValueError."""
        messages = [{"role": "user"}]

        with pytest.raises(ValueError, match="No valid message content found"):
            wrapper._extract_user_message(messages)


class TestBuildPayload:
    """Test _build_payload() method."""

    def test_build_payload(self, wrapper):
        """Test building payload for Langflow API."""
        messages = [{"role": "user", "content": "Test message"}]
        session_id = "session-123"

        payload = wrapper._build_payload(messages, session_id)

        assert payload["output_type"] == "chat"
        assert payload["input_type"] == "chat"
        assert payload["input_value"] == "Test message"
        assert payload["session_id"] == "session-123"


class TestBuildHeaders:
    """Test _build_headers() method."""

    def test_build_headers_with_api_key(self):
        """Test building headers with API key."""
        wrapper = LangflowAgentWrapper(
            agent_framework="langflow",
            agent_url="http://localhost:8000",
            agent_api_key="my-secret-key",
        )

        headers = wrapper._build_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["x-api-key"] == "my-secret-key"

    def test_build_headers_without_api_key(self):
        """Test building headers without API key."""
        wrapper = LangflowAgentWrapper(
            agent_framework="langflow",
            agent_url="http://localhost:8000",
        )

        headers = wrapper._build_headers()

        assert headers["Content-Type"] == "application/json"
        assert "x-api-key" not in headers


class TestValidateResponse:
    """Test validate_response() method."""

    def test_validate_dict_response(self, wrapper):
        """Test validating a dict response."""
        response = {"data": "test", "status": "success"}

        result = wrapper.validate_response(response)

        assert result == response

    def test_validate_raises_on_non_dict(self, wrapper):
        """Test that non-dict response raises ValueError."""
        with pytest.raises(ValueError, match="Expected dict response"):
            wrapper.validate_response("not a dict")

    def test_validate_empty_dict(self, wrapper):
        """Test validating empty dict."""
        response = {}

        result = wrapper.validate_response(response)

        assert result == response


class TestGenerate:
    """Test generate() method."""

    def test_generate_success(self, wrapper, mock_requests_post_success):
        """Test successful generation."""
        messages = [{"role": "user", "content": "Hello"}]
        session_id = "test-session"

        result = wrapper.generate(messages, session_id)

        # Verify request was made
        mock_requests_post_success.assert_called_once()
        call_args = mock_requests_post_success.call_args

        # Check URL
        assert call_args[0][0] == "http://localhost:7860/api/v1/run/test-flow"

        # Check payload
        payload = call_args[1]["json"]
        assert payload["input_value"] == "Hello"
        assert payload["session_id"] == "test-session"

        # Check headers
        headers = call_args[1]["headers"]
        assert headers["x-api-key"] == "test-api-key"

        # Check timeout
        assert call_args[1]["timeout"] == 30.0

        # Check result
        assert isinstance(result, dict)

    def test_generate_timeout_error(self, wrapper, mock_requests_post_timeout):
        """Test generation with timeout error."""
        messages = [{"role": "user", "content": "Hello"}]
        session_id = "test-session"

        with pytest.raises(BlockValidationError, match="Langflow request failed"):
            wrapper.generate(messages, session_id)

    def test_generate_http_error(self, wrapper, mock_requests_post_http_error):
        """Test generation with HTTP error."""
        messages = [{"role": "user", "content": "Hello"}]
        session_id = "test-session"

        with pytest.raises(BlockValidationError, match="Langflow request failed"):
            wrapper.generate(messages, session_id)

    def test_generate_json_parse_error(self, wrapper):
        """Test generation with JSON parsing error."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_post.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            session_id = "test-session"

            with pytest.raises(
                BlockValidationError, match="Failed to parse or validate"
            ):
                wrapper.generate(messages, session_id)

    def test_generate_validation_error(self, wrapper, mock_requests_post_success):
        """Test generation with response validation error."""
        # Mock validate_response to raise
        with patch.object(
            wrapper, "validate_response", side_effect=ValueError("Invalid response")
        ):
            messages = [{"role": "user", "content": "Hello"}]
            session_id = "test-session"

            with pytest.raises(
                BlockValidationError, match="Failed to parse or validate"
            ):
                wrapper.generate(messages, session_id)
