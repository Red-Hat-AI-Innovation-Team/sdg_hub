# SPDX-License-Identifier: Apache-2.0
"""Langflow agent wrapper implementation."""

from typing import Any, Optional
import traceback

import requests

from ....utils.error_handling import BlockValidationError
from ....utils.logger_config import setup_logger
from .base import BaseAgentWrapper

logger = setup_logger(__name__)


class LangflowAgentWrapper(BaseAgentWrapper):
    """Wrapper for Langflow agent framework.

    This wrapper adapts the Langflow API to work with the AgentBlock interface.
    It handles message conversion, HTTP requests, and response validation specific
    to Langflow's API format.

    Langflow expects:
    - input_value: A string (not a list of messages)
    - output_type: "chat"
    - input_type: "chat"
    - session_id: Optional session identifier
    """

    def __init__(
        self,
        agent_framework: str,
        agent_url: str,
        agent_api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize the Langflow wrapper.

        Parameters
        ----------
        agent_framework : str
            Should be "langflow".
        agent_url : str
            The Langflow API endpoint URL.
        agent_api_key : Optional[str], optional
            API key for Langflow authentication, by default None.
        timeout : float, optional
            Request timeout in seconds, by default 120.0.
        """
        super().__init__(agent_framework, agent_url, agent_api_key, timeout)

    def _extract_user_message(self, messages: list[dict[str, Any]]) -> str:
        """Extract the last user message content from messages list.

        Langflow expects a single string input, not a list of messages.
        This method extracts the most recent user message.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            List of message dictionaries with 'role' and 'content' fields.

        Returns
        -------
        str
            The content of the last user message.

        Raises
        ------
        ValueError
            If no user message is found or messages is empty.
        """
        if not messages:
            raise ValueError("Messages list is empty")

        # Try to find the last user message
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    logger.debug(
                        "Extracted user message with length=%d",
                        len(content),
                        extra={"content_preview": content[:100]},
                    )
                    return content

        # If no user message found, use the last message's content
        if messages and isinstance(messages[-1], dict):
            content = messages[-1].get("content", "")
            if content:
                logger.warning(
                    "No user message found, using last message content",
                    extra={"messages": messages},
                )
                return content

        raise ValueError("No valid message content found in messages list")

    def _build_payload(
        self, messages: list[dict[str, Any]], session_id: str
    ) -> dict[str, Any]:
        """Build the request payload for Langflow API.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            The messages to send.
        session_id : str
            The session ID.

        Returns
        -------
        dict[str, Any]
            The request payload.
        """
        input_value = self._extract_user_message(messages)

        payload = {
            "output_type": "chat",
            "input_type": "chat",
            "input_value": input_value,
            "session_id": session_id,
        }

        logger.debug(
            "Built Langflow payload with session_id=%s, input_length=%d",
            session_id,
            len(input_value),
            extra={"session_id": session_id, "input_length": len(input_value)},
        )

        return payload

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for the request.

        Returns
        -------
        dict[str, str]
            HTTP headers including API key if provided.
        """
        headers = {"Content-Type": "application/json"}
        if self.agent_api_key:
            headers["x-api-key"] = self.agent_api_key
        return headers

    def validate_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize the Langflow response.

        Parameters
        ----------
        response : dict[str, Any]
            The raw response from Langflow API.

        Returns
        -------
        dict[str, Any]
            The validated response.

        Raises
        ------
        ValueError
            If the response is invalid or missing required fields.
        """
        if not isinstance(response, dict):
            raise ValueError(
                f"Expected dict response from Langflow, got {type(response)}"
            )

        # Log the response structure for debugging
        logger.debug(
            "Validating Langflow response with keys: %s",
            list(response.keys()),
            extra={"response_keys": list(response.keys())},
        )

        # Langflow responses can have various structures depending on the flow
        # We'll do basic validation and return the full response
        # Users can override this for specific validation if needed

        return response

    def generate(
        self, messages: list[dict[str, Any]], session_id: str
    ) -> dict[str, Any]:
        """Generate a response from Langflow.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            The messages to send to the agent.
        session_id : str
            The session ID to use for the agent.

        Returns
        -------
        dict[str, Any]
            The validated response from the agent.

        Raises
        ------
        BlockValidationError
            If the request fails or response is invalid.
        """
        logger.info(
            "Sending request to Langflow with session_id=%s",
            session_id,
            extra={
                "session_id": session_id,
                "agent_url": self.agent_url,
                "timeout": self.timeout,
            },
        )

        try:
            payload = self._build_payload(messages, session_id)
            headers = self._build_headers()

            response = requests.post(
                self.agent_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

            logger.debug(
                "Received response with status_code=%d",
                response.status_code,
                extra={"status_code": response.status_code, "session_id": session_id},
            )

            response.raise_for_status()
            response_data = response.json()
            validated_response = self.validate_response(response_data)

            logger.info(
                "Successfully received response from Langflow for session_id=%s",
                session_id,
                extra={"session_id": session_id},
            )

            return validated_response

        except requests.exceptions.RequestException as e:
            # Catches all requests-related errors (timeout, HTTP errors, connection errors, etc.)
            error_msg = f"Langflow request failed: {str(e)}"
            logger.error(
                error_msg,
                extra={
                    "session_id": session_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            raise BlockValidationError(error_msg) from e

        except (ValueError, TypeError) as e:
            # Catches JSON parsing errors and validation errors
            error_msg = f"Failed to parse or validate Langflow response: {str(e)}"
            logger.error(
                error_msg,
                extra={
                    "session_id": session_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            raise BlockValidationError(error_msg) from e
