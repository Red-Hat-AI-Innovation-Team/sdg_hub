# SPDX-License-Identifier: Apache-2.0
"""Langflow agent framework connector."""

from typing import Any

from ...utils.logger_config import setup_logger
from ..exceptions import ConnectorError
from ..registry import ConnectorRegistry
from .base import BaseAgentConnector

logger = setup_logger(__name__)


@ConnectorRegistry.register("langflow")
class LangflowConnector(BaseAgentConnector):
    """Connector for Langflow agent framework.

    Langflow is a visual framework for building LLM-powered applications.
    This connector handles the specific request/response format used by
    Langflow's API.

    Langflow expects:
    - Single string input (not message array)
    - Session ID for conversation tracking
    - Returns structured response with outputs

    Example
    -------
    >>> from sdg_hub.core.connectors import ConnectorConfig, LangflowConnector
    >>>
    >>> config = ConnectorConfig(
    ...     url="http://localhost:7860/api/v1/run/my-flow",
    ...     api_key="your-api-key",
    ... )
    >>> connector = LangflowConnector(config=config)
    >>> response = connector.send(
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     session_id="session-123",
    ... )
    """

    def _build_headers(self) -> dict[str, str]:
        """Build headers for Langflow API.

        Langflow uses x-api-key header for authentication.

        Returns
        -------
        dict[str, str]
            HTTP headers.
        """
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            # Langflow uses x-api-key header
            headers["x-api-key"] = self.config.api_key
        return headers

    def build_request(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        """Build Langflow-specific request payload.

        Langflow expects a single string input, not a message array.
        We extract the last user message content.

        Parameters
        ----------
        messages : list[dict]
            Messages in standard format.
        session_id : str
            Session identifier.

        Returns
        -------
        dict
            Langflow API request payload.
        """
        input_value = self._extract_last_user_message(messages)

        return {
            "output_type": "chat",
            "input_type": "chat",
            "input_value": input_value,
            "session_id": session_id,
        }

    def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Parse Langflow response.

        Parameters
        ----------
        response : dict
            Raw response from Langflow API.

        Returns
        -------
        dict
            Validated response dict.

        Raises
        ------
        ConnectorError
            If response is not a valid dict.
        """
        if not isinstance(response, dict):
            raise ConnectorError(
                f"Expected dict response, got {type(response).__name__}"
            )

        return response

    # ------------------------------------------------------------------
    # Response field extraction
    # ------------------------------------------------------------------

    @classmethod
    def extract_text(cls, response: dict[str, Any]) -> str | None:
        """Extract text content from a Langflow response.

        Navigates ``outputs[0].outputs[0].results.message.text``.

        Parameters
        ----------
        response : dict
            Raw Langflow API response.

        Returns
        -------
        str or None
            Extracted text, or None if the field is missing or
            explicitly None.
        """
        try:
            text = response["outputs"][0]["outputs"][0]["results"]["message"]["text"]
            if text is None:
                logger.warning("Text field is None")
                return None
            return text
        except (KeyError, IndexError, TypeError):
            return None

    @classmethod
    def extract_session_id(cls, response: dict[str, Any]) -> str | None:
        """Extract session ID from a Langflow response.

        Parameters
        ----------
        response : dict
            Raw Langflow API response.

        Returns
        -------
        str or None
            Extracted session ID, or None if the field is missing or
            explicitly None.
        """
        if "session_id" not in response:
            return None
        if response["session_id"] is None:
            logger.warning("Session ID field is None")
            return None
        return response["session_id"]

    @classmethod
    def extract_tool_trace(
        cls, response: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """Extract tool call trace from Langflow content_blocks.

        Langflow agent responses include an 'Agent Steps' content block
        with structured entries for each step: user input (text), tool
        calls (tool_use with name, tool_input, output), and final output
        (text).

        Parameters
        ----------
        response : dict
            Raw Langflow API response.

        Returns
        -------
        list[dict] or None
            List of step dicts from content_blocks, or None if not found.
        """
        for path_fn in [
            lambda r: r["outputs"][0]["outputs"][0]["results"]["message"]["data"][
                "content_blocks"
            ],
            lambda r: r["outputs"][0]["outputs"][0]["results"]["message"][
                "content_blocks"
            ],
        ]:
            try:
                content_blocks = path_fn(response)
                if not content_blocks:
                    continue
                for block in content_blocks:
                    contents = block.get("contents")
                    if contents and isinstance(contents, list):
                        return contents
            except (KeyError, IndexError, TypeError):
                continue

        logger.warning("No content_blocks with tool trace found in Langflow response")
        return None

    def _extract_last_user_message(self, messages: list[dict[str, Any]]) -> str:
        """Extract the last user message content.

        Parameters
        ----------
        messages : list[dict]
            List of messages.

        Returns
        -------
        str
            Content of the last user message.

        Raises
        ------
        ConnectorError
            If no user message is found.
        """
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"]

        raise ConnectorError(
            "No user message found in messages. "
            "Expected at least one message with role='user' and content."
        )
