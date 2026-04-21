# SPDX-License-Identifier: Apache-2.0
"""Generic HTTP agent connector for arbitrary REST chat endpoints."""

from typing import Any, Optional

from pydantic import Field, field_validator

from ...utils.logger_config import setup_logger
from ..exceptions import ConnectorError
from ..registry import ConnectorRegistry
from .base import BaseAgentConnector

logger = setup_logger(__name__)


def _set_nested(obj: dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using dot-notation path."""
    keys = path.split(".")
    for key in keys[:-1]:
        obj = obj.setdefault(key, {})
    obj[keys[-1]] = value


def _get_nested(obj: dict, path: str) -> Any:
    """Get a value from a nested dict using dot-notation path.

    Returns None if any key in the path is missing.
    """
    for key in path.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


@ConnectorRegistry.register("generic_http")
class GenericHTTPConnector(BaseAgentConnector):
    """Connector for arbitrary REST chat endpoints.

    Uses declarative JSON path configuration to map between the standard
    message format and any REST API's request/response structure.

    Parameters
    ----------
    request_message_path : str
        Dot-notation path where the message content is placed in the
        request body. Example: ``"input.question"`` produces
        ``{"input": {"question": "<message>"}}``.
    response_text_path : str
        Dot-notation path to extract the text response.
        Example: ``"output.answer"`` extracts from
        ``{"output": {"answer": "..."}}``.
    response_session_id_path : str, optional
        Dot-notation path to extract session ID from the response.
    request_session_id_path : str, optional
        Dot-notation path where session ID is placed in the request body.

    Example
    -------
    >>> from sdg_hub.core.connectors import ConnectorConfig
    >>> from sdg_hub.core.connectors.agent.generic_http import GenericHTTPConnector
    >>>
    >>> config = ConnectorConfig(
    ...     url="http://localhost:8000/api/chat",
    ...     api_key="your-api-key",
    ... )
    >>> connector = GenericHTTPConnector(
    ...     config=config,
    ...     request_message_path="input.query",
    ...     response_text_path="output.result",
    ... )
    >>> response = connector.send(
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     session_id="session-123",
    ... )
    """

    request_message_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Dot-notation path where the message content is placed "
            "in the request body (e.g., 'input.question')."
        ),
    )
    response_text_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Dot-notation path to extract text from the response "
            "(e.g., 'output.answer')."
        ),
    )
    response_session_id_path: Optional[str] = Field(
        None,
        description="Dot-notation path to extract session ID from the response.",
    )
    request_session_id_path: Optional[str] = Field(
        None,
        description="Dot-notation path where session ID is placed in the request body.",
    )

    @field_validator(
        "request_message_path",
        "response_text_path",
        "request_session_id_path",
        "response_session_id_path",
    )
    @classmethod
    def validate_path_format(cls, v: str | None) -> str | None:
        """Validate that path contains only valid dot-notation segments."""
        if v is None:
            return v
        for segment in v.split("."):
            if not segment:
                raise ValueError(
                    f"Invalid path '{v}': empty segment. "
                    f"Use dot-notation like 'output.answer'."
                )
        return v

    def build_request(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        """Build request payload by placing message content at the configured path.

        Extracts the last user message and places it at
        ``request_message_path`` in the request body.

        Parameters
        ----------
        messages : list[dict]
            Messages in standard format.
        session_id : str
            Session identifier.

        Returns
        -------
        dict
            Request payload with message at the configured path.

        Raises
        ------
        ConnectorError
            If no user message is found.
        """
        content = self._extract_last_user_message(messages)

        payload: dict[str, Any] = {}
        _set_nested(payload, self.request_message_path, content)

        if self.request_session_id_path:
            message_path = self.request_message_path
            session_path = self.request_session_id_path
            if (
                message_path == session_path
                or message_path.startswith(f"{session_path}.")
                or session_path.startswith(f"{message_path}.")
            ):
                raise ConnectorError(
                    "request_message_path and request_session_id_path must not overlap"
                )
            _set_nested(payload, self.request_session_id_path, session_id)

        return payload

    def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Validate response and extract fields using configured paths.

        Extracts text (and optionally session ID) from the response using
        the configured dot-notation paths, storing them under known keys
        so the ``extract_text`` and ``extract_session_id`` classmethods
        can retrieve them without instance access.

        Parameters
        ----------
        response : dict
            Raw response from the endpoint.

        Returns
        -------
        dict
            Response dict with ``_extracted_text`` (and optionally
            ``_extracted_session_id``) injected.

        Raises
        ------
        ConnectorError
            If response is not a dict.
        """
        if not isinstance(response, dict):
            raise ConnectorError(
                f"Expected dict response, got {type(response).__name__}"
            )

        # Work on a copy to avoid mutating the caller's dict
        parsed = dict(response)

        # Remove reserved keys so upstream values can't leak through
        parsed.pop("_extracted_text", None)
        parsed.pop("_extracted_session_id", None)

        text = _get_nested(parsed, self.response_text_path)
        if text is not None:
            parsed["_extracted_text"] = str(text)

        if self.response_session_id_path:
            session_id = _get_nested(parsed, self.response_session_id_path)
            if session_id is not None:
                parsed["_extracted_session_id"] = str(session_id)

        return parsed

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

    # ------------------------------------------------------------------
    # Response field extraction
    # ------------------------------------------------------------------

    @classmethod
    def extract_text(cls, response: dict[str, Any]) -> str | None:
        """Extract text content from a generic HTTP response.

        Reads from ``_extracted_text``, which is injected by
        ``parse_response`` using the configured ``response_text_path``.

        Parameters
        ----------
        response : dict
            Response dict from ``parse_response``.

        Returns
        -------
        str or None
            Extracted text, or None if not found.
        """
        return response.get("_extracted_text")

    @classmethod
    def extract_session_id(cls, response: dict[str, Any]) -> str | None:
        """Extract session ID from a generic HTTP response.

        Reads from ``_extracted_session_id``, which is injected by
        ``parse_response`` using the configured ``response_session_id_path``.

        Parameters
        ----------
        response : dict
            Response dict from ``parse_response``.

        Returns
        -------
        str or None
            Extracted session ID, or None if not configured or not found.
        """
        return response.get("_extracted_session_id")

    @classmethod
    def extract_tool_trace(
        cls, response: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """Extract tool trace from a generic HTTP response.

        Generic HTTP endpoints typically don't produce tool traces.

        Parameters
        ----------
        response : dict
            Raw response from the endpoint.

        Returns
        -------
        None
            Always returns None.
        """
        return None
