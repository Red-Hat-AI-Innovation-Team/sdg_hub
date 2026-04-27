# SPDX-License-Identifier: Apache-2.0
"""Generic HTTP agent connector for arbitrary REST chat endpoints."""

from typing import Any, Optional
import uuid

from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentRequest,
    ChatAgentResponse,
)
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
    >>> request = ChatAgentRequest(
    ...     messages=[ChatAgentMessage(role="user", content="Hello!")],
    ... )
    >>> response = connector.send(request)
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
        request: ChatAgentRequest,
    ) -> dict[str, Any]:
        """Build request payload by placing message content at the configured path.

        Extracts the last user message and places it at
        ``request_message_path`` in the request body.

        Parameters
        ----------
        request : ChatAgentRequest
            Standardized agent request.

        Returns
        -------
        dict
            Request payload with message at the configured path.

        Raises
        ------
        ConnectorError
            If no user message is found.
        """
        content = self._extract_last_user_message(request.messages)
        session_id = (request.context.conversation_id if request.context else "") or ""

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

    def parse_response(self, response: dict[str, Any]) -> ChatAgentResponse:
        """Parse response and build ChatAgentResponse.

        Extracts text (and optionally session ID) from the response using
        the configured dot-notation paths.

        Parameters
        ----------
        response : dict
            Raw response from the endpoint.

        Returns
        -------
        ChatAgentResponse
            Standardized agent response.

        Raises
        ------
        ConnectorError
            If response is not a dict.
        """
        if not isinstance(response, dict):
            raise ConnectorError(
                f"Expected dict response, got {type(response).__name__}"
            )

        messages: list[ChatAgentMessage] = []

        text = _get_nested(response, self.response_text_path)
        if text is not None:
            messages.append(
                ChatAgentMessage(
                    role="assistant",
                    content=str(text),
                    id=str(uuid.uuid4()),
                )
            )
        else:
            messages.append(
                ChatAgentMessage(
                    role="assistant",
                    content="",
                    id=str(uuid.uuid4()),
                )
            )

        custom_outputs = None
        if self.response_session_id_path:
            session_id = _get_nested(response, self.response_session_id_path)
            if session_id is not None:
                custom_outputs = {"session_id": str(session_id)}

        return ChatAgentResponse(
            messages=messages,
            custom_outputs=custom_outputs,
        )

    def _extract_last_user_message(self, messages: list[ChatAgentMessage]) -> str:
        """Extract the last user message content.

        Parameters
        ----------
        messages : list[ChatAgentMessage]
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
            if msg.role == "user" and msg.content:
                return msg.content

        raise ConnectorError(
            "No user message found in messages. "
            "Expected at least one message with role='user' and content."
        )
