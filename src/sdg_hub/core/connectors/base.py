# SPDX-License-Identifier: Apache-2.0
"""Base connector classes for external service integrations."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)


class ConnectorConfig(BaseModel):
    """Base configuration for all connectors.

    Attributes
    ----------
    url : str, optional
        The base URL for the external service.
    api_key : str, optional
        API key for authentication.
    timeout : float
        Request timeout in seconds. Default is 120.0.
    max_retries : int
        Maximum number of retry attempts. Default is 3.
    """

    url: Optional[str] = Field(None, description="Base URL for the service")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    timeout: float = Field(120.0, description="Request timeout in seconds", gt=0)
    max_retries: int = Field(3, description="Maximum retry attempts", ge=0)
    extract_text: bool = Field(
        False,
        description="Extract just the text content from agent response",
    )

    model_config = ConfigDict(extra="allow")


class BaseConnector(BaseModel, ABC):
    """Abstract base class for all connectors.

    Connectors handle communication with external services.

    Attributes
    ----------
    config : ConnectorConfig
        Configuration for the connector.

    Example
    -------
    >>> class MyConnector(BaseConnector):
    ...     def execute(self, request: dict) -> dict:
    ...         return {"result": request.get("input")}
    ...
    >>> connector = MyConnector(config=ConnectorConfig(url="http://example.com"))
    >>> result = connector.execute({"input": "test"})
    """

    config: ConnectorConfig = Field(..., description="Connector configuration")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def execute(self, request: Any) -> Any:
        """Execute a synchronous request.

        Parameters
        ----------
        request : Any
            The request to execute (format depends on connector type).

        Returns
        -------
        Any
            The response from the external service.
        """
        pass

    async def aexecute(self, request: Any) -> Any:
        """Execute an asynchronous request.

        Default implementation wraps sync execute in a thread.
        Subclasses should override for true async support.

        Parameters
        ----------
        request : Any
            The request to execute.

        Returns
        -------
        Any
            The response from the external service.
        """
        import asyncio

        return await asyncio.to_thread(self.execute, request)
