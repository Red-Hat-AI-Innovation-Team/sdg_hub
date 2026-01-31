# SPDX-License-Identifier: Apache-2.0
"""Base connector classes for external service integrations."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)


class ConnectorConfig(BaseModel):
    """Base configuration for all connectors.

    This configuration class provides common settings used across
    all connector types, such as URL, authentication, and timeout settings.

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

    model_config = ConfigDict(extra="allow")


class BaseConnector(BaseModel, ABC):
    """Abstract base class for all connectors.

    Connectors handle communication with external services. They provide
    a unified interface with lazy initialization (warm_up pattern),
    capability flags, and context manager support.

    Features:
    - Lazy initialization via warm_up() pattern (inspired by Haystack)
    - Capability flags (supports_async, supports_streaming, supports_batch)
    - Context manager support for resource cleanup
    - Pydantic validation for configuration

    Attributes
    ----------
    config : ConnectorConfig
        Configuration for the connector.

    Class Attributes
    ----------------
    supports_async : bool
        Whether the connector supports async operations. Default False.
    supports_streaming : bool
        Whether the connector supports streaming responses. Default False.
    supports_batch : bool
        Whether the connector supports batch operations. Default False.

    Example
    -------
    >>> class MyConnector(BaseConnector):
    ...     def _initialize_client(self) -> None:
    ...         self._client = MyClient(self.config.url)
    ...
    ...     def execute(self, request: dict) -> dict:
    ...         return self._client.send(request)
    ...
    >>> with MyConnector(config=ConnectorConfig(url="http://example.com")) as conn:
    ...     result = conn.execute({"data": "test"})
    """

    config: ConnectorConfig = Field(..., description="Connector configuration")

    # Capability flags (set by subclasses as class variables)
    supports_async: ClassVar[bool] = False
    supports_streaming: ClassVar[bool] = False
    supports_batch: ClassVar[bool] = False

    # Private attributes for internal state
    _client: Optional[Any] = PrivateAttr(default=None)
    _is_warmed_up: bool = PrivateAttr(default=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def warm_up(self) -> None:
        """Initialize heavy resources lazily.

        This method should be called before using the connector.
        It ensures that resources are only initialized when needed.
        Multiple calls are idempotent.
        """
        if not self._is_warmed_up:
            logger.debug(f"Warming up {self.__class__.__name__}")
            self._initialize_client()
            self._is_warmed_up = True
            logger.debug(f"{self.__class__.__name__} warmed up successfully")

    @abstractmethod
    def _initialize_client(self) -> None:
        """Create underlying client - subclasses must implement.

        This method is called by warm_up() to initialize any
        heavy resources (HTTP clients, connections, etc.).
        """
        pass

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

    def _cleanup_client(self) -> None:
        """Release client resources - override in subclasses if needed."""
        pass

    def close(self) -> None:
        """Release all resources held by the connector."""
        if self._client is not None:
            logger.debug(f"Closing {self.__class__.__name__}")
            self._cleanup_client()
            self._client = None
            self._is_warmed_up = False

    def __enter__(self) -> "BaseConnector":
        """Context manager entry - warm up the connector."""
        self.warm_up()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close the connector."""
        self.close()

    @property
    def is_ready(self) -> bool:
        """Check if the connector is warmed up and ready to use."""
        return self._is_warmed_up
