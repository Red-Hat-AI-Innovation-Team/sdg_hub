# SPDX-License-Identifier: Apache-2.0
"""Base class for code interpreter connectors."""

from abc import abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from ...utils.logger_config import setup_logger
from ..base import BaseConnector

logger = setup_logger(__name__)


class CodeExecutionResult(BaseModel):
    """Result of code execution.

    Attributes
    ----------
    success : bool
        Whether the code executed successfully without errors.
    output : str, optional
        Captured stdout/stderr output from the code.
    error : str, optional
        Error message if execution failed.
    return_value : Any, optional
        Return value from the code execution, if applicable.
    execution_time_ms : float, optional
        Time taken to execute the code in milliseconds.
    """

    success: bool = Field(..., description="Whether execution completed successfully")
    output: Optional[str] = Field(None, description="Captured stdout/stderr output")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    return_value: Any = Field(None, description="Return value from execution")
    execution_time_ms: Optional[float] = Field(
        None, description="Execution time in milliseconds"
    )


class BaseCodeInterpreterConnector(BaseConnector):
    """Base class for code interpreter connectors.

    This class provides a common interface for executing code in sandboxed
    environments. Subclasses implement the actual execution logic for
    specific interpreter backends (Monty, SWE-ReX, etc.).

    Example
    -------
    >>> class MyInterpreter(BaseCodeInterpreterConnector):
    ...     def execute_code(self, code, inputs=None, timeout=None):
    ...         # Execute code safely
    ...         return CodeExecutionResult(success=True, output="Hello")
    ...
    >>> interpreter = MyInterpreter(config=ConnectorConfig())
    >>> result = interpreter.execute_code("print('Hello')")
    """

    @abstractmethod
    def execute_code(
        self,
        code: str,
        inputs: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CodeExecutionResult:
        """Execute code and return structured result.

        Parameters
        ----------
        code : str
            The code to execute.
        inputs : dict, optional
            Input variables to make available to the code.
        timeout : float, optional
            Maximum execution time in seconds. If not provided,
            uses the connector's default timeout.

        Returns
        -------
        CodeExecutionResult
            Structured result containing success status, output, and any errors.
        """
        pass

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute a request (BaseConnector interface).

        Parameters
        ----------
        request : dict
            Request containing 'code' key and optional 'inputs', 'timeout'.

        Returns
        -------
        dict
            Execution result as a dictionary.
        """
        result = self.execute_code(
            code=request["code"],
            inputs=request.get("inputs"),
            timeout=request.get("timeout"),
        )
        return result.model_dump()
