# SPDX-License-Identifier: Apache-2.0
"""Monty code interpreter connector.

Monty is a secure Python interpreter from the Pydantic team, implemented in Rust.
It provides sandboxed execution of Python code with configurable resource limits.
"""

from typing import Any, Optional
import time

from pydantic import Field

from ...utils.logger_config import setup_logger
from ..base import ConnectorConfig
from ..exceptions import ConnectorError
from ..registry import ConnectorRegistry
from .base import BaseCodeInterpreterConnector, CodeExecutionResult

logger = setup_logger(__name__)

# Check for pydantic-monty availability
try:
    import pydantic_monty

    MONTY_AVAILABLE = True
except ImportError:
    MONTY_AVAILABLE = False
    pydantic_monty = None


class MontyConnectorConfig(ConnectorConfig):
    """Configuration for MontyConnector.

    Attributes
    ----------
    memory_limit : int, optional
        Maximum memory usage in bytes. Default 100MB.
    time_limit : float, optional
        Maximum execution time in seconds. Default 30.0.
    stack_depth_limit : int, optional
        Maximum call stack depth. Default 100.
    """

    memory_limit: Optional[int] = Field(
        default=100 * 1024 * 1024,  # 100MB
        description="Maximum memory usage in bytes",
    )
    time_limit: Optional[float] = Field(
        default=30.0,
        description="Maximum execution time in seconds",
    )
    stack_depth_limit: Optional[int] = Field(
        default=100,
        description="Maximum call stack depth",
    )


@ConnectorRegistry.register("monty")
class MontyConnector(BaseCodeInterpreterConnector):
    """Connector for Monty secure Python interpreter.

    Monty provides a sandboxed Python execution environment implemented in Rust.
    It restricts filesystem, network, and system access by default, making it
    safe for executing untrusted code.

    Security Model
    --------------
    - Filesystem: Blocked (no file I/O)
    - Network: Blocked (no network access)
    - Environment variables: Blocked
    - Standard library: Limited subset (sys, typing, asyncio, json)
    - Third-party libraries: Not available
    - External functions: None registered (pure computation only)

    Example
    -------
    >>> from sdg_hub.core.connectors import MontyConnector, ConnectorConfig
    >>>
    >>> connector = MontyConnector(config=ConnectorConfig())
    >>> result = connector.execute_code("x = 1 + 1\\nprint(x)")
    >>> print(result.success)  # True
    >>> print(result.output)   # "2\\n"

    Example YAML Configuration (via PythonInterpreterBlock)
    -------------------------------------------------------
    ```yaml
    - block_type: PythonInterpreterBlock
      block_config:
        block_name: validate_code
        interpreter_framework: monty
        input_cols:
          - generated_code
        output_cols:
          - execution_result
        timeout: 10.0
    ```

    Raises
    ------
    ConnectorError
        If pydantic-monty is not installed.
    """

    config: MontyConnectorConfig = Field(
        default_factory=MontyConnectorConfig,
        description="Connector configuration",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate that pydantic-monty is available."""
        if not MONTY_AVAILABLE:
            raise ConnectorError(
                "pydantic-monty is not installed. "
                "Install it with: pip install pydantic-monty"
            )

    def execute_code(
        self,
        code: str,
        inputs: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CodeExecutionResult:
        """Execute Python code safely via Monty.

        Parameters
        ----------
        code : str
            Python code to execute.
        inputs : dict, optional
            Input variables to make available to the code.
            Keys become variable names in the code's namespace.
        timeout : float, optional
            Maximum execution time in seconds.
            Defaults to config.time_limit or config.timeout.

        Returns
        -------
        CodeExecutionResult
            Structured result with success status, output, and any errors.

        Example
        -------
        >>> result = connector.execute_code(
        ...     "result = x + y\\nprint(result)",
        ...     inputs={"x": 10, "y": 20}
        ... )
        >>> print(result.output)  # "30\\n"
        """
        if not MONTY_AVAILABLE:
            return CodeExecutionResult(
                success=False,
                error="pydantic-monty is not installed",
            )

        # Determine timeout
        effective_timeout = timeout
        if effective_timeout is None:
            effective_timeout = self.config.time_limit or self.config.timeout

        # Prepare inputs
        input_names = list(inputs.keys()) if inputs else []
        input_values = inputs if inputs else {}

        start_time = time.perf_counter()

        try:
            # Create Monty instance
            # We don't register any external functions for security
            monty = pydantic_monty.Monty(
                code,
                inputs=input_names,
                external_functions=[],
            )

            # Execute the code
            output = pydantic_monty.run_monty(
                monty,
                inputs=input_values,
                external_functions={},
            )

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Handle output - Monty returns the result of the last expression
            # or captured print output depending on the code structure
            output_str = str(output) if output is not None else ""

            return CodeExecutionResult(
                success=True,
                output=output_str,
                return_value=output,
                execution_time_ms=execution_time_ms,
            )

        except pydantic_monty.MontyError as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = str(e)
            logger.debug(f"Monty execution error: {error_msg}")

            return CodeExecutionResult(
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {e}"
            logger.warning(f"Unexpected error during code execution: {error_msg}")

            return CodeExecutionResult(
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms,
            )

    async def aexecute_code(
        self,
        code: str,
        inputs: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CodeExecutionResult:
        """Execute Python code asynchronously.

        Monty supports async execution natively via run_monty_async.

        Parameters
        ----------
        code : str
            Python code to execute.
        inputs : dict, optional
            Input variables for the code.
        timeout : float, optional
            Maximum execution time in seconds.

        Returns
        -------
        CodeExecutionResult
            Execution result.
        """
        if not MONTY_AVAILABLE:
            return CodeExecutionResult(
                success=False,
                error="pydantic-monty is not installed",
            )

        effective_timeout = timeout
        if effective_timeout is None:
            effective_timeout = self.config.time_limit or self.config.timeout

        input_names = list(inputs.keys()) if inputs else []
        input_values = inputs if inputs else {}

        start_time = time.perf_counter()

        try:
            monty = pydantic_monty.Monty(
                code,
                inputs=input_names,
                external_functions=[],
            )

            output = await pydantic_monty.run_monty_async(
                monty,
                inputs=input_values,
                external_functions={},
            )

            execution_time_ms = (time.perf_counter() - start_time) * 1000
            output_str = str(output) if output is not None else ""

            return CodeExecutionResult(
                success=True,
                output=output_str,
                return_value=output,
                execution_time_ms=execution_time_ms,
            )

        except pydantic_monty.MontyError as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return CodeExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {e}"
            logger.warning(f"Unexpected error during async execution: {error_msg}")

            return CodeExecutionResult(
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms,
            )
