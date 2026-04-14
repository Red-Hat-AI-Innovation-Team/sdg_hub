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
    pydantic_monty = None  # type: ignore[assignment]


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

    config: ConnectorConfig = Field(
        default_factory=lambda: ConnectorConfig(),  # type: ignore[call-arg]
        description="Connector configuration",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate that pydantic-monty is available."""
        if not MONTY_AVAILABLE:
            raise ConnectorError(
                "pydantic-monty is not installed. "
                "Install it with: uv pip install '.[code]'"
            )

    def _get_resource_limits(
        self, timeout: Optional[float] = None
    ) -> "pydantic_monty.ResourceLimits":
        """Build ResourceLimits for Monty execution.

        Parameters
        ----------
        timeout : float, optional
            Timeout override. Falls back to config.timeout (default 120s).

        Returns
        -------
        pydantic_monty.ResourceLimits
            Resource limits object for Monty execution.
        """
        effective_timeout = timeout if timeout is not None else self.config.timeout
        return pydantic_monty.ResourceLimits(max_duration_secs=effective_timeout)

    def _build_result(
        self, stdout: str, return_value: Any, start_time: float
    ) -> CodeExecutionResult:
        """Build success result with timing.

        Parameters
        ----------
        stdout : str
            Captured printed output from the code.
        return_value : Any
            The evaluated return value from the Monty runtime.
        start_time : float
            perf_counter timestamp from before execution.
        """
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        return CodeExecutionResult(
            success=True,
            output=stdout if stdout else None,
            error=None,
            return_value=return_value,
            execution_time_ms=execution_time_ms,
        )

    def _build_error(
        self, error: Exception, start_time: float, *, log_warning: bool = False
    ) -> CodeExecutionResult:
        """Build error result with timing."""
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        if isinstance(error, pydantic_monty.MontyError):
            error_msg = str(error)
            logger.debug(f"Monty execution error: {error_msg}")
        else:
            error_msg = f"{type(error).__name__}: {error}"
            if log_warning:
                logger.warning(f"Unexpected error during code execution: {error_msg}")
        return CodeExecutionResult(
            success=False,
            output=None,
            error=error_msg,
            return_value=None,
            execution_time_ms=execution_time_ms,
        )

    def _run_monty(
        self,
        code: str,
        inputs: Optional[dict[str, Any]],
        timeout: Optional[float],
    ) -> tuple["pydantic_monty.Monty", dict]:
        """Create Monty instance and build run kwargs."""
        input_names = list(inputs.keys()) if inputs else None
        monty = pydantic_monty.Monty(code, inputs=input_names)
        run_kwargs: dict[str, Any] = {"limits": self._get_resource_limits(timeout)}
        if inputs:
            run_kwargs["inputs"] = inputs
        return monty, run_kwargs

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
            Maximum execution time in seconds. Defaults to config.timeout.

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
        start_time = time.perf_counter()

        try:
            monty, run_kwargs = self._run_monty(code, inputs, timeout)
            captured: list[str] = []
            run_kwargs["print_callback"] = lambda _file, content: captured.append(
                content
            )
            return_value = monty.run(**run_kwargs)
            stdout = "".join(captured)
            return self._build_result(stdout, return_value, start_time)

        except Exception as e:
            return self._build_error(e, start_time, log_warning=True)

    async def aexecute_code(
        self,
        code: str,
        inputs: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CodeExecutionResult:
        """Execute Python code asynchronously.

        Monty supports async execution natively via run_async.

        Parameters
        ----------
        code : str
            Python code to execute.
        inputs : dict, optional
            Input variables for the code.
        timeout : float, optional
            Maximum execution time in seconds. Defaults to config.timeout.

        Returns
        -------
        CodeExecutionResult
            Execution result.
        """
        start_time = time.perf_counter()

        try:
            monty, run_kwargs = self._run_monty(code, inputs, timeout)
            captured: list[str] = []
            run_kwargs["print_callback"] = lambda _file, content: captured.append(
                content
            )
            return_value = await monty.run_async(**run_kwargs)
            stdout = "".join(captured)
            return self._build_result(stdout, return_value, start_time)

        except Exception as e:
            return self._build_error(e, start_time, log_warning=True)
