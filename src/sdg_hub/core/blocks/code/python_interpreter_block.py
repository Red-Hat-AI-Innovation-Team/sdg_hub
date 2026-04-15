# SPDX-License-Identifier: Apache-2.0
"""Python interpreter block for executing code from dataset rows."""

from typing import Any, Optional, cast
import asyncio

from pydantic import Field, PrivateAttr, field_validator
from tqdm.asyncio import tqdm_asyncio
import pandas as pd

from ...connectors.base import ConnectorConfig
from ...connectors.code_interpreter.base import (
    BaseCodeInterpreterConnector,
    CodeExecutionResult,
)
from ...connectors.registry import ConnectorRegistry
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "PythonInterpreterBlock",
    category="code",
    description="Execute Python code from dataset rows and capture results",
)
class PythonInterpreterBlock(BaseBlock):
    """Block for executing Python code from DataFrame rows.

    This block integrates with code interpreter connectors to safely execute
    Python code stored in dataset columns. It's designed for validating
    synthetic code datasets by testing whether generated code runs successfully.

    The block reads code from input_cols[0] and writes a structured result
    dict to output_cols[0] containing success status, output, and any errors.

    Parameters
    ----------
    input_cols : list[str]
        Single-element list with the column name containing code to execute.
    output_cols : list[str]
        Single-element list with the column name for execution results.
    interpreter_framework : str
        Name of the interpreter connector to use. Default is 'monty'.
    timeout : float
        Maximum execution time per code snippet in seconds. Default 30.0.
    max_concurrency : int
        Maximum concurrent executions. Default 10.

    Example YAML Configuration
    --------------------------
    ```yaml
    - block_type: PythonInterpreterBlock
      block_config:
        block_name: validate_generated_code
        interpreter_framework: monty
        input_cols:
          - generated_code
        output_cols:
          - execution_result
        timeout: 10.0
    ```

    Example
    -------
    >>> block = PythonInterpreterBlock(
    ...     block_name="validate_code",
    ...     input_cols=["code"],
    ...     output_cols=["result"],
    ...     timeout=5.0,
    ... )
    >>> df = pd.DataFrame({"code": ["print('Hello')", "1/0"]})
    >>> result = block(df)
    >>> print(result["result"].iloc[0])
    # {'success': True, 'output': 'Hello\\n', 'error': None, ...}
    >>> print(result["result"].iloc[1])
    # {'success': False, 'output': None, 'error': 'ZeroDivisionError: ...', ...}

    Output Format
    -------------
    Each row receives a dict in output_cols[0] with:
    - success: bool - Whether code executed without errors
    - output: str | None - Captured stdout/print output
    - error: str | None - Error message if execution failed
    - return_value: Any | None - Return value from execution
    - execution_time_ms: float | None - Execution time in milliseconds

    Additionally, a flat boolean column ``{output_cols[0]}_success`` is created
    for convenient downstream filtering (e.g., with ColumnValueFilterBlock).
    """

    interpreter_framework: str = Field(
        default="monty",
        description="Code interpreter connector to use (e.g., 'monty')",
    )
    timeout: float = Field(
        default=30.0,
        description="Maximum execution time per code snippet in seconds",
        gt=0,
    )
    max_concurrency: int = Field(
        default=10,
        description="Maximum concurrent executions",
        gt=0,
    )

    # Private attributes
    _connector: Optional[BaseCodeInterpreterConnector] = PrivateAttr(default=None)
    _connector_config_key: Optional[tuple] = PrivateAttr(default=None)

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_single_input_col(cls, v):
        """Validate that exactly one input column is specified."""
        if not isinstance(v, list) or len(v) != 1:
            raise ValueError("input_cols must be a list with exactly one column name")
        return v

    @field_validator("output_cols", mode="after")
    @classmethod
    def validate_single_output_col(cls, v):
        """Validate that exactly one output column is specified."""
        if not isinstance(v, list) or len(v) != 1:
            raise ValueError("output_cols must be a list with exactly one column name")
        return v

    def _get_connector(self) -> BaseCodeInterpreterConnector:
        """Get or create the interpreter connector instance."""
        config_key = (self.interpreter_framework, self.timeout)

        if self._connector is None or self._connector_config_key != config_key:
            connector_class = ConnectorRegistry.get(self.interpreter_framework)
            self._connector = cast(
                BaseCodeInterpreterConnector,
                connector_class(
                    config=ConnectorConfig()  # type: ignore[call-arg]
                ),
            )
            self._connector_config_key = config_key
        assert self._connector is not None
        return self._connector

    async def _execute_one(
        self,
        code: str,
        connector: BaseCodeInterpreterConnector,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        """Execute code for a single row."""
        async with semaphore:
            if pd.isna(code) or not isinstance(code, str) or not code.strip():
                return CodeExecutionResult(
                    success=False,
                    output=None,
                    error="Empty or invalid code",
                    return_value=None,
                    execution_time_ms=None,
                ).model_dump()

            try:
                if hasattr(connector, "aexecute_code"):
                    result = await connector.aexecute_code(code, timeout=self.timeout)
                else:
                    result = await asyncio.to_thread(
                        connector.execute_code,
                        code,
                        inputs=None,
                        timeout=self.timeout,
                    )
                return result.model_dump()
            except Exception as e:
                return CodeExecutionResult(
                    success=False,
                    output=None,
                    error=f"Execution error: {e}",
                    return_value=None,
                    execution_time_ms=None,
                ).model_dump()

    async def _execute_all(
        self,
        codes: list[str],
        connector: BaseCodeInterpreterConnector,
    ) -> list[dict[str, Any]]:
        """Execute all code snippets concurrently."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._execute_one(code, connector, semaphore) for code in codes]
        results = await tqdm_asyncio.gather(
            *tasks,
            desc=self.block_name,
        )
        return results

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Execute code from DataFrame rows and capture results."""
        df = samples.copy()
        connector = self._get_connector()
        # Validators guarantee these are single-element lists
        code_col = cast(list[str], self.input_cols)[0]
        output_col = cast(list[str], self.output_cols)[0]

        if code_col not in df.columns:
            raise ValueError(
                f"Code column '{code_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        codes = df[code_col].tolist()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in async context — run in a dedicated thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run,
                    self._execute_all(codes, connector),
                )
                results = future.result()
        else:
            results = asyncio.run(self._execute_all(codes, connector))

        df[output_col] = results
        # Add a flat boolean column for easy filtering in downstream blocks
        df[f"{output_col}_success"] = [r.get("success", False) for r in results]

        # Log summary
        success_count = sum(1 for r in results if r.get("success"))
        logger.info(
            f"Executed {len(df)} code snippets: "
            f"{success_count} succeeded, {len(df) - success_count} failed"
        )

        return df
