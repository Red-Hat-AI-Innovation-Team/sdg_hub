# SPDX-License-Identifier: Apache-2.0
"""Python interpreter block for executing code from dataset rows."""

from typing import Any, Optional
import asyncio

from pydantic import Field, PrivateAttr, field_validator
from tqdm import tqdm
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
    async_mode : bool
        Whether to use async execution for better throughput. Default False.
    max_concurrency : int
        Maximum concurrent executions in async mode. Default 10.

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
    async_mode: bool = Field(
        default=False,
        description="Use async execution for better throughput",
    )
    max_concurrency: int = Field(
        default=10,
        description="Maximum concurrent executions in async mode",
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
        """Get or create the interpreter connector instance.

        Returns
        -------
        BaseCodeInterpreterConnector
            The configured interpreter connector.

        Raises
        ------
        ConnectorError
            If the interpreter framework is not found.
        """
        config_key = (self.interpreter_framework, self.timeout)

        if self._connector is None or self._connector_config_key != config_key:
            connector_class = ConnectorRegistry.get(self.interpreter_framework)
            config = ConnectorConfig(timeout=self.timeout)
            self._connector = connector_class(config=config)
            self._connector_config_key = config_key

        return self._connector

    def _execute_row(
        self,
        code: str,
        connector: BaseCodeInterpreterConnector,
    ) -> dict[str, Any]:
        """Execute code for a single row.

        Parameters
        ----------
        code : str
            Code to execute.
        connector : BaseCodeInterpreterConnector
            Connector instance.

        Returns
        -------
        dict
            Execution result as a dictionary.
        """
        if pd.isna(code) or not isinstance(code, str) or not code.strip():
            return CodeExecutionResult(
                success=False,
                error="Empty or invalid code",
            ).model_dump()

        result = connector.execute_code(code, timeout=self.timeout)
        return result.model_dump()

    async def _execute_row_async(
        self,
        code: str,
        idx: int,
        connector: BaseCodeInterpreterConnector,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, dict[str, Any]]:
        """Execute code for a single row asynchronously.

        Parameters
        ----------
        code : str
            Code to execute.
        idx : int
            Row index.
        connector : BaseCodeInterpreterConnector
            Connector instance.
        semaphore : asyncio.Semaphore
            Semaphore for concurrency control.

        Returns
        -------
        tuple[int, dict]
            Row index and execution result.
        """
        async with semaphore:
            if pd.isna(code) or not isinstance(code, str) or not code.strip():
                return idx, CodeExecutionResult(
                    success=False,
                    error="Empty or invalid code",
                ).model_dump()

            # Check if connector supports async
            if hasattr(connector, "aexecute_code"):
                result = await connector.aexecute_code(code, timeout=self.timeout)
            else:
                # Fall back to sync in thread
                result = await asyncio.to_thread(
                    connector.execute_code, code, None, self.timeout
                )
            return idx, result.model_dump()

    async def _process_batch_async(
        self,
        df: pd.DataFrame,
        connector: BaseCodeInterpreterConnector,
        code_col: str,
    ) -> dict[int, dict[str, Any]]:
        """Process all rows asynchronously.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        connector : BaseCodeInterpreterConnector
            Connector instance.
        code_col : str
            Column containing code.

        Returns
        -------
        dict[int, dict]
            Mapping from row index to execution result.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [
            self._execute_row_async(row[code_col], idx, connector, semaphore)
            for idx, row in df.iterrows()
        ]

        results = {}
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"{self.block_name} (async)",
        ):
            idx, result = await coro
            results[idx] = result

        return results

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Execute code from DataFrame rows and capture results.

        Parameters
        ----------
        samples : pd.DataFrame
            Input DataFrame with code column.
        **kwargs : Any
            Runtime overrides.

        Returns
        -------
        pd.DataFrame
            DataFrame with execution results added.
        """
        df = samples.copy()
        connector = self._get_connector()
        code_col = self.input_cols[0]
        output_col = self.output_cols[0]

        if self.async_mode:
            # Async execution
            try:
                asyncio.get_running_loop()
                # Already in async context - use thread executor
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._process_batch_async(df, connector, code_col),
                    )
                    results = future.result()
            except RuntimeError:
                # No event loop - create one
                results = asyncio.run(
                    self._process_batch_async(df, connector, code_col)
                )

            # Apply results
            df[output_col] = df.index.map(results)
        else:
            # Sync execution with progress bar
            execution_results = []
            for idx, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc=self.block_name,
            ):
                result = self._execute_row(row[code_col], connector)
                execution_results.append(result)

            df[output_col] = execution_results

        # Log summary
        success_count = sum(
            1 for r in df[output_col] if isinstance(r, dict) and r.get("success")
        )
        logger.info(
            f"Executed {len(df)} code snippets: "
            f"{success_count} succeeded, {len(df) - success_count} failed"
        )

        return df
