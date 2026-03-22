# SPDX-License-Identifier: Apache-2.0
"""Agent block for integrating external agent frameworks."""

from typing import Any, Optional
import asyncio
import uuid

from pydantic import Field, PrivateAttr
from tqdm import tqdm
import pandas as pd

from ...connectors.agent.base import BaseAgentConnector
from ...connectors.base import ConnectorConfig
from ...connectors.exceptions import ConnectorError
from ...connectors.registry import ConnectorRegistry
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "AgentBlock",
    category="agent",
    description="Execute agent frameworks (Langflow, etc.) on DataFrame rows",
)
class AgentBlock(BaseBlock):
    """Block for executing external agent frameworks on DataFrame rows.

    This block integrates with various agent frameworks through the connector
    system. Each row in the DataFrame is processed by sending messages to the
    agent and storing the response.

    The block supports both sync and async execution modes for optimal
    performance with large datasets.

    Parameters
    ----------
    agent_framework : str
        Name of the connector to use (e.g., 'langflow').
    agent_url : str
        API endpoint URL for the agent.
    agent_api_key : str, optional
        API key for authentication.
    timeout : float
        Request timeout in seconds. Default 120.0.
    max_retries : int
        Maximum retry attempts. Default 3.
    session_id_col : str, optional
        Column containing session IDs. If not provided, generates UUIDs.
    async_mode : bool
        Whether to use async execution. Default False.
    max_concurrency : int
        Maximum concurrent requests in async mode. Default 10.

    Example YAML Configuration
    --------------------------
    ```yaml
    - block_type: AgentBlock
      block_config:
        block_name: my_agent
        agent_framework: langflow
        agent_url: http://localhost:7860/api/v1/run/my-flow
        agent_api_key: ${LANGFLOW_API_KEY}
        input_cols:
          messages: messages_col
        output_cols:
          - agent_response
    ```

    Example
    -------
    >>> block = AgentBlock(
    ...     block_name="qa_agent",
    ...     agent_framework="langflow",
    ...     agent_url="http://localhost:7860/api/v1/run/qa-flow",
    ...     input_cols={"messages": "question"},
    ...     output_cols=["response"],
    ... )
    >>> result_df = block(df)
    """

    block_type: str = "agent"

    # Required configuration
    agent_framework: str = Field(
        ...,
        description="Connector name (e.g., 'langflow')",
    )
    agent_url: str = Field(
        ...,
        description="Agent API endpoint URL",
    )

    # Optional configuration
    agent_api_key: Optional[str] = Field(
        None,
        description="API key for authentication",
    )
    timeout: float = Field(
        120.0,
        description="Request timeout in seconds",
        gt=0,
    )
    max_retries: int = Field(
        3,
        description="Maximum retry attempts",
        ge=0,
    )
    session_id_col: Optional[str] = Field(
        None,
        description="Column containing session IDs",
    )
    async_mode: bool = Field(
        False,
        description="Use async execution for better throughput",
    )
    max_concurrency: int = Field(
        10,
        description="Maximum concurrent requests in async mode",
        gt=0,
    )

    # Private attributes
    _connector: Optional[BaseAgentConnector] = PrivateAttr(default=None)
    _connector_config_key: Optional[tuple] = PrivateAttr(default=None)

    def _get_connector(self) -> BaseAgentConnector:
        """Get or create the connector instance.

        Invalidates the cached connector if the config has changed (e.g., due
        to runtime overrides).

        Returns
        -------
        BaseAgentConnector
            The configured connector instance.
        """
        config_key = (
            self.agent_framework,
            self.agent_url,
            self.agent_api_key,
            self.timeout,
            self.max_retries,
        )
        if self._connector is None or self._connector_config_key != config_key:
            connector_class = ConnectorRegistry.get(self.agent_framework)
            config = ConnectorConfig(
                url=self.agent_url,
                api_key=self.agent_api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            self._connector = connector_class(config=config)
            self._connector_config_key = config_key
        return self._connector

    def _get_messages_col(self) -> str:
        """Get the input column name for messages.

        Returns
        -------
        str
            Column name containing messages.
        """
        if isinstance(self.input_cols, dict):
            if "messages" in self.input_cols:
                return self.input_cols["messages"]
            elif self.input_cols:
                return list(self.input_cols.keys())[0]
            else:
                raise ConnectorError("input_cols must specify the messages column")
        elif isinstance(self.input_cols, list) and len(self.input_cols) > 0:
            return self.input_cols[0]
        else:
            raise ConnectorError("input_cols must specify the messages column")

    def _get_output_col(self) -> str:
        """Get the output column name for responses.

        Returns
        -------
        str
            Column name for storing responses.
        """
        if isinstance(self.output_cols, dict):
            return list(self.output_cols.keys())[0]
        elif isinstance(self.output_cols, list) and len(self.output_cols) > 0:
            return self.output_cols[0]
        else:
            return "agent_response"

    def _build_messages(self, content: Any) -> list[dict[str, Any]]:
        """Build message list from row content.

        Parameters
        ----------
        content : Any
            Content from the DataFrame cell.

        Returns
        -------
        list[dict]
            List of messages in standard format.
        """
        if isinstance(content, list):
            # Already a message list
            return content
        elif isinstance(content, dict):
            # Single message dict
            return [content]
        else:
            # Plain text - wrap as user message
            return [{"role": "user", "content": str(content)}]

    def _get_session_id(self, row: pd.Series, idx: int) -> str:
        """Get session ID for a row.

        Parameters
        ----------
        row : pd.Series
            DataFrame row.
        idx : int
            Row index.

        Returns
        -------
        str
            Session ID.
        """
        if self.session_id_col and self.session_id_col in row:
            return str(row[self.session_id_col])
        return str(uuid.uuid4())

    def _process_row_sync(
        self,
        row: pd.Series,
        idx: int,
        connector: BaseAgentConnector,
        messages_col: str,
    ) -> dict[str, Any]:
        """Process a single row synchronously.

        Parameters
        ----------
        row : pd.Series
            DataFrame row.
        idx : int
            Row index.
        connector : BaseAgentConnector
            Connector instance.
        messages_col : str
            Column containing messages.

        Returns
        -------
        dict
            Response from the agent.
        """
        messages = self._build_messages(row[messages_col])
        session_id = self._get_session_id(row, idx)
        return connector.send(messages, session_id)

    async def _process_row_async(
        self,
        row: pd.Series,
        idx: int,
        connector: BaseAgentConnector,
        messages_col: str,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, dict[str, Any]]:
        """Process a single row asynchronously.

        Parameters
        ----------
        row : pd.Series
            DataFrame row.
        idx : int
            Row index.
        connector : BaseAgentConnector
            Connector instance.
        messages_col : str
            Column containing messages.
        semaphore : asyncio.Semaphore
            Semaphore for concurrency control.

        Returns
        -------
        tuple[int, dict]
            Row index and response.
        """
        async with semaphore:
            messages = self._build_messages(row[messages_col])
            session_id = self._get_session_id(row, idx)
            response = await connector.asend(messages, session_id)
            return idx, response

    async def _process_batch_async(
        self,
        df: pd.DataFrame,
        connector: BaseAgentConnector,
        messages_col: str,
    ) -> dict[int, dict[str, Any]]:
        """Process all rows asynchronously.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        connector : BaseAgentConnector
            Connector instance.
        messages_col : str
            Column containing messages.

        Returns
        -------
        dict[int, dict]
            Mapping from row index to response.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [
            self._process_row_async(row, idx, connector, messages_col, semaphore)
            for idx, row in df.iterrows()
        ]

        results = {}
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"{self.block_name} (async)",
        ):
            idx, response = await coro
            results[idx] = response

        return results

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Process DataFrame rows through the agent.

        Parameters
        ----------
        samples : pd.DataFrame
            Input DataFrame with messages column.
        **kwargs : Any
            Runtime overrides.

        Returns
        -------
        pd.DataFrame
            DataFrame with agent responses added.
        """
        df = samples.copy()
        connector = self._get_connector()
        messages_col = self._get_messages_col()
        output_col = self._get_output_col()

        if self.async_mode:
            # Async execution
            try:
                asyncio.get_running_loop()
                # Already in async context - use thread executor
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._process_batch_async(df, connector, messages_col),
                    )
                    results = future.result()
            except RuntimeError:
                # No event loop - create one
                results = asyncio.run(
                    self._process_batch_async(df, connector, messages_col)
                )

            # Apply results
            df[output_col] = df.index.map(results)
        else:
            # Sync execution with progress bar
            responses = []
            for idx, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc=self.block_name,
            ):
                response = self._process_row_sync(row, idx, connector, messages_col)
                responses.append(response)

            df[output_col] = responses

        logger.info(f"Processed {len(df)} rows with {self.agent_framework} agent")
        return df
