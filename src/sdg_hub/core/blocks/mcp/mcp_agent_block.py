# SPDX-License-Identifier: Apache-2.0
"""MCP Agent Block for LLM agents with remote MCP tools."""

# Standard
from typing import Any, Optional
import asyncio
import json

# Third Party
from litellm import acompletion
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import ConfigDict, Field, SecretStr, field_validator
import pandas as pd

# Local
from ...utils.error_handling import BlockValidationError
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "MCPAgentBlock",
    "mcp",
    "LLM agent with remote MCP tools in an agentic loop",
)
class MCPAgentBlock(BaseBlock):
    """LLM agent block that connects to remote MCP servers for tool use.

    This block runs an agentic loop where an LLM can call tools provided
    by a remote MCP server to answer queries. It connects via streamable HTTP
    to the MCP server, fetches available tools, and iteratively calls the LLM
    until a final response is generated.

    The output is a dictionary containing the full agent trace with all messages,
    tool calls, and tool results - preserving the complete conversation history
    for downstream processing or analysis.

    Parameters
    ----------
    block_name : str
        Name of the block.
    mcp_server_url : str
        URL of the remote MCP server (e.g., "https://mcp.example.com/mcp").
    mcp_headers : Optional[dict[str, str]], optional
        HTTP headers for MCP server authentication.
    model : str
        Model identifier in LiteLLM format (e.g., "openai/gpt-4o").
    api_key : Optional[SecretStr], optional
        API key for the LLM provider. Falls back to environment variables.
    api_base : Optional[str], optional
        Base URL for the LLM API.
    max_iterations : int, optional
        Maximum number of agentic loop iterations, by default 10.
    system_prompt : Optional[str], optional
        System prompt to prepend to conversations.
    input_cols : Union[str, List[str]]
        Input column(s) containing queries to process.
    output_cols : Union[str, List[str]]
        Output column(s) for agent trace dictionaries.

    Output Format
    -------------
    Each output cell contains a dictionary with:
    - messages: list[dict] - Full conversation history including user, assistant,
      and tool messages with all tool calls and results preserved
    - iterations: int - Number of agentic loop iterations completed
    - max_iterations_reached: bool - Whether the loop hit the iteration limit

    Examples
    --------
    >>> block = MCPAgentBlock(
    ...     block_name="research_agent",
    ...     mcp_server_url="https://mcp.deepwiki.com/mcp",
    ...     model="openai/gpt-4o",
    ...     input_cols=["question"],
    ...     output_cols=["agent_trace"],
    ... )
    >>> result = block(dataset)
    >>> # Access the full trace
    >>> trace = result["agent_trace"].iloc[0]
    >>> print(trace["messages"])  # Full conversation history
    >>> print(trace["iterations"])  # Number of iterations
    """

    model_config = ConfigDict(extra="allow")

    block_type: str = "mcp"

    # MCP server configuration
    mcp_server_url: str = Field(..., description="URL of the remote MCP server")
    mcp_headers: Optional[dict[str, str]] = Field(
        None, exclude=True, description="HTTP headers for MCP server authentication"
    )

    # LLM configuration
    model: str = Field(..., description="Model identifier in LiteLLM format")
    api_key: Optional[SecretStr] = Field(
        None, exclude=True, description="API key for the LLM provider"
    )
    api_base: Optional[str] = Field(
        None, exclude=True, description="Base URL for the LLM API"
    )

    # Agent configuration
    max_iterations: int = Field(
        10, description="Maximum number of agentic loop iterations"
    )
    system_prompt: Optional[str] = Field(
        None, description="System prompt to prepend to conversations"
    )

    @field_validator("input_cols")
    @classmethod
    def validate_single_input_col(cls, v):
        """Ensure exactly one input column."""
        if isinstance(v, str):
            return [v]
        if isinstance(v, list) and len(v) == 1:
            return v
        if isinstance(v, list) and len(v) != 1:
            raise ValueError(
                f"MCPAgentBlock expects exactly one input column, got {len(v)}: {v}"
            )
        raise ValueError(f"Invalid input_cols format: {v}")

    @field_validator("output_cols")
    @classmethod
    def validate_single_output_col(cls, v):
        """Ensure exactly one output column."""
        if isinstance(v, str):
            return [v]
        if isinstance(v, list) and len(v) == 1:
            return v
        if isinstance(v, list) and len(v) != 1:
            raise ValueError(
                f"MCPAgentBlock expects exactly one output column, got {len(v)}: {v}"
            )
        raise ValueError(f"Invalid output_cols format: {v}")

    @field_validator("max_iterations")
    @classmethod
    def validate_max_iterations(cls, v):
        """Ensure max_iterations is positive."""
        if v < 1:
            raise ValueError(f"max_iterations must be at least 1, got {v}")
        return v

    def model_post_init(self, __context) -> None:
        """Initialize after Pydantic validation."""
        super().model_post_init(__context)
        logger.info(
            "Initialized MCPAgentBlock '%s' with model '%s' and MCP server '%s'",
            self.block_name,
            self.model,
            self.mcp_server_url,
            extra={
                "block_name": self.block_name,
                "model": self.model,
                "mcp_server_url": self.mcp_server_url,
                "max_iterations": self.max_iterations,
            },
        )

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate responses using LLM with MCP tools.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset containing the query column.
        **kwargs : Any
            Runtime parameters that override initialization defaults.

        Returns
        -------
        pd.DataFrame
            Dataset with agent trace dictionaries added to the output column.
            Each trace contains 'messages' (full conversation), 'iterations',
            and 'max_iterations_reached'.

        Raises
        ------
        BlockValidationError
            If required configuration is missing.
        """
        # input_cols is guaranteed to be list[str] by validator
        input_col = self.input_cols[0]  # type: ignore[index]
        queries = samples[input_col].tolist()

        logger.info(
            "Starting MCP agent generation for %d samples",
            len(queries),
            extra={
                "block_name": self.block_name,
                "model": self.model,
                "batch_size": len(queries),
            },
        )

        # Run agent for each query
        # Check if there's a running event loop
        has_running_loop = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            has_running_loop = False

        if not has_running_loop:
            # No running loop; safe to create one
            responses = asyncio.run(self._generate_all(queries))
        else:
            # Check if nest_asyncio is applied
            nest_asyncio_applied = (
                hasattr(loop, "_nest_patched")
                or getattr(asyncio.run, "__module__", "") == "nest_asyncio"
            )
            if nest_asyncio_applied:
                responses = asyncio.run(self._generate_all(queries))
            else:
                raise BlockValidationError(
                    f"MCPAgentBlock cannot be used from within a running event loop for '{self.block_name}'. "
                    "Use an async entrypoint or apply nest_asyncio.apply() in notebook environments."
                )

        logger.info(
            "MCP agent generation completed for %d samples",
            len(responses),
            extra={
                "block_name": self.block_name,
                "model": self.model,
                "batch_size": len(responses),
            },
        )

        result = samples.copy()
        # output_cols is guaranteed to be list[str] by validator
        output_col = self.output_cols[0]  # type: ignore[index]
        result[output_col] = responses
        return result

    async def _generate_all(self, queries: list[str]) -> list[dict[str, Any]]:
        """Generate responses for all queries.

        Parameters
        ----------
        queries : list[str]
            List of queries to process.

        Returns
        -------
        list[dict[str, Any]]
            List of agent trace dictionaries containing the full conversation.
        """
        responses: list[dict[str, Any]] = []
        for i, query in enumerate(queries):
            try:
                response = await self._run_agent(query)
                responses.append(response)
                logger.debug(
                    "Generated response %d/%d",
                    i + 1,
                    len(queries),
                    extra={
                        "block_name": self.block_name,
                        "query_index": i,
                    },
                )
            except Exception as e:
                logger.error(
                    "Failed to generate response for query %d: %s",
                    i,
                    str(e),
                    extra={
                        "block_name": self.block_name,
                        "query_index": i,
                        "error": str(e),
                    },
                )
                raise
        return responses

    async def _run_agent(self, query: str) -> dict[str, Any]:
        """Run the agentic loop for a single query.

        Parameters
        ----------
        query : str
            The user query to process.

        Returns
        -------
        dict[str, Any]
            The full agent trace containing messages, tool calls, and results.
        """
        headers = self.mcp_headers or {}

        async with streamablehttp_client(self.mcp_server_url, headers=headers) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Get available tools from MCP server
                tools_response = await session.list_tools()
                tools = self._to_openai_format(tools_response.tools)

                logger.debug(
                    "Retrieved %d tools from MCP server",
                    len(tools),
                    extra={
                        "block_name": self.block_name,
                        "tool_count": len(tools),
                    },
                )

                # Build initial messages
                messages: list[dict[str, Any]] = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": query})

                # Build completion kwargs
                completion_kwargs = self._build_completion_kwargs()

                # Track iterations for the trace
                iterations_completed = 0

                # Agentic loop
                for iteration in range(self.max_iterations):
                    iterations_completed = iteration + 1
                    logger.debug(
                        "Agent iteration %d/%d",
                        iterations_completed,
                        self.max_iterations,
                        extra={
                            "block_name": self.block_name,
                            "iteration": iterations_completed,
                        },
                    )

                    response = await acompletion(
                        messages=messages,
                        tools=tools if tools else None,
                        **completion_kwargs,
                    )

                    msg = response.choices[0].message

                    # Append assistant message to history
                    messages.append(msg.model_dump())

                    # Check if done (no tool calls)
                    if not msg.tool_calls:
                        return {
                            "messages": messages,
                            "iterations": iterations_completed,
                            "max_iterations_reached": False,
                        }

                    # Execute tool calls
                    for tc in msg.tool_calls:
                        tool_name = tc.function.name
                        try:
                            tool_args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            tool_args = {}

                        logger.debug(
                            "Calling tool '%s'",
                            tool_name,
                            extra={
                                "block_name": self.block_name,
                                "tool_name": tool_name,
                            },
                        )

                        try:
                            result = await session.call_tool(tool_name, tool_args)
                            # Serialize tool result to string for message content
                            result_content = self._serialize_tool_result(result)
                        except Exception as e:
                            logger.warning(
                                "Tool call '%s' failed: %s",
                                tool_name,
                                str(e),
                                extra={
                                    "block_name": self.block_name,
                                    "tool_name": tool_name,
                                    "error": str(e),
                                },
                            )
                            result_content = json.dumps({"error": str(e)})

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_content,
                            }
                        )

                # Max iterations reached
                logger.warning(
                    "Max iterations (%d) reached without final response",
                    self.max_iterations,
                    extra={
                        "block_name": self.block_name,
                        "max_iterations": self.max_iterations,
                    },
                )

                return {
                    "messages": messages,
                    "iterations": iterations_completed,
                    "max_iterations_reached": True,
                }

    def _to_openai_format(self, mcp_tools: list[Any]) -> list[dict[str, Any]]:
        """Convert MCP tools to OpenAI function calling format.

        Parameters
        ----------
        mcp_tools : list[Any]
            Tools from MCP server.

        Returns
        -------
        list[dict[str, Any]]
            Tools in OpenAI format.
        """
        openai_tools = []
        for tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                    if hasattr(tool, "inputSchema")
                    else {},
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools

    def _serialize_tool_result(self, result: Any) -> str:
        """Serialize MCP tool result to JSON string.

        Parameters
        ----------
        result : Any
            Result from MCP tool call.

        Returns
        -------
        str
            JSON string representation of the result.
        """
        if hasattr(result, "content"):
            contents = result.content
            if isinstance(contents, list):
                serialized: list[Any] = []
                for content in contents:
                    if hasattr(content, "text"):
                        serialized.append({"type": "text", "text": content.text})
                    elif hasattr(content, "model_dump"):
                        serialized.append(content.model_dump())
                    else:
                        serialized.append(str(content))
                return json.dumps(serialized)
            if hasattr(contents, "model_dump"):
                return json.dumps(contents.model_dump())
            return json.dumps(str(contents))
        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump())
        return json.dumps(str(result))

    def _build_completion_kwargs(self) -> dict[str, Any]:
        """Build kwargs for LiteLLM completion call.

        Returns
        -------
        dict[str, Any]
            Kwargs for litellm.acompletion().
        """
        completion_kwargs: dict[str, Any] = {
            "model": self.model,
        }

        if self.api_key is not None:
            completion_kwargs["api_key"] = self.api_key.get_secret_value()
        if self.api_base is not None:
            completion_kwargs["api_base"] = self.api_base

        return completion_kwargs

    def __repr__(self) -> str:
        """String representation of the block."""
        return (
            f"MCPAgentBlock(name='{self.block_name}', model='{self.model}', "
            f"mcp_server_url='{self.mcp_server_url}', max_iterations={self.max_iterations})"
        )
