# SPDX-License-Identifier: Apache-2.0
"""MCP blocks for agent-based data generation with Model Context Protocol tools.

This module provides blocks for running agentic loops that connect to remote
MCP servers, enabling LLMs to use external tools during data generation.
"""

from .mcp_agent_block import MCPAgentBlock

__all__ = [
    "MCPAgentBlock",
]
