"""Extract tool schemas from the ShopInsights MCP server and save as a HuggingFace Dataset.

Usage:
    uv run python examples/agentic/ecommerce_mcp/create_dataset.py [--output DIR]

This imports the server module directly (no running server needed) and creates
a single-row dataset with columns expected by the Toucan flow:
  - tool_list: list of tool dicts (name, description, inputSchema)
  - mcp_server_name: server name string
  - mcp_server_description: server description string
"""

from __future__ import annotations

from pathlib import Path
import argparse
import asyncio
import json
import sys

# Ensure we can import from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _extract_tools() -> tuple[list[dict], str, str]:
    """Import the MCP server and extract tool schemas."""
    from server import mcp

    async def _get():
        tools = await mcp.list_tools()
        return tools

    raw_tools = asyncio.run(_get())

    tool_list = []
    for t in raw_tools:
        mcp_tool = t.to_mcp_tool()
        tool_list.append(
            {
                "name": mcp_tool.name,
                "description": mcp_tool.description or "",
                "inputSchema": mcp_tool.inputSchema,
            }
        )

    server_name = mcp.name or "ShopInsights Analytics Platform"
    server_description = (
        "E-commerce analytics platform for an online retailer. "
        "Provides product search, sales analytics, customer insights, "
        "demand forecasting, and promotional management. "
        "Features 15 tools organized across product discovery, sales & revenue, "
        "customer analytics, and multi-step analytical workflows."
    )

    return tool_list, server_name, server_description


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create tool-schema dataset for Toucan flow"
    )
    parser.add_argument(
        "--output",
        default="./ecommerce_dataset",
        help="Output directory for the HuggingFace Dataset (default: ./ecommerce_dataset)",
    )
    args = parser.parse_args()

    from datasets import Dataset

    tool_list, server_name, server_description = _extract_tools()

    # The flow expects each row to contain these columns
    ds = Dataset.from_dict(
        {
            "tool_list": [tool_list],
            "mcp_server_name": [server_name],
            "mcp_server_description": [server_description],
        }
    )

    output_path = Path(args.output)
    ds.save_to_disk(str(output_path))

    # Print summary
    print(f"Dataset saved to {output_path.resolve()}")
    print(f"  Rows: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    print(f"  Tools ({len(tool_list)}):")
    for t in tool_list:
        desc = t["description"].split("\n")[0][:60]
        n_params = len(t["inputSchema"].get("properties", {}))
        print(f"    - {t['name']} ({n_params} params): {desc}")

    # Also save a human-readable JSON for debugging
    json_path = output_path / "tools_summary.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "server_name": server_name,
                "server_description": server_description,
                "tool_count": len(tool_list),
                "tools": [
                    {
                        "name": t["name"],
                        "description": t["description"].split("\n")[0],
                        "param_count": len(t["inputSchema"].get("properties", {})),
                        "params": list(t["inputSchema"].get("properties", {}).keys()),
                    }
                    for t in tool_list
                ],
            },
            f,
            indent=2,
        )
    print(f"\n  Tool summary JSON: {json_path}")


if __name__ == "__main__":
    main()
