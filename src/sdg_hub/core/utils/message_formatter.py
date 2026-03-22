"""Convert Langflow agent tool traces into structured function-calling conversations.

The output format is compatible with standard SFT training pipelines for
tool-use models (system with tool declarations, user, assistant function_call /
function response pairs, and final assistant answer).
"""

from __future__ import annotations

from typing import Any
import json


def tool_trace_to_messages(
    tool_trace: list[dict[str, Any]],
    tool_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert a Langflow tool trace and tool schemas into a structured
    function-calling conversation.

    Parameters
    ----------
    tool_trace : list[dict]
        The ``content_blocks`` contents extracted from a Langflow agent
        response (via ``AgentResponseExtractorBlock`` with
        ``extract_tool_trace=True``).  Each entry is one of:

        - ``{"type": "text", "header": {"title": "Input"}, "text": "..."}``
        - ``{"type": "tool_use", "name": "...", "tool_input": {...}, "output": ...}``
        - ``{"type": "text", "header": {"title": "Output"}, "text": "..."}``

    tool_list : list[dict]
        Tool schemas, each with ``name``, ``description``, and
        ``inputSchema`` keys (the format produced by
        ``create_dataset.py`` / the MCP tool listing).

    Returns
    -------
    list[dict]
        A conversation in the function-calling message format::

            [
                {"role": "system",    "content": "<tool declarations>"},
                {"role": "user",      "content": "question"},
                {"role": "assistant", "content": "", "function_call": {"name": ..., "arguments": ...}},
                {"role": "function",  "content": "result JSON", "name": "tool_name"},
                ...
                {"role": "assistant", "content": "final answer"},
            ]
    """
    # -- system message with tool declarations --------------------------------
    tools_json = json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"],
                },
            }
            for t in tool_list
        ]
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                f"<|im_system|>tool_declare<|im_middle|>{tools_json}<|im_end|>"
            ),
        }
    ]

    # -- walk through each trace step -----------------------------------------
    for step in tool_trace:
        step_type = step.get("type")

        if step_type == "text":
            title = step.get("header", {}).get("title", "")
            text = step.get("text", "")
            if title == "Input":
                messages.append({"role": "user", "content": text})
            elif title == "Output":
                messages.append({"role": "assistant", "content": text})
            else:
                messages.append({"role": "assistant", "content": text})

        elif step_type == "tool_use":
            name = step.get("name", "")
            tool_input = step.get("tool_input", {})
            output = step.get("output")

            # assistant decides to call a tool
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": name,
                        "arguments": json.dumps(tool_input),
                    },
                }
            )

            # tool returns its result
            result_text = _extract_tool_output(output)
            messages.append({"role": "function", "content": result_text, "name": name})

    return messages


def _extract_tool_output(output: Any) -> str:
    """Normalise a Langflow tool output into a JSON string."""
    if isinstance(output, dict):
        content_list = output.get("content", [])
        if content_list and isinstance(content_list, list):
            return content_list[0].get("text", "")
        return json.dumps(output.get("structuredContent", output))
    if isinstance(output, str):
        return output
    return json.dumps(output) if output else ""
