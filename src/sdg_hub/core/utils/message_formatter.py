"""Convert Langflow agent tool traces into structured tool-calling conversations.

The output format uses the modern OpenAI tool_calls/tool message format,
compatible with training pipelines for tool-use models (system with tool
declarations, user, assistant tool_calls / tool response pairs, and final
assistant answer).
"""

from __future__ import annotations

from typing import Any
import json
import uuid


def _build_system_message(tool_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the system message containing tool declarations.

    Parameters
    ----------
    tool_list : list[dict[str, Any]]
        Tool schemas with ``name``, ``description``, and ``inputSchema`` keys.

    Returns
    -------
    dict[str, Any]
        System message dict with tool declarations in the content.
    """
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
    return {
        "role": "system",
        "content": f"<|im_system|>tool_declare<|im_middle|>{tools_json}<|im_end|>",
    }


def _handle_text_step(step: dict[str, Any]) -> dict[str, Any]:
    """Convert a text trace step into a user or assistant message.

    Parameters
    ----------
    step : dict[str, Any]
        A trace step with ``type == "text"``.

    Returns
    -------
    dict[str, Any]
        A message dict with role "user" (for Input) or "assistant" (otherwise).
    """
    title = step.get("header", {}).get("title", "")
    text = step.get("text", "")
    role = "user" if title == "Input" else "assistant"
    return {"role": role, "content": text}


def _handle_tool_use_step(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a tool_use trace step into assistant + tool message pair.

    Parameters
    ----------
    step : dict[str, Any]
        A trace step with ``type == "tool_use"``.

    Returns
    -------
    list[dict[str, Any]]
        A two-element list: the assistant tool_calls message and the tool response.
    """
    name = step.get("name", "")
    tool_input = step.get("tool_input", {})
    output = step.get("output")

    tool_call_id = f"call_{uuid.uuid4().hex[:24]}"

    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(tool_input),
                },
            }
        ],
    }

    tool_msg = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": _extract_tool_output(output),
    }

    return [assistant_msg, tool_msg]


def tool_trace_to_messages(
    tool_trace: list[dict[str, Any]],
    tool_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert a Langflow tool trace and tool schemas into a structured
    tool-calling conversation.

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
        A conversation in the tool-calling message format::

            [
                {"role": "system",    "content": "<tool declarations>"},
                {"role": "user",      "content": "question"},
                {"role": "assistant", "content": null, "tool_calls": [...]},
                {"role": "tool",      "content": "result", "tool_call_id": "...", "name": "..."},
                ...
                {"role": "assistant", "content": "final answer"},
            ]
    """
    messages: list[dict[str, Any]] = [_build_system_message(tool_list)]

    for step in tool_trace:
        step_type = step.get("type")

        if step_type == "text":
            messages.append(_handle_text_step(step))
        elif step_type == "tool_use":
            messages.extend(_handle_tool_use_step(step))

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
