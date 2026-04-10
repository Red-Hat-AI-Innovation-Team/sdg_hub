"""Utilities for MCP evaluation benchmark.

Trace normalization, formatting, and programmatic metrics shared by
both ``generate.ipynb`` and ``evaluate.ipynb``.
"""

import json

# ── Trace normalization ──────────────────────────────────────────────
# Convert framework-specific traces (LangGraph, Langflow) into a
# canonical format: list of {"name": ..., "input": ..., "output": ...}.


def normalize_tool_trace(raw_trace: list[dict] | str) -> list[dict]:
    """Normalize a Langflow or LangGraph tool trace to canonical format.

    Canonical format: ``[{"name": ..., "input": ..., "output": ...}, ...]``

    Handles both:
    - LangGraph: ``{"type": "tool_use", "tool_calls": [...]}`` + ``{"type": "tool_result", ...}``
    - Langflow:  ``{"type": "tool_use", "name": ..., "tool_input": ..., "output": ...}``

    Strips UI metadata (duration, header, icon) and text-type entries.
    """
    if isinstance(raw_trace, str):
        raw_trace = json.loads(raw_trace)
    if not isinstance(raw_trace, list):
        return []

    cleaned: list[dict] = []
    pending: dict[str, dict] = {}  # tool_call id -> call dict
    for entry in raw_trace:
        if not isinstance(entry, dict):
            continue

        if entry.get("type") == "tool_use":
            # LangGraph format
            if "tool_calls" in entry:
                for tc in entry["tool_calls"]:
                    call: dict = {
                        "name": tc.get("name", ""),
                        "input": tc.get("args", {}),
                    }
                    tc_id = tc.get("id")
                    if tc_id:
                        pending[tc_id] = call
                    cleaned.append(call)
            # Langflow format
            elif "name" in entry:
                step: dict = {
                    "name": entry["name"],
                    "input": entry.get("tool_input", {}),
                }
                if entry.get("output"):
                    step["output"] = entry["output"]
                cleaned.append(step)

        elif entry.get("type") == "tool_result":
            # Match by tool_call id if available, else fall back to last call
            tc_id = entry.get("tool_call_id") or entry.get("id")
            if tc_id and tc_id in pending:
                pending[tc_id]["output"] = entry.get("content", "")
            elif cleaned and "output" not in cleaned[-1]:
                cleaned[-1]["output"] = entry.get("content", "")

    return cleaned


def extract_tool_names(trace: list[dict]) -> list[str]:
    """Extract ordered list of tool names from a canonical trace."""
    return [step["name"] for step in trace if "name" in step]


# ── Trace formatting ────────────────────────────────────────────────


def format_trace_for_judge(
    tool_trace: list[dict],
    max_args_len: int = 300,
    max_output_len: int = 200,
) -> str:
    """Format a canonical tool trace into a readable string for the judge prompt.

    Each tool call is formatted as::

      [N] tool_name({"arg": "value"})
          -> tool output (truncated)
    """
    if not tool_trace:
        return "  No tool calls made."
    lines = []
    for i, step in enumerate(tool_trace, 1):
        args_str = json.dumps(step.get("input", {}), ensure_ascii=False)
        if len(args_str) > max_args_len:
            args_str = args_str[:max_args_len] + "..."
        line = f"  [{i}] {step['name']}({args_str})"
        output = step.get("output")
        if output:
            out_str = str(output)
            if len(out_str) > max_output_len:
                out_str = out_str[:max_output_len] + "..."
            line += f"\n      -> {out_str}"
        lines.append(line)
    return "\n".join(lines)


# ── Programmatic tool metrics ────────────────────────────────────────


def compute_tool_metrics(
    model_tools: list[str],
    expert_tools: list[str],
    model_trace: list[dict] | None = None,
    expert_trace: list[dict] | None = None,
) -> dict[str, float]:
    """Compute tool recall, precision, order match, and parameter similarity.

    Returns dict with keys: tool_recall, tool_precision, order_match, param_match.
    """
    model_set, expert_set = set(model_tools), set(expert_tools)
    if not expert_set:
        return {
            "tool_recall": 1.0,
            "tool_precision": 1.0,
            "order_match": 1.0,
            "param_match": 1.0,
        }

    intersection = model_set & expert_set
    recall = len(intersection) / len(expert_set)
    precision = len(intersection) / len(model_set) if model_set else 0.0

    # LCS for order match
    m, n = len(model_tools), len(expert_tools)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                dp[i - 1][j - 1] + 1
                if model_tools[i - 1] == expert_tools[j - 1]
                else max(dp[i - 1][j], dp[i][j - 1])
            )
    order = dp[m][n] / len(expert_tools)

    # Parameter match: compare arguments by occurrence order.
    # Each model call is consumed once to handle repeated tool names correctly.
    param_match = 0.0
    if model_trace and expert_trace:
        matched, total = 0, 0
        available = list(range(len(model_trace)))  # indices not yet consumed
        for et in expert_trace:
            for idx in available:
                mt = model_trace[idx]
                if mt.get("name") == et.get("name"):
                    total += 1
                    available.remove(idx)  # consume this match
                    e_in = et.get("input", {})
                    m_in = mt.get("input", {})
                    if not e_in and not m_in:
                        matched += 1
                    elif isinstance(e_in, dict) and isinstance(m_in, dict):
                        e_keys = set(e_in.keys())
                        m_keys = set(m_in.keys())
                        if e_keys:
                            key_ov = len(e_keys & m_keys) / len(e_keys)
                            val_m = sum(
                                1
                                for k in e_keys & m_keys
                                if str(e_in[k]).lower() == str(m_in[k]).lower()
                            )
                            val_r = (
                                val_m / len(e_keys & m_keys) if (e_keys & m_keys) else 0
                            )
                            matched += (key_ov + val_r) / 2
                    break
        param_match = matched / total if total > 0 else 0.0

    return {
        "tool_recall": round(recall, 3),
        "tool_precision": round(precision, 3),
        "order_match": round(order, 3),
        "param_match": round(param_match, 3),
    }


# ── Zero scores for failures ────────────────────────────────────────

ZERO_JUDGE = {
    "task_fulfillment": 0,
    "grounding": 0,
    "tool_appropriateness": 0,
    "parameter_accuracy": 0,
    "dependency_awareness": 0,
    "parallelism_and_efficiency": 0,
}

ZERO_METRICS = {
    "tool_recall": 0.0,
    "tool_precision": 0.0,
    "order_match": 0.0,
    "param_match": 0.0,
}

ZERO_RESULT = {**ZERO_METRICS, **ZERO_JUDGE}
