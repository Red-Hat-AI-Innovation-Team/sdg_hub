"""Evaluation utilities for MCP benchmark.

Trace extraction, formatting, and programmatic metrics used by the
evaluation notebook and any standalone evaluation scripts.
"""

import json

# ── Trace extraction (from MCPAgentBlock output) ─────────────────────


def extract_model_tools(trace: dict) -> list[str]:
    """Extract ordered tool names from MCPAgentBlock trace."""
    tools = []
    for msg in trace.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", tc)
                if fn.get("name"):
                    tools.append(fn["name"])
    return tools


def extract_model_tool_trace(trace: dict) -> list[dict]:
    """Extract full tool trace from MCPAgentBlock output.

    Returns list of {"name": ..., "input": ..., "output": ...} dicts,
    matching the canonical format of expert_tool_trace in benchmark_tasks.jsonl.
    """
    calls, pending = [], {}
    for msg in trace.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", tc)
                call = {"name": fn.get("name", ""), "input": fn.get("arguments", {})}
                if isinstance(call["input"], str):
                    try:
                        call["input"] = json.loads(call["input"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                tc_id = tc.get("id")
                if tc_id:
                    pending[tc_id] = call
                calls.append(call)
        elif msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id and tc_id in pending:
                pending[tc_id]["output"] = msg.get("content", "")
    return calls


def extract_model_answer(trace: dict) -> str:
    """Extract final text answer from MCPAgentBlock trace."""
    for msg in reversed(trace.get("messages", [])):
        if msg.get("role") == "assistant" and not msg.get("tool_calls"):
            return msg.get("content", "") or ""
    return ""


# ── Trace formatting ─────────────────────────────────────────────────


def format_trace_for_judge(
    tool_trace: list[dict],
    max_args_len: int = 300,
    max_output_len: int = 200,
) -> str:
    """Format a tool trace into a readable string for the judge prompt.

    Each tool call is formatted as:
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

    # Parameter match: compare arguments for matching tool calls
    param_match = 0.0
    if model_trace and expert_trace:
        matched, total = 0, 0
        for et in expert_trace:
            for mt in model_trace:
                if mt.get("name") == et.get("name"):
                    total += 1
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
