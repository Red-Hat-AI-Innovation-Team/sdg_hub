# Unified Agent Interface: Standardized Input and Output for Agent Frameworks

**Date:** 2026-04-25
**Status:** Exploration / Approved for Phase 1
**Related:** [Issue #734](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/issues/734) (Phase 1 output standardization)

## Vision

A single, framework-agnostic interface for interacting with any agent — standardized types for what goes in, standardized types for what comes out, with a path toward a standard wire protocol as the ecosystem matures.

**Analogy:** LiteLLM gave us "one `completion()` call for any LLM." We want "one `send()` call for any agent" — same types in, same types out, regardless of whether the agent is Langflow, LangGraph, CrewAI, or something that doesn't exist yet.

## Problem

Agent frameworks have completely different interfaces at every layer:

| Layer | Langflow | LangGraph |
|-------|----------|-----------|
| **HTTP endpoint** | `POST /api/v1/run/{flow-id}` | `POST /threads` + `POST /threads/{id}/runs/wait` |
| **Request format** | `{"input_value": "text", "session_id": "..."}` | `{"assistant_id": "...", "input": {"messages": [...]}}` |
| **Auth header** | `x-api-key` | `x-api-key` |
| **Response structure** | Deeply nested `outputs[0].outputs[0].results.message.text` | Flat `{"messages": [...]}` with typed message objects |
| **Tool trace location** | `content_blocks[].contents[]` with `tool_use` entries | Messages with `tool_calls` field or `type: "tool"` |
| **Session tracking** | `session_id` in request/response | Thread-based, no session_id in response |

This forces per-framework translation at both the request and response level, and requires every downstream consumer to know which framework produced the response.

## Current State

sdg_hub has a connector system (`BaseAgentConnector`) with per-framework subclasses:

```
Caller → build_request(messages, session_id) → framework HTTP API → parse_response(raw_dict) → raw dict
```

- **Input**: Standard Python interface (`messages: list[dict]`, `session_id: str`), but per-framework wire translation in `build_request()`
- **Output**: Raw framework-specific dicts. `AgentResponseExtractorBlock` requires `agent_framework` field to dispatch to per-connector `extract_text()` / `extract_tool_trace()` class methods
- Adding a new framework means implementing both the connector AND the extraction logic

## Phased Approach

### Phase 1: Standardize the Types (MLFlow — Near-Term)

Use `mlflow-tracing` types as the standard interface on both sides:

```
ChatAgentRequest → [per-framework connector] → ChatAgentResponse
(standard in)       (wire translation)          (standard out)
```

**Dependency:** `mlflow-tracing` (1.5 MB, lightweight — only types + OTel, no numpy/pandas/sklearn)

**Input type — `ChatAgentRequest`** (`mlflow.types.agent`):

```python
ChatAgentRequest(
    messages=[ChatAgentMessage(role="user", content="How old is the moon?")],
    context=ChatContext(conversation_id="session-123", user_id="user-456"),
    custom_inputs={"temperature": 0.7},  # framework-specific extras
)
```

- `messages`: replaces `list[dict]` with typed `ChatAgentMessage` objects
- `context.conversation_id`: replaces `session_id` string
- `custom_inputs`: replaces ad-hoc kwargs for framework-specific settings

**Output type — `ChatAgentResponse`** (`mlflow.types.agent`):

```python
ChatAgentResponse(
    messages=[
        ChatAgentMessage(role="assistant", content="", tool_calls=[
            ToolCall(name="search", arguments={"q": "moon age"})
        ]),
        ChatAgentMessage(role="tool", name="search", content="4.5B years"),
        ChatAgentMessage(role="assistant", content="The moon is about 4.5 billion years old."),
    ],
    custom_outputs={"session_id": "abc123"},
    usage=ChatUsage(input_tokens=150, output_tokens=25, total_tokens=175),
)
```

**What changes:**

| Component | Before | After |
|-----------|--------|-------|
| `BaseAgentConnector.send()` | `send(messages: list[dict], session_id: str) → dict` | `send(request: ChatAgentRequest) → ChatAgentResponse` |
| `build_request()` | `(messages, session_id) → framework dict` | `(request: ChatAgentRequest) → framework dict` |
| `parse_response()` | `raw dict → raw dict` | `raw dict → ChatAgentResponse` |
| `extract_text/tool_trace/session_id` | Per-connector class methods | Removed — response is already structured |
| `AgentResponseExtractorBlock` | Requires `agent_framework` | Framework-agnostic, reads standardized fields |
| `AgentBlock` | Builds `messages` list, generates `session_id` | Builds `ChatAgentRequest` |

**Per-framework connectors still needed** because the wire protocols are fundamentally different. The connectors become thinner — pure wire translators — but they handle:
- Langflow: flatten `ChatAgentRequest` into `input_value` string, rebuild `ChatAgentResponse` from nested output
- LangGraph: translate `ChatAgentRequest` into thread + run workflow, rebuild `ChatAgentResponse` from state messages

**Result:**

```
Today:     N frameworks × (custom request + custom response) = 2N translation surfaces
Phase 1:   N frameworks × (wire translation only) = N translations, uniform Python types
```

**Implementation:** See [Issue #734](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/issues/734) for the output side. Input side (`ChatAgentRequest`) extends that work.

---

### Phase 2: Standardize the Wire Protocol (A2A — Future Exploration)

**Goal:** Eliminate per-framework connectors entirely. When agents expose A2A endpoints, a single `A2AConnector` replaces all framework-specific connectors.

```
ChatAgentRequest → [A2AConnector] → A2A protocol → any agent → A2A response → ChatAgentResponse
(standard in)      (one connector)   (standard wire)                            (standard out)
```

**What is A2A:** Google's Agent-to-Agent protocol (now Linux Foundation, Apache 2.0). JSON-RPC 2.0 over HTTP(S). 23k+ GitHub stars, 200+ partner organizations (including Red Hat, LangChain, Microsoft, AWS). Python SDK: `a2a-sdk`.

**What A2A gives us:**
- One wire protocol for any agent — `a2a.sendMessage` with `Message` + `Parts`
- Agent discovery via `AgentCard` (published at `/.well-known/agent-card.json`)
- Built-in multi-turn via `contextId` / `taskId`
- Streaming via SSE
- Framework samples exist for LangGraph, CrewAI, LlamaIndex, Semantic Kernel

**The open question — tool trace visibility:**

A2A treats agents as opaque by design. The protocol captures final outputs (Artifacts) but intentionally hides internal execution (tool calls, reasoning steps, intermediate LLM calls). This is the critical gap for sdg_hub, where MCP distillation flows need the full tool-use trace to generate training data.

**This is the primary thing to explore in Phase 2.**

#### Research Questions for Phase 2 Exploration

An agent should investigate the following questions to determine whether A2A can serve as sdg_hub's standard wire protocol:

**1. Can A2A carry tool traces?**

The A2A `Part` type supports `data` (arbitrary JSON) and `metadata` (key-value pairs). Artifacts also have `metadata`. Can these be used to carry structured tool traces?

- Build a sample LangGraph agent wrapped in an A2A server using `a2a-sdk`
- Have the agent make multiple tool calls during execution
- Try returning tool call details via: (a) `Part.data` with a structured schema, (b) `Artifact.metadata`, (c) multiple Artifacts (one per tool call + one for final output)
- Evaluate: does the A2A response preserve enough information to reconstruct a `ChatAgentResponse` with full `tool_calls` and `role="tool"` messages?

**2. Can the A2A wrapper access the agent's internal execution trace?**

When wrapping a LangGraph agent in an A2A `AgentExecutor`:

- Does the wrapper have access to the LangGraph state (which includes all messages with tool calls)?
- Can the wrapper forward the full message history (including tool calls) through the A2A response?
- How does this work for CrewAI, LlamaIndex, or other frameworks?
- Is there a common pattern for exposing internal execution details through A2A?

**3. Can OTel traces supplement A2A responses?**

MLFlow can ingest OpenTelemetry traces. Some agent frameworks can emit OTel traces.

- If the A2A-wrapped agent also emits OTel traces, can sdg_hub correlate the A2A response with the OTel trace?
- Could the A2A response include an `otel_trace_id` in metadata, allowing sdg_hub to fetch the full trace from an MLFlow tracking server?
- Evaluate: dual-channel approach (A2A for request/response, OTel for observability/trace detail)

**4. What is the A2A adoption timeline for target frameworks?**

- Does Langflow have A2A support or plans for it?
- Does LangGraph Cloud / LangGraph Platform expose A2A endpoints?
- Are there any frameworks where A2A is first-class (not just a sample/wrapper)?
- What is the cost of wrapping an existing agent in A2A vs. using the framework's native API?

**5. Can we define an sdg_hub trace convention for A2A?**

If tool traces can be carried in `Part.data`:

- Define a schema for tool trace data in A2A parts (similar to `ChatAgentMessage` with `tool_calls`)
- Propose this as a convention or extension to the A2A spec
- Evaluate whether the A2A community would accept this (given the intentional opacity design)

#### Success Criteria for Phase 2

Phase 2 is viable if ALL of the following are true:

1. An A2A-wrapped agent can return tool call details (names, arguments, results) through the A2A response — even if via `Part.data` convention
2. The `A2AConnector.parse_response()` can reconstruct a `ChatAgentResponse` with full tool trace messages (not just final text)
3. At least one target framework (Langflow or LangGraph) has native or near-native A2A support
4. The A2A wrapping cost is justified — simpler than maintaining per-framework connectors

If any of these fail, Phase 2 becomes: "add A2A as an optional connector for text-only use cases, keep framework-native connectors for tool trace scenarios."

#### Prototype Plan for Phase 2

1. **Build a proof-of-concept**: LangGraph agent → A2A wrapper → `A2AConnector` → `ChatAgentResponse`
2. **Test tool trace round-trip**: Agent makes 3+ tool calls → A2A response carries tool details → `ChatAgentResponse` has full `tool_calls` messages
3. **Compare output quality**: Same agent called via native LangGraph connector vs A2A connector — are the `ChatAgentResponse` objects equivalent?
4. **Evaluate DX**: Is the A2A wrapping + AgentCard setup simpler or more complex than writing a native connector?

---

## End State

```
Phase 1 (near-term):
    ChatAgentRequest → [LangflowConnector] → ChatAgentResponse
    ChatAgentRequest → [LangGraphConnector] → ChatAgentResponse
    ChatAgentRequest → [FutureConnector]    → ChatAgentResponse
    (N connectors, uniform Python types, per-framework wire translation)

Phase 2 (future, if viable):
    ChatAgentRequest → [A2AConnector] → ChatAgentResponse     ← any A2A agent
    ChatAgentRequest → [NativeConnector] → ChatAgentResponse   ← fallback for non-A2A / deep trace needs
    (1 universal + N fallback connectors)
```

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-20 | Use `mlflow-tracing` for output standardization | Lightweight (1.5MB), includes `ChatAgentResponse` types, no heavy deps |
| 2026-04-20 | Clean break on `AgentResponseExtractorBlock` | Agent connector system is new, breaking changes acceptable |
| 2026-04-21 | A2A cannot replace connectors today | No client-side routing, no Langflow/LangGraph native support, loses tool traces |
| 2026-04-21 | MLFlow has no client-side agent abstraction | Autolog is in-process only, AI Gateway speaks LLM not agent protocols |
| 2026-04-25 | Phase 1: MLFlow types for both input and output | Extends issue #734 to include `ChatAgentRequest` on input side |
| 2026-04-25 | Phase 2: A2A exploration for wire protocol | Needs investigation on tool trace viability before committing |
