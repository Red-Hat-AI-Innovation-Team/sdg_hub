# SPDX-License-Identifier: Apache-2.0

"""
LLM mocking setup for knowledge generation notebook integration testing.

This module provides deterministic LLM responses that follow the expected
flow execution pattern for knowledge generation workflows.
"""

# Mock LLM Setup - Deterministic responses for integration testing - MUST BE FIRST
from unittest.mock import MagicMock, patch


def create_knowledge_mock_response(call_count):
    """Generate deterministic responses based on call order."""

    # Summary responses (first 3 calls)
    if call_count <= 3:
        summaries = [
            "This is a comprehensive detailed summary covering key concepts and technical details.",
            "These are atomic facts: concept A, relationship B, implementation C.",
            "This extractive summary contains essential sentences from the source material.",
        ]
        return summaries[(call_count - 1) % 3]

    # Knowledge generation responses (4x calls per input after melt)
    elif call_count <= 11:  # 8 knowledge calls + 3 summary calls
        questions = [
            "What is the primary focus of this technology domain?",
            "How does this concept apply in practical scenarios?",
            "What are the key benefits of this approach?",
            "What considerations should be made when implementing this?",
            "What are the fundamental principles underlying this approach?",
            "How does this technology integrate with existing systems?",
            "What are the performance characteristics of this solution?",
            "What future developments can be expected in this area?",
        ]
        answers = [
            "The primary focus is on providing comprehensive technology solutions for enterprise needs.",
            "This concept applies through systematic implementation of best practices and proven methodologies.",
            "Key benefits include improved efficiency, reduced costs, and enhanced scalability.",
            "Important considerations include technical requirements, resource allocation, and timeline management.",
            "The fundamental principles involve systematic analysis, structured implementation, and continuous optimization.",
            "This technology integrates seamlessly through well-defined APIs and standard protocols.",
            "Performance characteristics include high throughput, low latency, and excellent scalability.",
            "Future developments will focus on enhanced automation, improved efficiency, and broader integration capabilities.",
        ]
        idx = (call_count - 4) % len(questions)
        return f"[QUESTION]\n{questions[idx]}\n[ANSWER]\n{answers[idx]}\n[END]"

    # Evaluation responses (remaining calls)
    else:
        if "faithfulness" in str(call_count) or call_count % 3 == 0:
            return "[Start of Explanation] The response is well-supported by the context. [End of Explanation] [Start of Answer] YES [End of Answer]"
        elif "relevancy" in str(call_count) or call_count % 3 == 1:
            return "[Start of Feedback] Subject Matter Relevance: 1, Query Focus Alignment: 1 [End of Feedback] [Start of Score] 2 [End of Score]"
        else:
            return "[Start of Explanation] The question is well-formulated and appropriate. [End of Explanation] [Start of Rating] 1.0 [End of Rating]"


# Create mock responses for knowledge generation (enough for the entire flow)
mock_responses = []
for i in range(100):  # Enough for multiple documents and flow steps
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = create_knowledge_mock_response(i + 1)
    mock_responses.append(mock_response)

# Create a cycling iterator for responses
response_iter = iter(mock_responses * 10)  # Repeat responses to ensure we don't run out


# Mock the litellm completion function used by SDG Hub
def mock_completion(*args, **kwargs):
    return next(response_iter)


# Mock the async completion function (following the working test pattern)
async def mock_async_completion(*_args, **_kwargs):
    return next(response_iter)


# COMPLETE LITELLM BYPASS STRATEGY
# Instead of letting LiteLLM route to different providers, we intercept at the entry point
# and immediately return our mock responses, bypassing ALL LiteLLM logic


# Patch LiteLLM's main entry points to completely bypass all routing and HTTP calls
def mock_litellm_completion(*args, **kwargs):
    """Completely bypass LiteLLM and return mock response immediately."""
    print(
        f"🔍 MOCK INTERCEPTED: litellm.completion called with model: {kwargs.get('model', 'unknown')}"
    )
    return next(response_iter)


async def mock_litellm_acompletion(*args, **kwargs):
    """Completely bypass LiteLLM async and return mock response immediately."""
    print(
        f"🔍 MOCK INTERCEPTED: litellm.acompletion called with model: {kwargs.get('model', 'unknown')}"
    )
    return next(response_iter)


# Patch at LiteLLM level - this catches ALL provider routing
litellm_completion_patcher = patch(
    "litellm.completion", side_effect=mock_litellm_completion
)
litellm_completion_patcher.start()

litellm_acompletion_patcher = patch(
    "litellm.acompletion", side_effect=mock_litellm_acompletion
)
litellm_acompletion_patcher.start()

# Also patch the client_manager imports (these import from litellm, so should be covered above)
# But we add these for extra safety in case there are direct imports
completion_patcher = patch(
    "sdg_hub.core.blocks.llm.client_manager.completion",
    side_effect=mock_litellm_completion,
)
completion_patcher.start()

acompletion_patcher = patch(
    "sdg_hub.core.blocks.llm.client_manager.acompletion",
    side_effect=mock_litellm_acompletion,
)
acompletion_patcher.start()

# CRITICAL: Also mock OpenAI client creation (like PR 269 does)
# LiteLLM creates OpenAI clients internally and calls them directly, bypassing our patches
mock_openai_client = MagicMock()
mock_openai_model = MagicMock()
mock_openai_model.id = "meta-llama/Llama-3.3-70B-Instruct"
mock_openai_client.models.list.return_value.data = [mock_openai_model]

# Mock both sync and async OpenAI completions
mock_openai_client.chat.completions.create.side_effect = mock_responses
mock_openai_client.chat.completions.with_raw_response.create.side_effect = (
    mock_responses
)

# Mock both OpenAI client classes that LiteLLM might use
openai_patcher = patch("openai.OpenAI", return_value=mock_openai_client)
openai_patcher.start()

openai_async_patcher = patch("openai.AsyncOpenAI", return_value=mock_openai_client)
openai_async_patcher.start()

print("✅ Mock LLM setup complete for knowledge generation")
print(f"   - Mocked {len(mock_responses)} responses with cycling iterator")
print("   - Patched: sdg_hub.core.blocks.llm.client_manager.completion")
print("   - Patched: sdg_hub.core.blocks.llm.client_manager.acompletion")
print("   - Patched: litellm.completion")
print("   - Patched: litellm.acompletion")
print("   - Patched: openai.OpenAI")
print("   - Patched: openai.AsyncOpenAI")
