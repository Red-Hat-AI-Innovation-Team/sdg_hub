# SPDX-License-Identifier: Apache-2.0

"""
LLM mocking setup for knowledge generation notebook integration testing.

This module provides deterministic LLM responses that follow the expected
flow execution pattern for knowledge generation workflows.
"""

# Mock LLM Setup - Deterministic responses for integration testing
from unittest.mock import patch, MagicMock
import asyncio

def create_knowledge_mock_response(call_count):
    """Generate deterministic responses based on call order."""
    
    # Summary responses (first 3 calls)
    if call_count <= 3:
        summaries = [
            'This is a comprehensive detailed summary covering key concepts and technical details.',
            'These are atomic facts: concept A, relationship B, implementation C.',
            'This extractive summary contains essential sentences from the source material.'
        ]
        return summaries[(call_count - 1) % 3]
    
    # Knowledge generation responses (4x calls per input after melt)
    elif call_count <= 11:  # 8 knowledge calls + 3 summary calls
        questions = [
            'What is the primary focus of this technology domain?',
            'How does this concept apply in practical scenarios?',
            'What are the key benefits of this approach?',
            'What considerations should be made when implementing this?',
            'What are the fundamental principles underlying this approach?',
            'How does this technology integrate with existing systems?',
            'What are the performance characteristics of this solution?',
            'What future developments can be expected in this area?'
        ]
        answers = [
            'The primary focus is on providing comprehensive technology solutions for enterprise needs.',
            'This concept applies through systematic implementation of best practices and proven methodologies.',
            'Key benefits include improved efficiency, reduced costs, and enhanced scalability.',
            'Important considerations include technical requirements, resource allocation, and timeline management.',
            'The fundamental principles involve systematic analysis, structured implementation, and continuous optimization.',
            'This technology integrates seamlessly through well-defined APIs and standard protocols.',
            'Performance characteristics include high throughput, low latency, and excellent scalability.',
            'Future developments will focus on enhanced automation, improved efficiency, and broader integration capabilities.'
        ]
        idx = (call_count - 4) % len(questions)
        return f'[QUESTION]\n{questions[idx]}\n[ANSWER]\n{answers[idx]}\n[END]'
    
    # Evaluation responses (remaining calls)
    else:
        if 'faithfulness' in str(call_count) or call_count % 3 == 0:
            return '[Start of Explanation] The response is well-supported by the context. [End of Explanation] [Start of Answer] YES [End of Answer]'
        elif 'relevancy' in str(call_count) or call_count % 3 == 1: 
            return '[Start of Feedback] Subject Matter Relevance: 1, Query Focus Alignment: 1 [End of Feedback] [Start of Score] 2 [End of Score]'
        else:
            return '[Start of Explanation] The question is well-formulated and appropriate. [End of Explanation] [Start of Rating] 1.0 [End of Rating]'

# Global call counter for deterministic responses
global_call_count = 0

async def mock_completion(*args, **kwargs):
    """Mock completion function with deterministic responses."""
    global global_call_count
    global_call_count += 1
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = create_knowledge_mock_response(global_call_count)
    
    return mock_response

# Apply the comprehensive mocking
completion_patcher = patch('sdg_hub.core.blocks.llm.client_manager.completion', side_effect=mock_completion)
completion_patcher.start()

print('✅ Mock LLM setup complete - all API calls will be intercepted')