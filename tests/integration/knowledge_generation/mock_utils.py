# SPDX-License-Identifier: Apache-2.0

"""
Utilities for loading mock cells for knowledge generation notebook testing.
"""


def create_llm_mock_cell() -> dict:
    """
    Create LLM mock setup cell with comprehensive litellm mocking.
    
    Returns:
        Dictionary representing a Jupyter notebook cell
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["injected-mock"]},
        "outputs": [],
        "source": [
            "import sys\n",
            "from unittest.mock import MagicMock, patch\n",
            "\n",
            "def create_knowledge_mock_response(call_count):\n",
            "    if call_count <= 3:\n",
            "        summaries = [\n",
            "            'This is a comprehensive detailed summary covering key concepts and technical details.',\n",
            "            'These are atomic facts: concept A, relationship B, implementation C.',\n",
            "            'This extractive summary contains essential sentences from the source material.'\n",
            "        ]\n",
            "        return summaries[(call_count - 1) % 3]\n",
            "    elif call_count <= 11:\n",
            "        questions = [\n",
            "            'What is the primary focus of this technology domain?',\n",
            "            'How does this concept apply in practical scenarios?',\n",
            "            'What are the key benefits of this approach?',\n",
            "            'What considerations should be made when implementing this?'\n",
            "        ]\n",
            "        answers = [\n",
            "            'The primary focus is on providing comprehensive technology solutions for enterprise needs.',\n",
            "            'This concept applies through systematic implementation of best practices and proven methodologies.',\n",
            "            'Key benefits include improved efficiency, reduced costs, and enhanced scalability.',\n",
            "            'Important considerations include technical requirements, resource allocation, and timeline management.'\n",
            "        ]\n",
            "        idx = (call_count - 4) % len(questions)\n",
            "        return f'[QUESTION]\\\\n{questions[idx]}\\\\n[ANSWER]\\\\n{answers[idx]}\\\\n[END]'\n",
            "    else:\n",
            "        return ('[Start of Explanation]The response is well-supported by the context and highly relevant. '\n",
            "               'The question is well-formulated and appropriate.[End of Explanation]\\\\n'\n",
            "               '[Start of Answer]YES[End of Answer]\\\\n'\n",
            "               '[Start of Feedback]Subject Matter Relevance: 1, Query Focus Alignment: 1[End of Feedback]\\\\n'\n",
            "               '[Start of Score]2.0[End of Score]\\\\n'\n",
            "               '[Start of Rating]1.0[End of Rating]')\n",
            "\n",
            "mock_responses = []\n",
            "for i in range(100):\n",
            "    mock_response = MagicMock()\n",
            "    mock_response.choices = [MagicMock()]\n",
            "    mock_response.choices[0].message = MagicMock()\n",
            "    mock_response.choices[0].message.content = create_knowledge_mock_response(i + 1)\n",
            "    mock_responses.append(mock_response)\n",
            "\n",
            "response_iter = iter(mock_responses * 10)\n",
            "\n",
            "def mock_completion(*args, **kwargs):\n",
            "    return next(response_iter)\n",
            "\n",
            "async def mock_async_completion(*_args, **_kwargs):\n",
            "    return next(response_iter)\n",
            "\n",
            "def mock_litellm_completion(*args, **kwargs):\n",
            "    return next(response_iter)\n",
            "\n",
            "async def mock_litellm_acompletion(*args, **kwargs):\n",
            "    return next(response_iter)\n",
            "\n",
            "litellm_completion_patcher = patch('litellm.completion', side_effect=mock_litellm_completion)\n",
            "litellm_completion_patcher.start()\n",
            "\n",
            "litellm_acompletion_patcher = patch('litellm.acompletion', side_effect=mock_litellm_acompletion)\n",
            "litellm_acompletion_patcher.start()\n",
            "\n",
            "completion_patcher = patch('sdg_hub.core.blocks.llm.client_manager.completion', side_effect=mock_litellm_completion)\n",
            "completion_patcher.start()\n",
            "\n",
            "acompletion_patcher = patch('sdg_hub.core.blocks.llm.client_manager.acompletion', side_effect=mock_litellm_acompletion)\n",
            "acompletion_patcher.start()\n",
            "\n",
            "print('Mock LLM setup complete for knowledge generation')\n",
        ],
    }


def create_test_data_cell() -> dict:
    """
    Create test data setup cell.
    
    Returns:
        Dictionary representing a Jupyter notebook cell
    """
    return {
        "cell_type": "code", 
        "execution_count": None,
        "metadata": {"tags": ["test-data"]},
        "outputs": [],
        "source": [
            "import os\n",
            "from datasets import Dataset\n",
            "\n",
            "test_data = [\n",
            "    {\n",
            "        'document': 'Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. It enables computers to learn and improve from experience without being explicitly programmed for every task.',\n",
            "        'document_outline': '1. Definition of machine learning\\\\n2. Relationship to AI\\\\n3. Core concepts: algorithms and statistical models\\\\n4. Learning from experience\\\\n5. Automation benefits',\n",
            "        'domain': 'technology',\n",
            "        'seed_examples': 'Examples of ML applications include recommendation systems, image recognition, and natural language processing.',\n",
            "        'icl_document': 'Artificial intelligence encompasses machine learning, deep learning, and other computational approaches to simulate human intelligence.',\n",
            "        'icl_query_1': 'What is the relationship between AI and machine learning?',\n",
            "        'icl_response_1': 'Machine learning is a subset of artificial intelligence, focusing specifically on algorithms that can learn from data.',\n",
            "        'icl_query_2': 'How do machine learning algorithms work?',\n",
            "        'icl_response_2': 'They analyze patterns in data to make predictions or decisions without explicit programming for each scenario.',\n",
            "        'icl_query_3': 'What are common applications of machine learning?',\n",
            "        'icl_response_3': 'Common applications include recommendation engines, fraud detection, image recognition, and autonomous vehicles.'\n",
            "    },\n",
            "    {\n",
            "        'document': 'Cloud computing provides on-demand access to computing resources over the internet. It offers scalability, flexibility, and cost-effectiveness for businesses of all sizes by eliminating the need for physical infrastructure management.',\n",
            "        'document_outline': '1. Cloud computing definition\\\\n2. On-demand resource access\\\\n3. Internet-based delivery\\\\n4. Scalability benefits\\\\n5. Cost advantages\\\\n6. Infrastructure management',\n",
            "        'domain': 'technology',\n",
            "        'seed_examples': 'Cloud services include Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS).',\n",
            "        'icl_document': 'Traditional computing required organizations to maintain physical servers and infrastructure on-premises.',\n",
            "        'icl_query_1': 'What are the main benefits of cloud computing?',\n",
            "        'icl_response_1': 'Key benefits include scalability, cost reduction, flexibility, and reduced infrastructure management overhead.',\n",
            "        'icl_query_2': 'What are the different types of cloud services?',\n",
            "        'icl_response_2': 'The main types are IaaS (infrastructure), PaaS (platform), and SaaS (software) as a service.',\n",
            "        'icl_query_3': 'How does cloud computing differ from traditional computing?',\n",
            "        'icl_response_3': 'Cloud computing provides remote access to resources over the internet, while traditional computing relies on local physical infrastructure.'\n",
            "    }\n",
            "]\n",
            "\n",
            "test_output_dir = 'test_sdg_demo_output'\n",
            "os.makedirs(test_output_dir, exist_ok=True)\n",
            "\n",
            "test_ds = Dataset.from_list(test_data)\n",
            "test_ds.to_json(f'{test_output_dir}/seed_data.jsonl', orient='records', lines=True)\n",
            "\n",
            "print(f'Test data setup complete - {len(test_data)} samples saved to {test_output_dir}/seed_data.jsonl')\n",
        ],
    }


def get_knowledge_generation_mock_cells() -> list:
    """
    Get all mock cells needed for knowledge generation notebook testing.
    
    Returns:
        List of mock cell dictionaries
    """
    return [
        create_llm_mock_cell(),
        create_test_data_cell()
    ]