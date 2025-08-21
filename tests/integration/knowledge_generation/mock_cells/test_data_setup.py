# SPDX-License-Identifier: Apache-2.0

"""
Test data setup for knowledge generation notebook integration testing.

This module creates a small, deterministic dataset that replaces the large
seed data files used in production runs, enabling fast and predictable testing.
"""

# Test Data Setup - Replace large seed data with small test dataset
import os
from datasets import Dataset

# Create test seed data
test_data = [
    {
        'document': 'Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. It enables computers to learn and improve from experience without being explicitly programmed for every task.',
        'document_outline': '1. Definition of machine learning\n2. Relationship to AI\n3. Core concepts: algorithms and statistical models\n4. Learning from experience\n5. Automation benefits',
        'domain': 'technology',
        'seed_examples': 'Examples of ML applications include recommendation systems, image recognition, and natural language processing.',
        'icl_document': 'Artificial intelligence encompasses machine learning, deep learning, and other computational approaches to simulate human intelligence.',
        'icl_query_1': 'What is the relationship between AI and machine learning?',
        'icl_response_1': 'Machine learning is a subset of artificial intelligence, focusing specifically on algorithms that can learn from data.',
        'icl_query_2': 'How do machine learning algorithms work?',
        'icl_response_2': 'They analyze patterns in data to make predictions or decisions without explicit programming for each scenario.',
        'icl_query_3': 'What are common applications of machine learning?',
        'icl_response_3': 'Common applications include recommendation engines, fraud detection, image recognition, and autonomous vehicles.'
    },
    {
        'document': 'Cloud computing provides on-demand access to computing resources over the internet. It offers scalability, flexibility, and cost-effectiveness for businesses of all sizes by eliminating the need for physical infrastructure management.',
        'document_outline': '1. Cloud computing definition\n2. On-demand resource access\n3. Internet-based delivery\n4. Scalability benefits\n5. Cost advantages\n6. Infrastructure management',
        'domain': 'technology',
        'seed_examples': 'Cloud services include Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS).',
        'icl_document': 'Traditional computing required organizations to maintain physical servers and infrastructure on-premises.',
        'icl_query_1': 'What are the main benefits of cloud computing?',
        'icl_response_1': 'Key benefits include scalability, cost reduction, flexibility, and reduced infrastructure management overhead.',
        'icl_query_2': 'What are the different types of cloud services?',
        'icl_response_2': 'The main types are IaaS (infrastructure), PaaS (platform), and SaaS (software) as a service.',
        'icl_query_3': 'How does cloud computing differ from traditional computing?',
        'icl_response_3': 'Cloud computing provides remote access to resources over the internet, while traditional computing relies on local physical infrastructure.'
    }
]

# Create output directory
test_output_dir = 'test_sdg_demo_output'
os.makedirs(test_output_dir, exist_ok=True)

# Save test seed data
test_ds = Dataset.from_list(test_data)
test_ds.to_json(f'{test_output_dir}/seed_data.jsonl', orient='records', lines=True)

print(f'✅ Test data setup complete - {len(test_data)} samples saved to {test_output_dir}/seed_data.jsonl')