#!/usr/bin/env python3

from openai import OpenAI

# Test connection to your vLLM server
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8001/v1",
)

try:
    # List available models
    models = client.models.list()
    print("Available models:")
    for model in models.data:
        print(f"  - {model.id}")
    
    # Test a simple completion
    if models.data:
        model_id = models.data[0].id
        print(f"\nTesting model: {model_id}")
        
        response = client.completions.create(
            model=model_id,
            prompt="Hello",
            max_tokens=10
        )
        print("✓ Connection successful!")
        print(f"Response: {response.choices[0].text}")
    else:
        print("No models found!")
        
except Exception as e:
    print(f"Error connecting to vLLM server: {e}")
    print("Make sure your vLLM server is running at http://localhost:8001/v1")