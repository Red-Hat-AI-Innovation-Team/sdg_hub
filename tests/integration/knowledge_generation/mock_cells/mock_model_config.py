# SPDX-License-Identifier: Apache-2.0

"""
Mock model configuration setup for knowledge generation notebook integration testing.

This module configures the flow to use a mock model configuration that won't try
to connect to real API endpoints.
"""

# Override the model configuration to avoid real API calls
print("Setting up mock model configuration...")

# Instead of trying to connect to a real vLLM server, configure it to use a dummy/mock setup
flow.set_model_config(
    model="mock_model",  # Use a fake model name
    api_base="http://mock.localhost:8000/v1",  # Fake API base
    api_key="MOCK_KEY",
)

print('✅ Mock model configuration complete')