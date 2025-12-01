import requests
import os
import uuid

from .base import BaseAgentWrapper
from typing import Any

class LangflowAgentWrapper(BaseAgentWrapper):
    def __init__(self, agent_framework: str, agent_url: str, agent_api_key: str):
        super().__init__(agent_framework, agent_url, agent_api_key)

    def generate(self, messages: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:

        # Extract the user message content from the messages list
        # Langflow expects input_value to be a string, not a list of messages
        input_value = ""
        if isinstance(messages, list) and len(messages) > 0:
            # Get the last user message content
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    input_value = msg.get("content", "")
                    break
            # If no user message found, use the last message's content
            if not input_value and messages:
                input_value = messages[-1].get("content", "")
        elif isinstance(messages, str):
            input_value = messages

        payload = {
            "output_type": "chat",
            "input_type": "chat",
            "input_value": input_value
        }
        payload["session_id"] = session_id

        # Set headers with API key if provided
        headers = {}
        if self.agent_api_key:
            headers["x-api-key"] = self.agent_api_key

        try:
            response = requests.request("POST", self.agent_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making API request: {e}")
        except ValueError as e:
            raise Exception(f"Error parsing response: {e}")