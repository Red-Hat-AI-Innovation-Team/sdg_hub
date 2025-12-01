from abc import ABC, abstractmethod
from typing import Any

class BaseAgentWrapper(ABC):
    
    @abstractmethod
    def __init__(self, agent_framework: str, agent_url: str, agent_api_key: str):
        self.agent_framework = agent_framework
        self.agent_url = agent_url
        self.agent_api_key = agent_api_key

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
        """Generate responses from the agent.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            The messages to send to the agent.
        session_id : str
            The session ID to use for the agent.

        Returns
        -------
        list[dict[str, Any]]
            The responses from the agent.
        """
        pass