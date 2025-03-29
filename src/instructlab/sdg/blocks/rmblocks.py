# Standard
from typing import Dict, List
import json
from urllib.parse import urljoin

# Third Party
from datasets import Dataset
import requests

# Local
from .block import Block
from ..logger_config import setup_logger
from ..registry import BlockRegistry

logger = setup_logger(__name__)

@BlockRegistry.register("PRMBlock")
class PRMBlock(Block):
    def __init__(
        self,
        block_name: str,
        host: str,
        port: int,
        model_name: str,
        prompt_col: str,
        response_col: str,
        output_col: str = "step_rewards",
        system_prompt: str = None,
        endpoint: str = "pooling",
        step_separator: str = "\n\n",
        step_fill_token: str = "<extra_0>",
    ) -> None:
        """Initialize the PRM (Preference/Reward Model) Block.
        
        Args:
            block_name: Name of the block
            host: Hostname of the PRM service (e.g., "0.0.0.0" or "localhost")
            port: Port number the service is running on
            model_name: Name of the PRM model to use
            prompt_col: Column name containing the prompt
            response_col: Column name containing the response
            output_col: Column name to store the reward scores (default: "step_rewards")
            system_prompt: Optional system prompt to use for scoring
            endpoint: API endpoint name (default: "pooling")
            step_separator: Separator between steps in the response (default: "\n\n")
            step_fill_token: Model specific fill token for steps in the response (default: "<extra_0>" used by Qwen2.5-Math-PRM)
        """
        super().__init__(block_name)
        # Construct base URL from host and port
        self.base_url = f"http://{host.strip('/')}:{port}/"
        self.endpoint = endpoint.strip('/')
        
        # Construct the full API URL using urljoin
        self.api_url = urljoin(self.base_url, self.endpoint)
        logger.info(f"Initialized PRMBlock with API URL: {self.api_url}")
        
        self.model_name = model_name
        self.prompt_col = prompt_col
        self.response_col = response_col
        self.output_col = output_col
        self.system_prompt = system_prompt
        self.step_separator = step_separator
        self.step_fill_token = step_fill_token

    def _post_request(self, messages: List[Dict]) -> requests.Response:
        """Make POST request to PRM API endpoint."""
        headers = {"User-Agent": "PRMBlock Client"}
        prompt = {
            "model": self.model_name,
            "messages": messages
        }
        response = requests.post(self.api_url, headers=headers, json=prompt)
        return response

    def _format_messages(self, sample: Dict) -> List[Dict]:
        """Format input sample into messages for the PRM API."""
        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
            
        messages.append({
            "role": "user",
            "content": sample[self.prompt_col]
        })
        messages.append({
            "role": "assistant",
            "content": self.step_fill_token.join(sample[self.response_col].split(self.step_separator)) + self.step_fill_token
        })
        return messages

    def _extract_rewards(self, response: requests.Response) -> List[float]:
        """Extract reward scores from API response."""
        try:
            response_data = response.json()
            rewards = [x[1] for x in response_data['data'][0]['data']]
            return rewards
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error extracting rewards from response: {e}")
            return []

    def _generate(self, sample: dict) -> dict:
        """Generate reward scores for the input samples.
        
        Args:
            sample: Input sample to score
            
        Returns:
            Dictionary with added reward scores column
        """
        messages = self._format_messages(sample)
        rm_response = self._post_request(messages)
            
        if rm_response.status_code != 200:
            logger.error(f"API request failed with status {rm_response.status_code}")
            rewards = [0.0] * len(messages)  # Default to 0 scores on failure
        else:
            rewards = self._extract_rewards(rm_response)
        
        sample[self.output_col] = rewards
        return sample

    def generate(self, samples: Dataset, batch_size: int = 4) -> Dataset:
        """Generate reward scores for the input samples."""
        return samples.map(self._generate, num_proc=batch_size)
