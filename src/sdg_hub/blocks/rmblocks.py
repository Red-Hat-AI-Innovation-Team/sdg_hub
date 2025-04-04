"""Module containing blocks for scoring responses using Reward Models."""

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
    """A block for scoring responses using a ProcessReward Model (PRM) via HTTP API.

    This block sends prompts and responses to a PRM endpoint and returns reward scores
    for each step in the response.
    """

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
        r"""Initialize the PRM (Process Reward Model) Block.

        Parameters
        ----------
        block_name : str
            Name of the block
        host : str
            Hostname of the PRM service (e.g., "0.0.0.0" or "localhost")
        port : int
            Port number the service is running on
        model_name : str
            Name of the PRM model to use
        prompt_col : str
            Column name containing the prompt
        response_col : str
            Column name containing the response
        output_col : str, optional
            Column name to store the reward scores, by default "step_rewards"
        system_prompt : str, optional
            Optional system prompt to use for scoring, by default None
        endpoint : str, optional
            API endpoint name, by default "pooling"
        step_separator : str, optional
            Separator between steps in the response, by default "\n\n"
        step_fill_token : str, optional
            Model specific fill token for steps in the response, by default "<extra_0>" used by Qwen2.5-Math-PRM
        """
        super().__init__(block_name)
        # Construct base URL from host and port
        self.base_url = f"http://{host.strip('/')}:{port}/"
        self.endpoint = endpoint.strip("/")

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
        """Make POST request to PRM API endpoint.

        Parameters
        ----------
        messages : List[Dict]
            List of message dictionaries to send to the API

        Returns
        -------
        requests.Response
            Response from the API
        """
        headers = {"User-Agent": "PRMBlock Client"}
        prompt = {"model": self.model_name, "messages": messages}
        response = requests.post(self.api_url, headers=headers, json=prompt)
        return response

    def _format_messages(self, question: str, response: str) -> List[Dict]:
        """Format input sample into messages for the PRM API.

        Parameters
        ----------
        question : str
            Question to score
        response : str
            Response to score

        Returns
        -------
        List[Dict]
            Formatted messages for the API
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": question})
        messages.append(
            {
                "role": "assistant",
                "content": self.step_fill_token.join(
                    response.split(self.step_separator)
                )
                + self.step_fill_token,
            }
        )
        return messages

    def _extract_rewards(self, response: requests.Response) -> List[float]:
        """Extract reward scores from API response.

        Parameters
        ----------
        response : requests.Response
            Response from the API

        Returns
        -------
        List[float]
            List of reward scores
        """
        try:
            response_data = response.json()
            rewards = [x[1] for x in response_data["data"][0]["data"]]
            return rewards
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error extracting rewards from response: {e}")
            return []

    def _generate(self, sample: dict) -> dict:
        """Generate reward scores for the input samples.

        Parameters
        ----------
        sample : dict
            Input sample to score

        Returns
        -------
        dict
            Dictionary with added reward scores column
        """
        if isinstance(sample[self.response_col], list):
            messages_list = [
                self._format_messages(sample[self.prompt_col], response)
                for response in sample[self.response_col]
            ]
        else:
            messages_list = [
                self._format_messages(
                    sample[self.prompt_col], sample[self.response_col]
                )
            ]
        rm_response_list = [self._post_request(messages) for messages in messages_list]

        rewards = []
        for i, rm_response in enumerate(rm_response_list):
            if isinstance(sample[self.response_col], list):
                response_steps = sample[self.response_col][i].split(self.step_separator)
            else:
                response_steps = sample[self.response_col].split(self.step_separator)

            if rm_response.status_code != 200:
                logger.error(
                    f"API request failed with status {rm_response.status_code} returning 0 rewards"
                )
                if isinstance(sample[self.response_col], list):
                    rewards.append([0.0] * len(response_steps))
                else:
                    rewards.append([0.0] * len(response_steps))
            else:
                step_rewards = self._extract_rewards(rm_response)
                if step_rewards:
                    rewards.append(step_rewards)
                else:
                    rewards.append([0.0] * len(response_steps))

        sample[self.output_col] = (
            rewards if isinstance(sample[self.response_col], list) else rewards[0]
        )
        return sample

    def generate(self, samples: Dataset, batch_size: int = 4) -> Dataset:
        """Generate reward scores for the input samples.

        Parameters
        ----------
        samples : Dataset
            Input dataset containing samples to score
        batch_size : int, optional
            Number of processes to use for parallel processing, by default 4

        Returns
        -------
        Dataset
            Dataset with added reward scores
        """
        return samples.map(self._generate, num_proc=batch_size)
