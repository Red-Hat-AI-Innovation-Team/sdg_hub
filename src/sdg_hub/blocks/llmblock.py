# SPDX-License-Identifier: Apache-2.0
"""LLM-based blocks for text generation and processing.

This module provides blocks for interacting with language models.
"""

# Standard
from typing import Any, Dict, List, Optional, Union
import json
import re

# Third Party
from datasets import Dataset
from jinja2 import Template
import openai

# Local
from .block import Block
from ..logger_config import setup_logger
from ..registry import BlockRegistry, PromptRegistry

logger = setup_logger(__name__)


def server_supports_batched(client: openai.OpenAI, model_id: str) -> bool:
    """Check if the server supports batched inputs.

    This function checks if the server supports batched inputs by making a test call to the server.

    Parameters
    ----------
    client : openai.OpenAI
        The client to use to make the test call.
    model_id : str
        The model ID to use for the test call.
    """
    supported = getattr(client, "server_supports_batched", None)
    if supported is not None:
        return supported
    try:
        # Make a test call to the server to determine whether it supports
        # multiple input prompts per request and also the n parameter
        response = client.completions.create(
            model=model_id, prompt=["test1", "test2"], max_tokens=1, n=3
        )
        # Number outputs should be 2 * 3 = 6
        supported = len(response.choices) == 6
    except openai.InternalServerError:
        supported = False
    client.server_supports_batched = supported
    logger.info(f"LLM server supports batched inputs: {client.server_supports_batched}")
    return supported


@BlockRegistry.register("LLMBlock")
class LLMBlock(Block):
    """Block for generating text using language models.

    This block handles text generation, prompt formatting, and output parsing
    for language model interactions.

    Parameters
    ----------
    block_name : str
        Name of the block.
    config_path : str
        Path to the configuration file.
    client : openai.OpenAI
        OpenAI client instance.
    output_cols : List[str]
        List of output column names.
    parser_kwargs : Dict[str, Any], optional
        Keyword arguments for the parser, by default {}.
    model_prompt : str, optional
        Template string for model prompt, by default "{prompt}".
    model_id : Optional[str], optional
        Model ID to use, by default None.
    **batch_kwargs : Dict[str, Any]
        Additional keyword arguments for batch processing.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        block_name: str,
        config_path: str,
        client: openai.OpenAI,
        output_cols: List[str],
        parser_kwargs: Dict[str, Any] = {},
        model_prompt: str = "{prompt}",
        model_id: Optional[str] = None,
        **batch_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(block_name)
        self.block_config = self._load_config(config_path)
        self.prompt_struct = (
            """{system}\n{introduction}\n{principles}\n{examples}\n{generation}"""
        )
        filtered_config = {
            k: (v if v is not None else "") for k, v in self.block_config.items()
        }
        self.prompt_template = Template(self.prompt_struct.format(**filtered_config))
        self.client = client
        if model_id:
            self.model = model_id
        else:
            # get the default model id from client
            self.model = self.client.models.list().data[0].id

        self.model_prompt = model_prompt
        self.output_cols = output_cols
        self.batch_params = batch_kwargs.get("batch_kwargs", {})
        self.parser_name = parser_kwargs.get("parser_name", None)
        self.parsing_pattern = parser_kwargs.get("parsing_pattern", None)
        self.parser_cleanup_tags = parser_kwargs.get("parser_cleanup_tags", None)
        self.defaults = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 4096,
        }

        # Whether the LLM server supports a list of input prompts
        # and supports the n parameter to generate n outputs per input
        self.server_supports_batched = server_supports_batched(client, self.model)

    def _extract_matches(
        self, text: str, start_tag: Optional[str], end_tag: Optional[str]
    ) -> List[str]:
        if not text:
            return []
        if not start_tag and not end_tag:
            return [text.strip()]

        pattern = ""
        if start_tag:
            pattern += re.escape(start_tag)
        pattern += r"(.*?)"
        if end_tag:
            pattern += re.escape(end_tag)
        elif start_tag:
            # Enforce matching till end of string when only start_tag is provided.
            pattern += "$"

        return [match.strip() for match in re.findall(pattern, text, re.DOTALL)]

    def _parse(self, generated_string: str) -> dict:
        matches = {}

        if self.parser_name is not None and self.parser_name == "custom":
            pattern = re.compile(self.parsing_pattern, re.DOTALL)
            all_matches = pattern.findall(generated_string)
            matches = {column_name: [] for column_name in self.output_cols}
            if all_matches and isinstance(all_matches[0], tuple):
                for match in all_matches:
                    for column_name, value in zip(self.output_cols, match):
                        value = value.strip()
                        for clean_tag in self.parser_cleanup_tags:
                            value = value.replace(clean_tag, "")
                        matches[column_name].append(value)
            else:
                matches[self.output_cols[0]] = (
                    [match.strip() for match in all_matches] if all_matches else []
                )
        else:
            for start_tag, end_tag, output_col in zip(
                self.block_config.get("start_tags", []),
                self.block_config.get("end_tags", []),
                self.output_cols,
            ):
                matches[output_col] = self._extract_matches(
                    generated_string, start_tag, end_tag
                )

        return matches

    def _format_prompt(self, sample: Dict) -> str:
        prompt_templated_str = self.prompt_template.render(sample).strip()
        return PromptRegistry.render_template(
            self.model_prompt, prompt_templated_str, add_generation_prompt=True
        ).strip()

    def _generate(self, samples: Dataset, **gen_kwargs: Dict[str, Any]) -> list:
        prompts = [self._format_prompt(sample) for sample in samples]
        logger.debug("Prompt: %s", prompts[0])
        generate_args = {**self.defaults, **gen_kwargs}

        if self.server_supports_batched:
            response = self.client.completions.create(prompt=prompts, **generate_args)
            # if stop is provided, then we need to add the stop token to the generated text,
            # this is because the stop token is not included in the generated text - this is a limitation of the openai api
            # we need to add the stop token to the generated text to make it consistent for the parser
            if "stop" in generate_args:
                return [
                    choice.text.strip() + "".join(generate_args["stop"])
                    for choice in response.choices
                ]
            return [choice.text.strip() for choice in response.choices]

        n = gen_kwargs.get("n", 1)
        results = []
        for prompt in prompts:
            for _ in range(n):
                response = self.client.completions.create(
                    prompt=prompt, **generate_args
                )
                if "stop" in generate_args:
                    results.append(
                        response.choices[0].text.strip()
                        + "".join(generate_args["stop"])
                    )
                results.append(response.choices[0].text.strip())
        return results

    def generate(self, samples: Dataset, **gen_kwargs: Dict[str, Any]) -> Dataset:
        """Generate the output from the block.

        This method should first validate the input data,
        then generate the output, and finally parse the generated output before returning it.

        Returns
        -------
        Dataset
            The parsed output after generation.
        """
        num_samples = self.block_config.get("num_samples", None)
        logger.debug("Generating outputs for {} samples".format(len(samples)))

        if (num_samples is not None) and ("num_samples" not in samples.column_names):
            samples = samples.add_column("num_samples", [num_samples] * len(samples))

        # validate each sample
        # Log errors and remove invalid samples
        valid_samples = []

        for sample in samples:
            if self._validate(self.prompt_template, sample):
                valid_samples.append(sample)
            else:
                logger.warning(
                    f"Sample failed validation: {sample}"
                )  # Log details of the failed sample

        samples = valid_samples

        if len(samples) == 0:
            logger.warning(
                "No valid samples to generate outputs for, returning empty dataset"
            )
            return Dataset.from_list([])

        # generate the output

        outputs = self._generate(samples, **gen_kwargs)

        logger.debug("Generated outputs: %s", outputs)

        num_parallel_samples = gen_kwargs.get("n", 1)
        extended_samples = []

        # Duplicate each input sample n times, where n is the number
        # of output sequences generated per input, so that we can
        # pair up the inputs and outputs.
        for item in samples:
            extended_samples.extend([item] * num_parallel_samples)

        new_data = []
        for sample, output in zip(extended_samples, outputs):
            parsed_outputs = self._parse(output)
            max_length = max(len(value) for value in parsed_outputs.values())
            for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
                new_data.append({**sample, **dict(zip(parsed_outputs.keys(), values))})

        return Dataset.from_list(new_data)


@BlockRegistry.register("ConditionalLLMBlock")
class ConditionalLLMBlock(LLMBlock):
    """Block for conditional text generation using language models.

    This block selects different prompt templates based on a selector column value.

    Parameters
    ----------
    block_name : str
        Name of the block.
    config_paths : Dict[str, str]
        Dictionary mapping selector values to their config file paths.
    client : openai.OpenAI
        OpenAI client instance.
    model_id : str
        Model ID to use.
    output_cols : List[str]
        List of output column names.
    selector_column_name : str
        Name of the column used to select the prompt template.
    model_prompt : str, optional
        Template string for model prompt, by default "{prompt}".
    **batch_kwargs : Dict[str, Any]
        Additional keyword arguments for batch processing.
    """

    def __init__(
        self,
        block_name: str,
        config_paths: Dict[str, str],
        client: openai.OpenAI,
        model_id: str,
        output_cols: List[str],
        selector_column_name: str,
        model_prompt: str = "{prompt}",
        **batch_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            block_name=block_name,
            config_path=list(config_paths.values())[0],
            client=client,
            model_id=model_id,
            output_cols=output_cols,
            model_prompt=model_prompt,
            **batch_kwargs,
        )
        self.selector_column_name = selector_column_name
        self.prompt_template = {}
        if "All" in config_paths:
            self.prompt_template = self.prompt_struct.format(**self.block_config)
        else:
            for config_key, config in config_paths.items():
                filtered_config = {
                    k: (v if v is not None else "")
                    for k, v in self.block_config.items()
                }
                self.prompt_template[config_key] = Template(
                    self.prompt_struct.format(**self._load_config(config))
                )

    def _format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format the prompt based on the selector column value.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample containing the selector column.

        Returns
        -------
        str
            Formatted prompt string.
        """
        if isinstance(self.prompt_template, dict):
            return (
                self.prompt_template[sample[self.selector_column_name]]
                .render(**sample)
                .strip()
            )

        return self.prompt_template.render(**sample).strip()

    def _validate(self, prompt_template: Union[str, Template], input_dict: Dict[str, Any]) -> bool:
        """Validate the input data for this block.

        Parameters
        ----------
        prompt_template : Union[str, Template]
            The template to validate against.
        input_dict : Dict[str, Any]
            Input data to validate.

        Returns
        -------
        bool
            True if the input data is valid, False otherwise.
        """
        if isinstance(prompt_template, dict):
            prompt_template = prompt_template[input_dict[self.selector_column_name]]
        return super()._validate(prompt_template, input_dict)
