# SPDX-License-Identifier: Apache-2.0
"""DEPRECATED: LLM-based blocks for text generation and processing.

This module provides backwards-compatible blocks for interacting with language models.

DEPRECATED: The LLMBlock is deprecated and will be removed in a future version.
Use the new modular approach with PromptBuilderBlock, LLMChatBlock, and TextParserBlock instead.
"""

# Standard
from typing import Any, Dict, List, Optional
import tempfile
import warnings

# Third Party
from datasets import Dataset
import openai
import yaml

# Local
from ...logger_config import setup_logger
from ...registry import BlockRegistry
from ..block import Block
from ..llm.llm_chat_block import LLMChatBlock
from ..llm.prompt_builder_block import PromptBuilderBlock
from ..llm.text_parser_block import TextParserBlock

logger = setup_logger(__name__)


def server_supports_batched(client: Any, model_id: str) -> bool:
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
    setattr(client, "server_supports_batched", supported)
    logger.info(
        f"LLM server supports batched inputs: {getattr(client, 'server_supports_batched', False)}"
    )
    return supported


@BlockRegistry.register("LLMBlock")
class LLMBlock(Block):
    """DEPRECATED: Block for generating text using language models.

    This block maintains backwards compatibility with the old LLMBlock interface
    by internally using the new modular blocks: PromptBuilderBlock, LLMChatBlock, and TextParserBlock.

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

    def __init__(
        self,
        block_name: str,
        config_path: str,
        client: Any,
        output_cols: List[str],
        parser_kwargs: Dict[str, Any] = {},
        model_prompt: str = "{prompt}",
        model_id: Optional[str] = None,
        **batch_kwargs: Dict[str, Any],
    ) -> None:
        # Issue deprecation warning
        warnings.warn(
            "LLMBlock is deprecated and will be removed in a future version. "
            "Use the new modular approach with PromptBuilderBlock, LLMChatBlock, and TextParserBlock instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(block_name)

        # Store original parameters for compatibility
        self.config_path = config_path
        self.client = client
        self.output_cols = output_cols
        self.parser_kwargs = parser_kwargs or {}
        self.model_prompt = model_prompt
        self.batch_kwargs = batch_kwargs.get("batch_kwargs", {})

        # Load original config
        self.block_config = self._load_config(config_path)

        # Set model
        if model_id:
            self.model = model_id
        else:
            # get the default model id from client
            self.model = self.client.models.list().data[0].id

        # Create temporary config file for new prompt builder
        self._temp_prompt_config = self._create_prompt_config()

        # Initialize the three new blocks
        self._setup_internal_blocks()

    def _create_prompt_config(self) -> str:
        """Create a temporary YAML config file for the new PromptBuilderBlock format."""
        # Convert old config format to new message-based format
        messages = []

        # Add system message if present
        system_content = self.block_config.get("system")
        if system_content:
            messages.append({"role": "system", "content": system_content})

        # Create user message with the structured prompt
        user_content_parts = []
        for field in ["introduction", "principles", "examples", "generation"]:
            field_content = self.block_config.get(field)
            if field_content:
                if field == "introduction":
                    user_content_parts.append(field_content)
                elif field == "principles":
                    user_content_parts.append(f"Requirements:\n{field_content}")
                elif field == "examples":
                    user_content_parts.append(f"Examples:\n{field_content}")
                elif field == "generation":
                    user_content_parts.append(f"Task:\n{field_content}")

        if user_content_parts:
            messages.append(
                {"role": "user", "content": "\n\n".join(user_content_parts)}
            )

        # Write to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.safe_dump(messages, temp_file, default_flow_style=False)
        temp_file.flush()
        return temp_file.name

    def _setup_internal_blocks(self) -> None:
        """Initialize the three internal blocks."""
        # 1. PromptBuilderBlock
        self.prompt_builder = PromptBuilderBlock(
            block_name=f"{self.block_name}_prompt_builder",
            input_cols="sample_data",  # Will map all columns
            output_cols=["messages"],
            prompt_config_path=self._temp_prompt_config,
            format_as_messages=True,
            prompt_template_config=None,
            prompt_renderer=None,
        )

        # 2. LLMChatBlock
        # Convert client to LiteLLM format - support OpenAI and hosted_vllm
        if self.model.startswith("openai/") or self.model.startswith("hosted_vllm/"):
            model_name = self.model
        else:
            # Local/hosted model
            model_name = f"hosted_vllm/{self.model}"

        # Extract generation parameters from batch_kwargs and defaults
        defaults = {
            "temperature": 0,
            "max_tokens": 4096,
        }
        gen_params = {**defaults, **self.batch_kwargs}

        # Convert URL to string if needed and handle mock objects
        api_base = getattr(self.client, "base_url", None)
        if api_base is not None:
            api_base_str = str(api_base)
            # Skip mock objects
            api_base = (
                api_base_str if not api_base_str.startswith("<MagicMock") else None
            )

        # Handle api_key - convert to string or set to None for mocks
        api_key = getattr(self.client, "api_key", None)
        if api_key is not None:
            api_key_str = str(api_key)
            # Skip mock objects
            api_key = api_key_str if not api_key_str.startswith("<MagicMock") else None

        self.llm_chat = LLMChatBlock(
            block_name=f"{self.block_name}_llm_chat",
            input_cols=["messages"],
            output_cols=["raw_response"],
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            **gen_params,
        )

        # 3. TextParserBlock
        parser_config = {}

        # Handle parsing configuration
        parser_name = self.parser_kwargs.get("parser_name")
        if parser_name == "custom":
            parsing_pattern = self.parser_kwargs.get("parsing_pattern")
            cleanup_tags = self.parser_kwargs.get("parser_cleanup_tags")
            if parsing_pattern:
                parser_config["parsing_pattern"] = parsing_pattern
            if cleanup_tags:
                parser_config["parser_cleanup_tags"] = cleanup_tags
        else:
            # Use start/end tags from config
            start_tags = self.block_config.get("start_tags", [])
            end_tags = self.block_config.get("end_tags", [])
            if start_tags or end_tags:
                parser_config["start_tags"] = start_tags
                parser_config["end_tags"] = end_tags

        # Only create parser if we have parsing configuration
        if parser_config:
            self.text_parser: Optional[TextParserBlock] = TextParserBlock(
                block_name=f"{self.block_name}_text_parser",
                input_cols=["raw_response"],
                output_cols=self.output_cols,
                **parser_config,
            )
        else:
            self.text_parser = None

    def generate(self, samples: Dataset, **gen_kwargs: Dict[str, Any]) -> Dataset:
        """Generate the output from the block.

        This method maintains backwards compatibility by internally using the three new blocks.
        """
        logger.debug(
            "Generating outputs for {} samples using deprecated LLMBlock".format(
                len(samples)
            )
        )

        # Validate num_samples handling
        num_samples = self.block_config.get("num_samples")
        if (num_samples is not None) and ("num_samples" not in samples.column_names):
            samples = samples.add_column("num_samples", [num_samples] * len(samples))

        try:
            # Step 1: Format prompts using PromptBuilderBlock
            # Create a single sample with all column data for template rendering
            formatted_samples = []
            for sample in samples:
                # Create a sample_data column that contains the entire sample
                sample_with_data = {"sample_data": sample}
                formatted_samples.append(sample_with_data)

            prompt_dataset = Dataset.from_list(formatted_samples)
            prompt_result = self.prompt_builder.generate(prompt_dataset)

            # Step 2: Generate responses using LLMChatBlock
            chat_result = self.llm_chat.generate(prompt_result, **gen_kwargs)

            # Step 3: Handle n parameter before parsing
            num_parallel_samples = gen_kwargs.get("n", 1)

            if num_parallel_samples > 1:
                # When n > 1, we need to expand the list responses before parsing
                # TextParserBlock expects individual strings, not lists
                expanded_chat_data = []

                for sample in chat_result:
                    raw_responses = sample["raw_response"]
                    if isinstance(raw_responses, list):
                        # Create one row per response
                        for response in raw_responses:
                            expanded_sample = {**sample}
                            expanded_sample["raw_response"] = response
                            expanded_chat_data.append(expanded_sample)
                    else:
                        # Single response (fallback)
                        expanded_chat_data.append(sample)

                expanded_chat_result = Dataset.from_list(expanded_chat_data)

                # Step 4: Parse the expanded responses using TextParserBlock (if configured)
                if self.text_parser:
                    final_result = self.text_parser.generate(expanded_chat_result)
                else:
                    # If no parser, just rename the raw_response column to the first output column
                    if self.output_cols:
                        final_result = expanded_chat_result.rename_column(
                            "raw_response", self.output_cols[0]
                        )
                    else:
                        final_result = expanded_chat_result

                # Step 5: Merge with original samples (each original sample maps to n result samples)
                merged_data = []
                result_idx = 0

                for orig_sample in samples:
                    # Each original sample should have n corresponding results
                    for i in range(num_parallel_samples):
                        if result_idx < len(final_result):
                            result_sample = final_result[result_idx]
                            merged_sample = {**orig_sample}
                            for output_col in self.output_cols:
                                if output_col in result_sample:
                                    merged_sample[output_col] = result_sample[
                                        output_col
                                    ]
                                else:
                                    merged_sample[output_col] = ""
                            merged_data.append(merged_sample)
                            result_idx += 1
                        else:
                            # Missing result - create empty
                            merged_sample = {**orig_sample}
                            for output_col in self.output_cols:
                                merged_sample[output_col] = ""
                            merged_data.append(merged_sample)

                return Dataset.from_list(merged_data)

            else:
                # Step 4: Parse responses using TextParserBlock (if configured) - n=1 case
                if self.text_parser:
                    final_result = self.text_parser.generate(chat_result)
                else:
                    # If no parser, just rename the raw_response column to the first output column
                    if self.output_cols:
                        final_result = chat_result.rename_column(
                            "raw_response", self.output_cols[0]
                        )
                    else:
                        final_result = chat_result
                # Step 5: Merge with original samples for n=1 case
                merged_data = []
                for orig_sample, result_sample in zip(samples, final_result):
                    merged_sample = {**orig_sample}
                    for output_col in self.output_cols:
                        if output_col in result_sample:
                            response = result_sample[output_col]
                            # Handle case where response might still be a list with 1 item
                            if isinstance(response, list) and len(response) == 1:
                                merged_sample[output_col] = response[0]
                            elif isinstance(response, list):
                                # Multiple responses but n=1 - take first one
                                merged_sample[output_col] = (
                                    response[0] if response else ""
                                )
                            else:
                                merged_sample[output_col] = response
                        else:
                            merged_sample[output_col] = ""
                    merged_data.append(merged_sample)

                return Dataset.from_list(merged_data)

        except Exception as e:
            logger.error(f"Error in deprecated LLMBlock generation: {e}")
            # Fall back to empty dataset with proper structure
            empty_data = []
            for sample in samples:
                empty_sample = {**sample}
                for output_col in self.output_cols:
                    empty_sample[output_col] = ""
                empty_data.append(empty_sample)
            return Dataset.from_list(empty_data)

    def __del__(self):
        """Clean up temporary files."""
        try:
            # Standard
            import os

            if hasattr(self, "_temp_prompt_config"):
                os.unlink(self._temp_prompt_config)
        except Exception:
            pass
