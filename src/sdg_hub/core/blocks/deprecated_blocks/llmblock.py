# SPDX-License-Identifier: Apache-2.0
"""DEPRECATED: LLM-based blocks for text generation and processing.

This module provides backwards-compatible blocks for interacting with language models.

DEPRECATED: The LLMBlock is deprecated and will be removed in a future version.
Use the new modular approach with PromptBuilderBlock, LLMChatBlock, and TextParserBlock instead.
"""

# Standard
from typing import Any, Optional
import os
import tempfile
import warnings

# Third Party
from datasets import Dataset
from jinja2 import Environment, meta
import openai
import yaml

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..llm.llm_chat_block import LLMChatBlock
from ..llm.prompt_builder_block import PromptBuilderBlock
from ..llm.text_parser_block import TextParserBlock
from ..registry import BlockRegistry

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
    client.server_supports_batched = supported
    logger.info(
        f"LLM server supports batched inputs: {getattr(client, 'server_supports_batched', False)}"
    )
    return supported


@BlockRegistry.register(
    block_name="LLMBlock",
    category="deprecated",
    description="DEPRECATED: Use the new modular approach with PromptBuilderBlock, LLMChatBlock, and TextParserBlock instead",
)
class LLMBlock(BaseBlock):
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
        output_cols: list[str],
        parser_kwargs: dict[str, Any] = None,
        model_prompt: str = "{prompt}",
        model_id: Optional[str] = None,
        **batch_kwargs: dict[str, Any],
    ) -> None:
        # Issue deprecation warning
        if parser_kwargs is None:
            parser_kwargs = {}
        warnings.warn(
            "LLMBlock is deprecated and will be removed in a future version. "
            "Use the new modular approach with PromptBuilderBlock, LLMChatBlock, and TextParserBlock instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Load config and extract input columns before calling super().__init__()
        block_config = self._load_config_static(config_path)
        input_cols = self._extract_template_variables_static(block_config)

        super().__init__(
            block_name=block_name, input_cols=input_cols, output_cols=output_cols
        )

        # Now we can set instance attributes
        self.config_path = config_path
        self.block_config = block_config

        # Store original parameters for compatibility
        self.client = client
        self.parser_kwargs = parser_kwargs or {}
        self.model_prompt = model_prompt
        self.batch_kwargs = batch_kwargs.get("batch_kwargs", {})

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

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        return self._load_config_static(config_path)

    @staticmethod
    def _load_config_static(config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file (static version)."""
        try:
            with open(config_path, encoding="utf-8") as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    def _extract_template_variables(self) -> list[str]:
        """Extract Jinja2 template variables from all config fields."""
        return self._extract_template_variables_static(self.block_config)

    @staticmethod
    def _extract_template_variables_static(block_config: dict[str, Any]) -> list[str]:
        """Extract Jinja2 template variables from all config fields (static version)."""
        variables: set[str] = set()
        env = Environment()

        # Extract variables from all string fields in config
        for field in ["system", "introduction", "principles", "examples", "generation"]:
            field_content = block_config.get(field, "")
            if isinstance(field_content, str) and field_content.strip():
                try:
                    ast = env.parse(field_content)
                    field_vars = meta.find_undeclared_variables(ast)
                    variables.update(field_vars)
                except Exception as e:
                    logger.debug(
                        f"Could not parse template variables from {field}: {e}"
                    )

        return list(variables)

    def _create_prompt_config(self) -> str:
        """Create a temporary YAML config file for the new PromptBuilderBlock format."""
        # Convert old config format to new message-based format
        messages = []

        # Create user message with the structured prompt (matching old format)
        # Build prompt using the original structure: {system}\n{introduction}\n{principles}\n{examples}\n{generation}
        prompt_parts = []
        for field in ["system", "introduction", "principles", "examples", "generation"]:
            field_content = self.block_config.get(field, "")
            prompt_parts.append(field_content)

        # Join with single newlines to match original prompt_struct
        user_content = "\n".join(prompt_parts)

        if user_content.strip():
            messages.append({"role": "user", "content": user_content})

        # Write to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml.j2", delete=False
        )
        yaml.safe_dump(messages, temp_file, default_flow_style=False)
        temp_file.flush()
        return temp_file.name

    def _setup_internal_blocks(self) -> None:
        """Initialize the three internal blocks."""
        # 1. PromptBuilderBlock
        self.prompt_builder = PromptBuilderBlock(
            block_name=f"{self.block_name}_prompt_builder",
            input_cols=self.input_cols,  # Pass through original input columns for template access
            output_cols=["messages"],
            prompt_config_path=self._temp_prompt_config,
            format_as_messages=True,
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

    def generate(self, samples: Dataset, **gen_kwargs: dict[str, Any]) -> Dataset:
        """Generate the output from the block.

        This method maintains backwards compatibility by internally using the three new blocks.
        """
        logger.debug(
            f"Generating outputs for {len(samples)} samples using deprecated LLMBlock"
        )

        # Validate num_samples handling
        num_samples = self.block_config.get("num_samples")
        if (num_samples is not None) and ("num_samples" not in samples.column_names):
            samples = samples.add_column("num_samples", [num_samples] * len(samples))

        try:
            # Step 1: Format prompts using PromptBuilderBlock
            # Pass the original dataset directly so template variables can be accessed
            prompt_result = self.prompt_builder.generate(samples)

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
                    for _ in range(num_parallel_samples):
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
                    logger.info(
                        f"DEPRECATED LLMBlock '{self.block_name}' before parsing (n=1): {len(chat_result)} samples"
                    )
                    final_result = self.text_parser.generate(chat_result)
                    logger.info(
                        f"DEPRECATED LLMBlock '{self.block_name}' after parsing (n=1): {len(final_result)} samples"
                    )

                else:
                    # If no parser, just rename the raw_response column to the first output column
                    if self.output_cols:
                        final_result = chat_result.rename_column(
                            "raw_response", self.output_cols[0]
                        )
                    else:
                        final_result = chat_result

                # Step 5: Merge with original samples for n=1 case
                # Handle different parsing outcomes: expansion, contraction, or 1:1
                if len(final_result) != len(samples):
                    # Row count changed - parsing found different number of results than inputs
                    if len(final_result) > len(samples):
                        logger.info(
                            f"DEPRECATED LLMBlock '{self.block_name}' detected row expansion: {len(samples)} -> {len(final_result)}"
                        )
                    else:
                        logger.info(
                            f"DEPRECATED LLMBlock '{self.block_name}' detected row contraction: {len(samples)} -> {len(final_result)}"
                        )

                    # For both expansion and contraction, return parsed results
                    # Keep only the expected output columns plus any preserved input columns
                    # Remove intermediate processing columns to avoid duplicates
                    desired_columns = set(self.output_cols)  # Required output columns
                    available_columns = set(final_result.column_names)

                    # Add input columns that were preserved (excluding processing columns like raw_response, messages)
                    processing_columns = {
                        "raw_response",
                        "messages",
                    }  # Common intermediate columns
                    for col in available_columns:
                        if col not in processing_columns and col not in desired_columns:
                            # This is likely a preserved input column
                            desired_columns.add(col)

                    # Filter to only the columns we want
                    columns_to_keep = [
                        col
                        for col in final_result.column_names
                        if col in desired_columns
                    ]
                    final_dataset = final_result.select_columns(columns_to_keep)

                else:
                    # Normal 1:1 case - merge with original samples to preserve all input columns
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
                    final_dataset = Dataset.from_list(merged_data)

                return final_dataset

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
            if hasattr(self, "_temp_prompt_config"):
                os.unlink(self._temp_prompt_config)
        except Exception:
            pass
