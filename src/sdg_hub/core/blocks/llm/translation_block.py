# SPDX-License-Identifier: Apache-2.0
"""Translation block for multilingual SDG using LLM-based translation."""

# Standard
from typing import Any, Dict, List, Optional
import asyncio
import os

from jinja2 import Template
from litellm import acompletion, completion
from pydantic import ConfigDict, Field, field_validator
import litellm
import pandas as pd

from ...utils.error_handling import BlockValidationError
from ...utils.logger_config import setup_logger

# Local
from ..base import BaseBlock
from ..registry import BlockRegistry

litellm.drop_params = True
logger = setup_logger(__name__)

# ISO 639-1 language code to full name mapping
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "it": "Italian",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "he": "Hebrew",
}


@BlockRegistry.register(
    "TranslationBlock",
    "llm",
    "Translates prompts from source to target language using LLM",
)
class TranslationBlock(BaseBlock):
    """Translates prompt content from source language to target language.

    This block uses an LLM to intelligently translate prompt scaffolding while
    preserving content that is already in the target language. It's designed for
    multilingual synthetic data generation workflows where templates need translation
    but injected content (documents, ICL examples) should remain unchanged.

    Parameters
    ----------
    block_name : str
        Unique identifier for the block.
    input_cols : str
        Input column containing prompts (messages or text format).
    output_cols : str
        Output column for translated prompts.
    source_language : str
        Source language ISO 639-1 code (e.g., "en" for English).
    target_language : str
        Target language ISO 639-1 code (e.g., "es" for Spanish).
    model : Optional[str], optional
        LiteLLM model identifier. Can be set via flow.set_model_config().
    api_key : Optional[str], optional
        API key for the provider. Falls back to environment variables.
    api_base : Optional[str], optional
        Base URL for the API. Required for local models.
    async_mode : bool, optional
        Whether to use async processing, by default False.
    timeout : float, optional
        Request timeout in seconds, by default 120.0.
    num_retries : int, optional
        Number of retry attempts, by default 6.
    drop_params : bool, optional
        Whether to drop unsupported parameters, by default True.
    **kwargs : Any
        Additional LiteLLM completion parameters (temperature, max_tokens, etc.).

    Examples
    --------
    >>> # Translate English prompts to Spanish
    >>> block = TranslationBlock(
    ...     block_name="translate_to_spanish",
    ...     input_cols="english_prompt",
    ...     output_cols="spanish_prompt",
    ...     source_language="en",
    ...     target_language="es",
    ...     model="openai/gpt-4",
    ...     temperature=0.3
    ... )

    >>> # In a flow YAML
    >>> # - block_type: TranslationBlock
    >>> #   block_config:
    >>> #     block_name: translate_prompt
    >>> #     input_cols: english_prompt
    >>> #     output_cols: spanish_prompt
    >>> #     source_language: "en"
    >>> #     target_language: "es"
    >>> #     temperature: 0.3
    """

    model_config = ConfigDict(extra="allow")

    # Translation-specific fields
    source_language: str = Field(
        ..., description="Source language ISO 639-1 code (e.g., 'en')"
    )
    target_language: str = Field(
        ..., description="Target language ISO 639-1 code (e.g., 'es')"
    )

    # LiteLLM fields (same pattern as LLMChatBlock - all exclude=True)
    model: Optional[str] = Field(
        None, exclude=True, description="Model identifier in LiteLLM format"
    )
    api_key: Optional[str] = Field(
        None, exclude=True, description="API key for the provider"
    )
    api_base: Optional[str] = Field(
        None, exclude=True, description="Base URL for the API"
    )
    async_mode: bool = Field(
        False, exclude=True, description="Whether to use async processing"
    )
    timeout: float = Field(
        120.0, exclude=True, description="Request timeout in seconds"
    )
    num_retries: int = Field(
        6, exclude=True, description="Number of retry attempts"
    )
    drop_params: bool = Field(
        True, description="Whether to drop unsupported parameters"
    )

    @field_validator("source_language", "target_language")
    @classmethod
    def validate_language_code(cls, v: str) -> str:
        """Validate that language code is supported."""
        if v not in LANGUAGE_NAMES:
            logger.warning(
                "Language code '%s' not in predefined list. Proceeding anyway. "
                "Supported codes: %s",
                v,
                ", ".join(sorted(LANGUAGE_NAMES.keys())),
            )
        return v

    @field_validator("input_cols")
    @classmethod
    def validate_single_input_col(cls, v):
        """Ensure exactly one input column."""
        if isinstance(v, str):
            return [v]
        if isinstance(v, list) and len(v) == 1:
            return v
        if isinstance(v, list) and len(v) != 1:
            raise ValueError(
                f"TranslationBlock expects exactly one input column, got {len(v)}: {v}"
            )
        raise ValueError(f"Invalid input_cols format: {v}")

    @field_validator("output_cols")
    @classmethod
    def validate_single_output_col(cls, v):
        """Ensure exactly one output column."""
        if isinstance(v, str):
            return [v]
        if isinstance(v, list) and len(v) == 1:
            return v
        if isinstance(v, list) and len(v) != 1:
            raise ValueError(
                f"TranslationBlock expects exactly one output column, got {len(v)}: {v}"
            )
        raise ValueError(f"Invalid output_cols format: {v}")

    def model_post_init(self, __context) -> None:
        """Initialize after Pydantic validation."""
        super().model_post_init(__context)

        # Load translation prompt template
        template_path = os.path.join(
            os.path.dirname(__file__), "prompts", "translate.yaml"
        )
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                import yaml

                self._translation_template = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise BlockValidationError(
                f"Translation prompt template not found at {template_path}"
            ) from e

        # Log initialization only when model is configured
        if self.model:
            logger.info(
                "Initialized TranslationBlock '%s' translating %s → %s with model '%s'",
                self.block_name,
                self.source_language,
                self.target_language,
                self.model,
                extra={
                    "block_name": self.block_name,
                    "source_language": self.source_language,
                    "target_language": self.target_language,
                    "model": self.model,
                    "async_mode": self.async_mode,
                },
            )

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate translated prompts.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset containing the prompts column.
        **kwargs : Any
            Runtime parameters that override initialization defaults.

        Returns
        -------
        pd.DataFrame
            Dataset with translated prompts added to the output column.

        Raises
        ------
        BlockValidationError
            If model is not configured before calling generate().
        """
        # Validate that model is configured
        if not self.model:
            raise BlockValidationError(
                f"Model not configured for block '{self.block_name}'. "
                f"Call flow.set_model_config() before generating."
            )

        # Extract flow-specific parameters
        flow_max_concurrency = kwargs.pop("_flow_max_concurrency", None)

        # Build completion kwargs from ALL fields + runtime overrides
        completion_kwargs = self._build_completion_kwargs(**kwargs)

        # Extract prompts from pandas DataFrame
        prompts_list = samples[self.input_cols[0]].tolist()

        # Log generation start
        logger.info(
            "Starting %s translation for %d samples (%s → %s)%s",
            "async" if self.async_mode else "sync",
            len(prompts_list),
            self.source_language,
            self.target_language,
            (
                f" (max_concurrency={flow_max_concurrency})"
                if flow_max_concurrency
                else ""
            ),
            extra={
                "block_name": self.block_name,
                "batch_size": len(prompts_list),
                "async_mode": self.async_mode,
                "flow_max_concurrency": flow_max_concurrency,
            },
        )

        # Generate translations
        if self.async_mode:
            try:
                # Check if there's already a running event loop
                loop = asyncio.get_running_loop()
                # Check if nest_asyncio is applied
                nest_asyncio_applied = (
                    hasattr(loop, "_nest_patched")
                    or getattr(asyncio.run, "__module__", "") == "nest_asyncio"
                )

                if nest_asyncio_applied:
                    translations = asyncio.run(
                        self._generate_async(
                            prompts_list, completion_kwargs, flow_max_concurrency
                        )
                    )
                else:
                    raise BlockValidationError(
                        f"async_mode=True cannot be used from within a running event loop for '{self.block_name}'. "
                        "Use an async entrypoint, set async_mode=False, or apply nest_asyncio.apply() in notebook environments."
                    )
            except RuntimeError:
                # No running loop; safe to create one
                translations = asyncio.run(
                    self._generate_async(
                        prompts_list, completion_kwargs, flow_max_concurrency
                    )
                )
        else:
            translations = self._generate_sync(prompts_list, completion_kwargs)

        # Log completion
        logger.info(
            "Translation completed successfully for %d samples",
            len(translations),
            extra={
                "block_name": self.block_name,
                "batch_size": len(translations),
            },
        )

        # Add translations as new column
        result = samples.copy()
        result[self.output_cols[0]] = translations
        return result

    def _build_completion_kwargs(self, **overrides) -> dict[str, Any]:
        """Build kwargs for LiteLLM completion call.

        Returns
        -------
        dict[str, Any]
            Kwargs for litellm.completion() or litellm.acompletion().
        """
        # Start with extra fields (temperature, max_tokens, etc.)
        extra_values = self.model_dump(exclude_unset=True)

        # Remove block-operational fields that shouldn't go to LiteLLM
        block_only_fields = {
            "block_name",
            "input_cols",
            "output_cols",
            "async_mode",
            "source_language",
            "target_language",
        }

        completion_kwargs = {
            k: v for k, v in extra_values.items() if k not in block_only_fields
        }

        # Add essential LiteLLM fields
        if self.model is not None:
            completion_kwargs["model"] = self.model
        if self.api_key is not None:
            completion_kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            completion_kwargs["api_base"] = self.api_base
        if self.timeout is not None:
            completion_kwargs["timeout"] = self.timeout
        if self.num_retries is not None:
            completion_kwargs["num_retries"] = self.num_retries

        # Apply non-block-field overrides
        non_block_overrides = {
            k: v for k, v in overrides.items() if k not in self.__class__.model_fields
        }
        completion_kwargs.update(non_block_overrides)

        # Ensure drop_params is set
        completion_kwargs["drop_params"] = self.drop_params

        return completion_kwargs

    def _detect_input_format(self, input_data: Any) -> str:
        """Detect if input is messages format or plain text.

        Parameters
        ----------
        input_data : Any
            Input data to check.

        Returns
        -------
        str
            Either "messages" or "text".

        Raises
        ------
        BlockValidationError
            If input format is unsupported.
        """
        if isinstance(input_data, list) and all(
            isinstance(msg, dict) and "role" in msg and "content" in msg
            for msg in input_data
        ):
            return "messages"
        elif isinstance(input_data, str):
            return "text"
        else:
            raise BlockValidationError(
                f"Unsupported input format: {type(input_data)}. "
                f"Expected list of messages or string."
            )

    def _build_translation_prompt(self, content: str) -> List[Dict[str, str]]:
        """Build translation prompt from template.

        Parameters
        ----------
        content : str
            Text content to translate.

        Returns
        -------
        List[Dict[str, str]]
            Messages list for LLM completion.
        """
        # Get language names
        source_lang_name = LANGUAGE_NAMES.get(
            self.source_language, self.source_language.title()
        )
        target_lang_name = LANGUAGE_NAMES.get(
            self.target_language, self.target_language.title()
        )

        # Render template
        messages = []
        for msg_template in self._translation_template:
            content_template = Template(msg_template["content"])
            rendered_content = content_template.render(
                source_language_name=source_lang_name,
                target_language_name=target_lang_name,
                content=content,
            ).strip()
            messages.append({"role": msg_template["role"], "content": rendered_content})

        return messages

    def _translate_text(self, text: str, completion_kwargs: dict[str, Any]) -> str:
        """Translate a single text string using LLM.

        Parameters
        ----------
        text : str
            Text to translate.
        completion_kwargs : dict[str, Any]
            Kwargs for LiteLLM completion.

        Returns
        -------
        str
            Translated text.
        """
        # Build translation prompt
        messages = self._build_translation_prompt(text)

        # Call LLM
        response = completion(messages=messages, **completion_kwargs)

        # Extract translated text
        translated_text = response.choices[0].message.content

        return translated_text

    async def _translate_text_async(
        self, text: str, completion_kwargs: dict[str, Any]
    ) -> str:
        """Translate a single text string using LLM (async).

        Parameters
        ----------
        text : str
            Text to translate.
        completion_kwargs : dict[str, Any]
            Kwargs for LiteLLM acompletion.

        Returns
        -------
        str
            Translated text.
        """
        # Build translation prompt
        messages = self._build_translation_prompt(text)

        # Call LLM
        response = await acompletion(messages=messages, **completion_kwargs)

        # Extract translated text
        translated_text = response.choices[0].message.content

        return translated_text

    def _translate_messages(
        self, messages: List[Dict[str, str]], completion_kwargs: dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Translate each message's content while preserving roles.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of messages to translate.
        completion_kwargs : dict[str, Any]
            Kwargs for LiteLLM completion.

        Returns
        -------
        List[Dict[str, str]]
            Translated messages.
        """
        translated_messages = []
        for msg in messages:
            translated_content = self._translate_text(msg["content"], completion_kwargs)
            translated_messages.append(
                {"role": msg["role"], "content": translated_content}
            )
        return translated_messages

    async def _translate_messages_async(
        self, messages: List[Dict[str, str]], completion_kwargs: dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Translate each message's content while preserving roles (async).

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of messages to translate.
        completion_kwargs : dict[str, Any]
            Kwargs for LiteLLM acompletion.

        Returns
        -------
        List[Dict[str, str]]
            Translated messages.
        """
        translated_messages = []
        for msg in messages:
            translated_content = await self._translate_text_async(
                msg["content"], completion_kwargs
            )
            translated_messages.append(
                {"role": msg["role"], "content": translated_content}
            )
        return translated_messages

    def _generate_sync(
        self,
        prompts_list: List[Any],
        completion_kwargs: dict[str, Any],
    ) -> List[Any]:
        """Generate translations synchronously.

        Parameters
        ----------
        prompts_list : List[Any]
            List of prompts to translate.
        completion_kwargs : dict[str, Any]
            Kwargs for LiteLLM completion.

        Returns
        -------
        List[Any]
            List of translated prompts.
        """
        translations = []

        for i, prompt in enumerate(prompts_list):
            try:
                # Detect format
                format_type = self._detect_input_format(prompt)

                # Translate based on format
                if format_type == "messages":
                    translation = self._translate_messages(prompt, completion_kwargs)
                else:  # text
                    translation = self._translate_text(prompt, completion_kwargs)

                translations.append(translation)

                # Log progress for large batches
                if (i + 1) % 10 == 0:
                    logger.debug(
                        "Translated %d/%d prompts",
                        i + 1,
                        len(prompts_list),
                        extra={
                            "block_name": self.block_name,
                            "progress": f"{i + 1}/{len(prompts_list)}",
                        },
                    )

            except Exception as e:
                logger.error(
                    "Failed to translate prompt %d: %s",
                    i,
                    str(e),
                    extra={
                        "block_name": self.block_name,
                        "sample_index": i,
                        "error": str(e),
                    },
                )
                raise

        return translations

    async def _generate_async(
        self,
        prompts_list: List[Any],
        completion_kwargs: dict[str, Any],
        flow_max_concurrency: Optional[int] = None,
    ) -> List[Any]:
        """Generate translations asynchronously.

        Parameters
        ----------
        prompts_list : List[Any]
            List of prompts to translate.
        completion_kwargs : dict[str, Any]
            Kwargs for LiteLLM acompletion.
        flow_max_concurrency : Optional[int], optional
            Maximum concurrency for async requests.

        Returns
        -------
        List[Any]
            List of translated prompts.
        """

        async def translate_single(prompt: Any, semaphore: Optional[asyncio.Semaphore] = None) -> Any:
            """Translate a single prompt with optional concurrency control."""
            # Detect format
            format_type = self._detect_input_format(prompt)

            # Translate based on format
            if semaphore:
                async with semaphore:
                    if format_type == "messages":
                        return await self._translate_messages_async(
                            prompt, completion_kwargs
                        )
                    else:  # text
                        return await self._translate_text_async(
                            prompt, completion_kwargs
                        )
            else:
                if format_type == "messages":
                    return await self._translate_messages_async(
                        prompt, completion_kwargs
                    )
                else:  # text
                    return await self._translate_text_async(prompt, completion_kwargs)

        try:
            if flow_max_concurrency is not None:
                # Validate max_concurrency parameter
                if flow_max_concurrency < 1:
                    raise ValueError(
                        f"max_concurrency must be greater than 0, got {flow_max_concurrency}"
                    )

                # Use semaphore for concurrency control
                semaphore = asyncio.Semaphore(flow_max_concurrency)
                tasks = [
                    translate_single(prompt, semaphore) for prompt in prompts_list
                ]
            else:
                # No concurrency limit
                tasks = [translate_single(prompt) for prompt in prompts_list]

            translations = await asyncio.gather(*tasks)
            return translations

        except Exception as e:
            logger.error(
                "Failed to generate async translations: %s",
                str(e),
                extra={
                    "block_name": self.block_name,
                    "batch_size": len(prompts_list),
                    "error": str(e),
                },
            )
            raise

    def _validate_custom(self, dataset: pd.DataFrame) -> None:
        """Custom validation for TranslationBlock input format.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to validate.

        Raises
        ------
        BlockValidationError
            If input format validation fails.
        """
        prompts_col = dataset[self.input_cols[0]]

        # Check that all prompts are either messages or text
        for idx, prompt in prompts_col.items():
            try:
                self._detect_input_format(prompt)
            except BlockValidationError as e:
                raise BlockValidationError(
                    f"Invalid prompt format in row {idx}: {str(e)}",
                    details=f"Block: {self.block_name}, Row: {idx}, Value type: {type(prompt)}",
                ) from e

    def __repr__(self) -> str:
        """String representation of the block."""
        return (
            f"TranslationBlock(name='{self.block_name}', "
            f"{self.source_language}→{self.target_language}, "
            f"model='{self.model}', async_mode={self.async_mode})"
        )
