# SPDX-License-Identifier: Apache-2.0
"""Translation block for dataset column translation operations.

This module provides a block for translating text from one language to another
using external translation services like IndicTrans2 or NLLB.
"""

# Standard
from typing import Any, Dict, List, Optional
import logging

# Third Party
from datasets import Dataset
from pydantic import Field, field_validator
from tqdm import tqdm

# Local
from ...utils.logger_config import setup_logger
from ...utils.error_handling import BlockValidationError
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "TranslationBlock",
    "translation",
    "Translation block for dataset column translation operations",
)
class TranslationBlock(BaseBlock):
    """Block for translating text from one language to another.

    This block supports translation using external services like IndicTrans2 or NLLB
    models through HTTP APIs. It can translate multiple columns simultaneously and
    supports various language pairs.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : List[str]
        Input column names containing text to translate.
    output_cols : List[str]
        Output column names for translated text.
    prompt_config_path : Optional[str]
        Path to YAML file containing prompt configuration (optional for translation).
    source_lang : str
        Source language code (e.g., "kan_Knda" for Kannada).
    target_lang : str
        Target language code (e.g., "eng_Latn" for English).
    trans_model_id : str
        Translation model identifier.
    client_url : str
        URL of the translation service endpoint.
    format_as_messages : bool
        Whether to format as chat messages (default: False for translation).
    max_length : int
        Maximum length for translation output.

    Examples
    --------
    >>> # Translate documents from Kannada to English
    >>> block = TranslationBlock(
    ...     block_name="translate_docs",
    ...     input_cols=["title", "text"],
    ...     output_cols=["title_translated", "text_translated"],
    ...     source_lang="kan_Knda",
    ...     target_lang="eng_Latn",
    ...     trans_model_id="ai4bharat/indictrans2-indic-en-dist-200M",
    ...     client_url="http://localhost:8000/v1"
    ... )
    """

    # Translation-specific parameters
    source_lang: str = Field(default="eng_Latn", description="Source language code")
    target_lang: str = Field(default="hin_Deva", description="Target language code")
    trans_model_id: str = Field(description="Translation model identifier")
    client_url: str = Field(description="URL of the translation service endpoint")
    max_length: int = Field(
        default=512, description="Maximum length for translation output"
    )
    format_as_messages: bool = Field(
        default=False, description="Whether to format as chat messages"
    )

    # Optional prompt configuration
    prompt_config_path: Optional[str] = Field(
        None, description="Path to YAML file containing prompt configuration"
    )

    # Internal client (excluded from serialization)
    client: Optional[Any] = Field(
        None, exclude=True, description="Internal translation client"
    )

    def model_post_init(self, __context) -> None:
        """Initialize after Pydantic validation."""
        super().model_post_init(__context)
        self._setup_client()

    def _setup_client(self) -> None:
        """Set up the translation client."""
        try:
            # Import OpenAI client for HTTP API communication
            from openai import OpenAI

            self._client = OpenAI(base_url=self.client_url, api_key="EMPTY")

            logger.info(
                f"Initialized TranslationBlock '{self.block_name}' with model '{self.trans_model_id}'",
                extra={
                    "block_name": self.block_name,
                    "model": self.trans_model_id,
                    "source_lang": self.source_lang,
                    "target_lang": self.target_lang,
                    "client_url": self.client_url,
                },
            )
        except ImportError as e:
            raise BlockValidationError(
                f"OpenAI client not available for TranslationBlock '{self.block_name}': {e}"
            )

    def _translate_text(self, text: str) -> str:
        """Translate a single text string.

        Parameters
        ----------
        text : str
            Text to translate.

        Returns
        -------
        str
            Translated text, or original text if translation fails.
        """
        if not text or not text.strip():
            return text

        try:
            response = self._client.completions.create(
                model=self.trans_model_id,
                prompt=text,
                extra_body={
                    "source_lang": self.source_lang,
                    "target_lang": self.target_lang,
                    "max_length": self.max_length,
                },
            )
            logger.info(f"Translation response: {response}")

            # Check if response contains an error
            if hasattr(response, "error") and response.error:
                raise Exception(f"Server returned error: {response.error}")

            # Check if choices exist and are not None
            if not response.choices or len(response.choices) == 0:
                raise Exception("No translation choices returned from server")

            return response.choices[0].text.strip()
        except Exception as e:
            logger.error(
                f"Translation failed for text: {str(e)}",
                extra={
                    "block_name": self.block_name,
                    "model": self.trans_model_id,
                    "error": str(e),
                },
            )
            return text  # Return original text as fallback

    def _translate_samples(self, samples: List[Dict[str, Any]]) -> List[List[str]]:
        """Translate multiple samples.

        Parameters
        ----------
        samples : List[Dict[str, Any]]
            List of samples to translate.

        Returns
        -------
        List[List[str]]
            List of translated text lists for each sample.
        """
        results = []
        progress_bar = tqdm(
            samples,
            desc=f"{self.block_name} Translation",
            disable=len(samples) < 10,  # Only show progress bar for larger batches
        )

        for sample in progress_bar:
            translated_texts = []

            for col in self.input_cols:
                if col in sample:
                    text = sample[col]
                    translated_text = self._translate_text(str(text))
                    translated_texts.append(translated_text)
                else:
                    logger.warning(
                        f"Column '{col}' not found in sample, skipping translation",
                        extra={"block_name": self.block_name, "missing_column": col},
                    )
                    translated_texts.append("")

            results.append(translated_texts)

        return results

    def generate(self, samples: Dataset) -> Dataset:
        """Generate translated output from input samples.

        Parameters
        ----------
        samples : Dataset
            Input dataset containing text to translate.

        Returns
        -------
        Dataset
            Dataset with original samples plus translated columns.

        Raises
        ------
        BlockValidationError
            If validation fails or translation service is unavailable.
        """
        if not self._client:
            raise BlockValidationError(
                f"Translation client not initialized for block '{self.block_name}'"
            )

        # Convert to list for processing
        samples_list = list(samples)

        if not samples_list:
            return Dataset.from_list([])

        logger.info(
            f"Starting translation for {len(samples_list)} samples",
            extra={
                "block_name": self.block_name,
                "model": self.trans_model_id,
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "input_cols": self.input_cols,
                "output_cols": self.output_cols,
            },
        )

        # Translate samples
        translated_outputs = self._translate_samples(samples_list)

        # Combine original samples with translations
        new_data = []
        for sample, translations in zip(samples_list, translated_outputs):
            translated_data = {}

            # Add translations to output columns
            for i, output_col in enumerate(self.output_cols):
                if i < len(translations):
                    translated_data[output_col] = translations[i]
                else:
                    translated_data[output_col] = ""

            # Combine with original sample
            new_data.append({**sample, **translated_data})

        logger.info(
            f"Translation completed for {len(new_data)} samples",
            extra={
                "block_name": self.block_name,
                "model": self.trans_model_id,
                "output_samples": len(new_data),
            },
        )

        return Dataset.from_list(new_data)

    @field_validator("input_cols", "output_cols")
    @classmethod
    def validate_column_lists(cls, v):
        """Validate that column lists are not empty."""
        if not v:
            raise ValueError("Column lists cannot be empty")
        return v

    def _validate_custom(self, dataset: Dataset) -> None:
        """Custom validation for TranslationBlock.

        Parameters
        ----------
        dataset : Dataset
            The dataset to validate.

        Raises
        ------
        BlockValidationError
            If validation fails.
        """
        # Validate input columns exist
        missing_cols = [
            col for col in self.input_cols if col not in dataset.column_names
        ]
        if missing_cols:
            raise BlockValidationError(
                f"Missing input columns in dataset: {missing_cols}",
                details=f"Block: {self.block_name}, Available columns: {dataset.column_names}",
            )

        # Validate input/output column count match
        if len(self.input_cols) != len(self.output_cols):
            raise BlockValidationError(
                f"Number of input columns ({len(self.input_cols)}) must match "
                f"number of output columns ({len(self.output_cols)})",
                details=f"Block: {self.block_name}, Input: {self.input_cols}, Output: {self.output_cols}",
            )

    def __repr__(self) -> str:
        """String representation of the block."""
        return (
            f"TranslationBlock(name='{self.block_name}', "
            f"model='{self.trans_model_id}', {self.source_lang}->{self.target_lang})"
        )
