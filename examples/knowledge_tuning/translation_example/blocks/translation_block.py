# First Party
from sdg_hub.logger_config import setup_logger
from sdg_hub.blocks import BlockRegistry, Block
import logging
from datasets import Dataset
from tqdm import tqdm

logger = setup_logger(__name__)


@BlockRegistry.register("TranslationBlock")
class TranslationBlock(Block):
    def __init__(
        self,
        block_name: str,
        config_path: str,
        client: object,
        output_cols: list[str],
        trans_model_id: str = None,
        source_lang: str = "eng_Latn",
        target_lang: str = "hin_Deva",
        **batch_kwargs,
    ) -> None:
        super().__init__(block_name=block_name)
        self.block_config = self._load_config(config_path)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.client = client
        if trans_model_id:
            self.trans_model_id = trans_model_id
        else:
            # get the default model id from client
            self.trans_model_id = self.client.models.list().data[0].id
        self.output_cols = output_cols
        self.defaults = {
            "temperature": 0,
            "max_tokens": 4096,
        }

        # Whether the LLM server supports a list of input prompts
        # and supports the n parameter to generate n outputs per input
        self.server_supports_batched = False

    def _translate(self, text: str) -> str:
        """
        Translate a single text string.

        Parameters
        ----------
        text : str
            Text to translate.

        Returns
        -------
        str
            Translated text, or original text if translation fails.
        """
        logging.debug(f"Translating text using model {self.trans_model_id}")

        try:
            response = self.client.completions.create(
                model=self.trans_model_id,
                prompt=text,
                extra_body={
                    "source_lang": self.source_lang,
                    "target_lang": self.target_lang,
                    "max_length": 512,
                },
            )

            return response.choices[0].text
        except Exception as e:
            logger.error(f"Translation failed with error: {str(e)}")
            return text  # Return original text as fallback

    def _translate_samples(self, samples) -> list:
        logger.debug("Starting translation...")

        results = []
        progress_bar = tqdm(range(len(samples)), desc=f"{self.block_name} Translation")
        for sample in samples:
            columns_to_translate = [
                sample[key]
                for key in self.block_config.keys()
                if key not in ("system", "generation")
            ]

            translated_texts = []

            for text in columns_to_translate:
                translated_text = self._translate(text)
                translated_texts.append(translated_text)

            results.append(translated_texts)
            progress_bar.update(1)
        return results

    def generate(self, samples: Dataset) -> Dataset:
        """
        Generate translated output from input samples.

        Parameters
        ----------
        samples : Dataset
            Input dataset containing text to translate.

        Returns
        -------
        Dataset
            Dataset with original samples plus translated columns.
        """

        # validate each sample
        # Log errors and remove invalid samples
        valid_samples = []
        for sample in samples:
            is_valid = True
            for key in self.block_config.keys():
                # we don't need these keys for translation, atleast for NLLB and IndicTrans
                if key in ["system", "generation"]:
                    continue
                if key not in sample:
                    is_valid = False

            if is_valid:
                valid_samples.append(sample)

        samples = valid_samples

        if len(samples) == 0:
            return Dataset.from_list([])

        # generate the output

        outputs = self._translate_samples(samples)

        new_data = []
        for sample, output in zip(samples, outputs):

            translated_data = {}

            index = 0
            for key in self.output_cols:
                translated_data[key] = output[index]
                index = index + 1

            new_data.append({**sample, **translated_data})

        return Dataset.from_list(new_data)
