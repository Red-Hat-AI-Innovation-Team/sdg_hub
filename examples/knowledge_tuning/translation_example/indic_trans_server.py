"""
FastAPI-based translation server for Indic languages.

This module implements a REST API server that provides translation services
between English and Indic languages using pretrained IndicTrans2 models.
The server exposes OpenAI-compatible endpoints for translation requests.

Models:
    - ai4bharat/indictrans2-en-indic-dist-200M: English to Indic translation
    - ai4bharat/indictrans2-indic-en-dist-200M: Indic to English translation
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from IndicTransToolkit.processor import IndicProcessor
from typing import Dict, Any

app = FastAPI()

# Load the model and tokenizer
EN_INDIC_MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
en_indic_tokenizer = AutoTokenizer.from_pretrained(
    EN_INDIC_MODEL_NAME, trust_remote_code=True
)
en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
    EN_INDIC_MODEL_NAME, trust_remote_code=True
)
en_indic_model.eval()

INDIC_EN_MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
indic_en_tokenizer = AutoTokenizer.from_pretrained(
    INDIC_EN_MODEL_NAME, trust_remote_code=True
)
indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
    INDIC_EN_MODEL_NAME, trust_remote_code=True
)
indic_en_model.eval()

ip = IndicProcessor(inference=True)


def translate_text(
    text_batch: list[str], model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer
):
    """
    Translate a batch of text using the specified model and tokenizer.

    Parameters
    ----------
    text_batch : list[str]
        List of text strings to translate.
    model : AutoModelForSeq2SeqLM
        The translation model to use.
    tokenizer : AutoTokenizer
        The tokenizer corresponding to the model.

    Returns
    -------
    list[str]
        List of translated text strings.

    Raises
    ------
    RuntimeError
        If translation fails during processing.
    """
    try:
        inputs = tokenizer(
            text_batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )

        with torch.no_grad():
            output = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        return tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    except Exception as e:
        raise RuntimeError(f"Translation failed: {str(e)}") from e


# Input format mimicking OpenAI completion endpoint
class CompletionRequest(BaseModel):
    """
    Request model for translation completion endpoints.

    Attributes
    ----------
    prompt : str
        The text to be translated.
    source_lang : str, default="kan_Knda"
        Source language code in BCP-47 format.
    target_lang : str, default="eng_Latn"
        Target language code in BCP-47 format.
    max_length : int, default=512
        Maximum length of the generated translation.
    """

    prompt: str
    source_lang: str = "kan_Knda"
    target_lang: str = "eng_Latn"
    max_length: int = 512


@app.post("/v1/completions")
async def translate_post(req: CompletionRequest) -> Dict[str, Any]:
    """
    Translate text via POST request.

    Parameters
    ----------
    req : CompletionRequest
        The request object containing prompt, source_lang, target_lang, and max_length.

    Returns
    -------
    Dict[str, Any]
        Translation response in OpenAI-compatible format or error message.
    """
    if not req.prompt or not req.prompt.strip():
        return {"error": "Empty prompt provided"}

    try:
        batch = ip.preprocess_batch(
            [req.prompt],
            src_lang=req.source_lang,
            tgt_lang=req.target_lang,
        )

        if req.source_lang == "eng_Latn":
            generated_tokens = translate_text(batch, en_indic_model, en_indic_tokenizer)
        else:
            generated_tokens = translate_text(batch, indic_en_model, indic_en_tokenizer)

        translations = ip.postprocess_batch(generated_tokens, lang=req.target_lang)

        # Return response in OpenAI-style format
        return {
            "id": "indictrans-translation",
            "object": "text_completion",
            "model": (
                INDIC_EN_MODEL_NAME
                if req.target_lang == "eng_Latn"
                else EN_INDIC_MODEL_NAME
            ),
            "choices": [
                {
                    "text": translations[0],
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        }
    except Exception as e:
        return {"error": f"Translation failed: {str(e)}"}


@app.get("/v1/completions")
async def translate_get(source_lang: str, target_lang: str, prompt: str):
    """
    Translate text via GET request.

    Parameters
    ----------
    source_lang : str
        Source language code in BCP-47 format.
    target_lang : str
        Target language code in BCP-47 format.
    prompt : str
        The text to be translated.

    Returns
    -------
    Dict[str, Any]
        Translation response in OpenAI-compatible format or error message.
    """
    # Create a request object and reuse the POST endpoint logic
    req = CompletionRequest(
        prompt=prompt, source_lang=source_lang, target_lang=target_lang
    )
    return await translate_post(req)
