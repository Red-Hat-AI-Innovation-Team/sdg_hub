"""
FastAPI server for translation using Facebook's NLLB model.

This module provides translation services through REST endpoints compatible
with OpenAI's completion API format. It supports multiple language pairs
using the NLLB-200-1.3B model from Hugging Face Transformers.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Dict, Any

app = FastAPI()

# Load the model and tokenizer
MODEL_NAME = "facebook/nllb-200-1.3B"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to load model {MODEL_NAME}: {str(e)}")


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
    source_lang: str = "eng_Latn"
    target_lang: str = "hin_Deva"
    max_length: int = 512


async def _perform_translation(
    prompt: str, source_lang: str, target_lang: str, max_length: int = 512
) -> Dict[str, Any]:
    """Core translation logic shared between endpoints.

    Parameters
    ----------
    prompt : str
        Text to translate.
    source_lang : str
        Source language code.
    target_lang : str
        Target language code.
    max_length : int, default=512
        Maximum length of translation.

    Returns
    -------
    Dict[str, Any]
        Translation response in OpenAI format.
    """
    # Validate language codes
    if (
        source_lang not in tokenizer.lang_code_to_id
        or target_lang not in tokenizer.lang_code_to_id
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid language code. Available codes: "
                f"{list(tokenizer.lang_code_to_id.keys())}"
            ),
        )

    # Set source language
    tokenizer.src_lang = source_lang

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
            max_length=max_length,
        )

    # Decode and format response
    translated_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    return {
        "id": "nllb-translation",
        "object": "text_completion",
        "model": MODEL_NAME,
        "choices": [
            {
                "text": translated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/v1/completions")
async def translate(req: CompletionRequest) -> Dict[str, Any]:
    """
    Translate text via POST request.

    Parameters
    ----------
    req : CompletionRequest
        The translation request containing prompt and language parameters.

    Returns
    -------
    Dict[str, Any]
        Translation response in OpenAI-compatible format or error message.

    Raises
    ------
    HTTPException
        400 for invalid language codes, 500 for translation failures.
    """
    try:
        return await _perform_translation(req.prompt, req.source_lang, req.target_lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


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
    try:
        return await _perform_translation(prompt, source_lang, target_lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
