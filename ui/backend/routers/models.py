# SPDX-License-Identifier: Apache-2.0
"""Model configuration and testing endpoints."""

import logging
import os
import time

from fastapi import APIRouter, HTTPException
import httpx

from models.common import ModelConfigRequest, ModelTestRequest, ModelTestResponse
from state import current_config
from utils.api_key_utils import (
    mask_api_key,
    sanitize_model_config,
    resolve_env_variable,
    validate_api_key_format,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/model/recommendations")
async def get_model_recommendations():
    """Get model recommendations for the selected flow."""
    try:
        if not current_config["flow"]:
            raise HTTPException(status_code=400, detail="No flow selected")

        flow = current_config["flow"]
        recommendations = flow.get_model_recommendations()
        default_model = flow.get_default_model()

        return {
            "default_model": default_model,
            "recommendations": recommendations,
            "requires_config": flow.is_model_config_required(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/model/configure")
async def configure_model(config: ModelConfigRequest):
    """Configure model settings for the selected flow."""
    try:
        if not current_config["flow"]:
            raise HTTPException(status_code=400, detail="No flow selected")

        flow = current_config["flow"]

        # Validate API key format
        if config.api_key:
            is_valid, error_msg = validate_api_key_format(config.api_key, config.model)
            if not is_valid:
                raise HTTPException(
                    status_code=400, detail=f"Invalid API key: {error_msg}"
                )

        # Build kwargs from config
        kwargs = {}
        if config.model:
            kwargs["model"] = config.model
        if config.api_base:
            kwargs["api_base"] = config.api_base
        if config.api_key:
            # Resolve environment variable if referenced
            resolved_key = resolve_env_variable(config.api_key)
            if resolved_key is None and config.api_key.startswith("env:"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Environment variable not found: {config.api_key[4:]}",
                )
            kwargs["api_key"] = resolved_key
        if config.blocks:
            kwargs["blocks"] = config.blocks

        # Add any additional parameters
        if config.additional_params:
            kwargs.update(config.additional_params)

        # Apply configuration
        flow.set_model_config(**kwargs)

        # Store config (keep the original reference, not resolved value)
        current_config["model_config"] = config.model_dump()

        logger.info(f"Model configured: {config.model} (API key validated)")

        # Return sanitized config (mask API key)
        safe_config = sanitize_model_config(
            current_config["model_config"], mask_key=True
        )

        return {
            "status": "success",
            "message": "Model configuration applied",
            "config": safe_config,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/model/test", response_model=ModelTestResponse)
async def test_model_connection(request: ModelTestRequest):
    """
    Test model connection with a simple prompt.

    This endpoint sends a simple test prompt to verify the model configuration
    is correct and the model is responding.
    """
    try:
        # Resolve API key if it's an environment variable reference
        api_key = request.api_key
        if api_key.startswith("env:"):
            env_var = api_key[4:]
            api_key = os.environ.get(env_var, "")
            if not api_key:
                return ModelTestResponse(
                    success=False,
                    error=f"Environment variable {env_var} not found or empty",
                )

        # Handle EMPTY key for local models
        if api_key == "EMPTY":
            api_key = "EMPTY"

        # Build the request
        api_url = request.api_base.rstrip("/") + "/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        # Extract model name (remove provider prefix if present)
        model_name = request.model
        if "/" in model_name:
            # For hosted_vllm/model-name format, use the part after the provider
            parts = model_name.split("/", 1)
            if parts[0] in [
                "hosted_vllm",
                "openai",
                "anthropic",
                "azure",
                "together",
                "anyscale",
            ]:
                model_name = parts[1]

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": request.test_prompt}],
            "max_tokens": 50,
            "temperature": 0.1,
        }

        # Send request and measure latency
        start_time = time.time()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(api_url, json=payload, headers=headers)

        latency_ms = int((time.time() - start_time) * 1000)

        if response.status_code != 200:
            return ModelTestResponse(
                success=False,
                error=f"API returned status {response.status_code}: {response.text[:200]}",
            )

        result = response.json()

        # Extract the response text
        response_text = (
            result.get("choices", [{}])[0].get("message", {}).get("content", "")
        )

        if not response_text:
            return ModelTestResponse(success=False, error="Model returned empty response")

        return ModelTestResponse(
            success=True, response=response_text.strip(), latency_ms=latency_ms
        )

    except httpx.TimeoutException:
        return ModelTestResponse(
            success=False, error="Request timed out after 30 seconds"
        )
    except httpx.ConnectError:
        return ModelTestResponse(
            success=False,
            error=f"Could not connect to {request.api_base}. Please check the URL.",
        )
    except Exception as e:
        return ModelTestResponse(success=False, error=str(e))
