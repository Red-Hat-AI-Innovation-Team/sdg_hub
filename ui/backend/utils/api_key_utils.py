# SPDX-License-Identifier: Apache-2.0
"""API key handling: masking, sanitization, env resolution, validation."""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def mask_api_key(key: Optional[str]) -> str:
    """Mask an API key for safe display."""
    if not key:
        return ""
    if key == "EMPTY":
        return "EMPTY"
    if len(key) > 8:
        return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"
    else:
        return "*" * len(key)


def sanitize_model_config(
    config: Dict[str, Any], mask_key: bool = True
) -> Dict[str, Any]:
    """Remove or mask sensitive information from model configuration."""
    if not config:
        return {}

    sanitized = config.copy()

    if "api_key" in sanitized:
        api_key = sanitized["api_key"]

        is_safe_value = (
            api_key in ["EMPTY", "NONE", ""]
            or api_key.startswith("env:")
        )

        if is_safe_value:
            pass
        elif mask_key:
            sanitized["api_key"] = mask_api_key(api_key)
        else:
            del sanitized["api_key"]

    return sanitized


def resolve_env_variable(value: str) -> Optional[str]:
    """Resolve environment variable references like 'env:OPENAI_API_KEY'."""
    if not value:
        return value

    if value.startswith("env:"):
        env_var_name = value[4:]
        env_value = os.getenv(env_var_name)
        if env_value:
            logger.info(f"✅ Resolved environment variable: {env_var_name}")
            return env_value
        else:
            logger.warning(f"⚠️ Environment variable not found: {env_var_name}")
            return None

    return value


def get_safe_api_key(config: Dict[str, Any]) -> Optional[str]:
    """Get API key from config, resolving env vars if needed."""
    api_key = config.get("api_key")
    if not api_key:
        return None
    return resolve_env_variable(api_key)


def validate_api_key_format(
    api_key: str, provider: Optional[str] = None
) -> tuple[bool, Optional[str]]:
    """Validate API key format (not functionality).

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is required"

    if api_key in ["EMPTY", "NONE"]:
        return True, None

    if api_key.startswith("env:"):
        env_var_name = api_key[4:]
        if not env_var_name:
            return False, "Environment variable name cannot be empty"
        if not env_var_name.replace("_", "").isalnum():
            return False, "Invalid environment variable name"
        return True, None

    if len(api_key) < 8:
        return False, "API key too short (minimum 8 characters)"

    if len(api_key) > 512:
        return False, "API key too long (maximum 512 characters)"

    if provider:
        provider_lower = provider.lower()

        if "openai" in provider_lower:
            if not (api_key.startswith("sk-") or api_key.startswith("sess-")):
                return False, "OpenAI keys should start with 'sk-' or 'sess-'"

        elif "anthropic" in provider_lower:
            if not api_key.startswith("sk-ant-"):
                return False, "Anthropic keys should start with 'sk-ant-'"

        elif "cohere" in provider_lower:
            if len(api_key) < 40:
                return False, "Cohere keys are typically longer"

    if api_key in ["your-api-key", "your-key-here", "test", "example"]:
        return False, "Please replace placeholder with actual API key"

    return True, None
