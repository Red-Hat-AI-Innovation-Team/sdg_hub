# SPDX-License-Identifier: Apache-2.0
"""Configuration system for LLM blocks supporting all providers via LiteLLM."""

# Standard
from dataclasses import dataclass
from typing import Any, Optional, Union
import os


@dataclass
class LLMConfig:
    """Configuration for LLM blocks supporting all providers via LiteLLM.

    This configuration supports 100+ LLM providers including OpenAI, Anthropic,
    Google, local models (vLLM, Ollama), and more through LiteLLM.

    Parameters
    ----------
    model : Optional[str], optional
        Model identifier in LiteLLM format. Can be None initially and set later via set_model_config(). Examples:
        - "openai/gpt-4"
        - "anthropic/claude-3-sonnet-20240229"
        - "hosted_vllm/meta-llama/Llama-2-7b-chat-hf"
        - "ollama/llama2"

    api_key : Optional[str], optional
        API key for the provider. Falls back to environment variables:
        - OPENAI_API_KEY for OpenAI models
        - ANTHROPIC_API_KEY for Anthropic models
        - GOOGLE_API_KEY for Google models
        - etc.

    api_base : Optional[str], optional
        Base URL for the API. Required for local models.

    Examples
    --------
        - "http://localhost:8000/v1" for local vLLM
        - "http://localhost:11434" for Ollama

    timeout : float, optional
        Request timeout in seconds, by default 120.0

    max_retries : int, optional
        Maximum number of retry attempts, by default 6

    ### Generation Parameters ###

    temperature : Optional[float], optional
        Sampling temperature (0.0 to 2.0), by default None

    max_tokens : Optional[int], optional
        Maximum tokens to generate, by default None

    top_p : Optional[float], optional
        Nucleus sampling parameter (0.0 to 1.0), by default None

    frequency_penalty : Optional[float], optional
        Frequency penalty (-2.0 to 2.0), by default None

    presence_penalty : Optional[float], optional
        Presence penalty (-2.0 to 2.0), by default None

    stop : Optional[Union[str, List[str]]], optional
        Stop sequences, by default None

    seed : Optional[int], optional
        Random seed for reproducible outputs, by default None

    response_format : Optional[Dict[str, Any]], optional
        Response format specification (e.g., JSON mode), by default None

    stream : Optional[bool], optional
        Whether to stream responses, by default None

    n : Optional[int], optional
        Number of completions to generate, by default None

    logprobs : Optional[bool], optional
        Whether to return log probabilities, by default None

    top_logprobs : Optional[int], optional
        Number of top log probabilities to return, by default None

    user : Optional[str], optional
        End-user identifier, by default None

    extra_headers : Optional[Dict[str, str]], optional
        Additional headers to send with requests, by default None

    extra_body : Optional[Dict[str, Any]], optional
        Additional parameters for the request body, by default None

    provider_specific : Optional[Dict[str, Any]], optional
        Provider-specific parameters that don't map to standard OpenAI params, by default None
    """

    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 6

    # Generation parameters (OpenAI-compatible)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, list[str]]] = None
    seed: Optional[int] = None
    response_format: Optional[dict[str, Any]] = None
    stream: Optional[bool] = None
    n: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None

    # Additional parameters
    extra_headers: Optional[dict[str, str]] = None
    extra_body: Optional[dict[str, Any]] = None
    provider_specific: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_model()
        self._validate_parameters()
        self._resolve_api_key()

    def _validate_model(self) -> None:
        """Validate model identifier format."""
        # Model is optional - will be set later via set_model_config()
        if self.model is None:
            return

        # Check if it's a valid LiteLLM model format
        if "/" not in self.model:
            raise ValueError(
                f"Model '{self.model}' should be in format 'provider/model-name'. "
                f"Examples: 'openai/gpt-4', 'anthropic/claude-3-sonnet-20240229', "
                f"'hosted_vllm/meta-llama/Llama-2-7b-chat-hf'"
            )

    def _validate_parameters(self) -> None:
        """Validate generation parameters."""
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

        if self.top_p is not None and not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.top_p}")

        if self.frequency_penalty is not None and not (
            -2.0 <= self.frequency_penalty <= 2.0
        ):
            raise ValueError(
                f"frequency_penalty must be between -2.0 and 2.0, got {self.frequency_penalty}"
            )

        if self.presence_penalty is not None and not (
            -2.0 <= self.presence_penalty <= 2.0
        ):
            raise ValueError(
                f"presence_penalty must be between -2.0 and 2.0, got {self.presence_penalty}"
            )

        if self.n is not None and self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")

        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )

        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")

    def _resolve_api_key(self) -> None:
        """Resolve API key from environment variables if not provided.

        This method only reads from environment variables and does not modify them,
        ensuring thread-safety when multiple instances are used concurrently.
        """
        if self.api_key is not None:
            return

        # Skip API key resolution if model is not set yet
        if self.model is None:
            return

        # Extract provider from model
        provider = self.model.split("/")[0].lower()

        # Map provider to environment variable
        provider_env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "azure": "AZURE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "replicate": "REPLICATE_API_KEY",
            "together": "TOGETHER_API_KEY",
            "anyscale": "ANYSCALE_API_KEY",
            "perplexity": "PERPLEXITY_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "deepinfra": "DEEPINFRA_API_KEY",
            "ai21": "AI21_API_KEY",
            "nlp_cloud": "NLP_CLOUD_API_KEY",
            "aleph_alpha": "ALEPH_ALPHA_API_KEY",
            "bedrock": "AWS_ACCESS_KEY_ID",
            "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
        }

        env_var = provider_env_map.get(provider)
        if env_var:
            self.api_key = os.getenv(env_var)

    def get_generation_kwargs(self) -> dict[str, Any]:
        """Get generation parameters as kwargs for LiteLLM completion."""
        kwargs = {}

        # Standard parameters
        for param in [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "response_format",
            "stream",
            "n",
            "logprobs",
            "top_logprobs",
            "user",
            "timeout",
        ]:
            value = getattr(self, param)
            if value is not None:
                kwargs[param] = value

        # Additional parameters
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        if self.provider_specific:
            kwargs.update(self.provider_specific)

        return kwargs

    def merge_overrides(self, **overrides: Any) -> "LLMConfig":
        """Create a new config with runtime overrides.

        Parameters
        ----------
        **overrides : Any
            Runtime parameter overrides.

        Returns
        -------
        LLMConfig
            New configuration with overrides applied.
        """
        # Get current values as dict
        # Standard
        from dataclasses import fields

        current_values = {
            field.name: getattr(self, field.name) for field in fields(self)
        }

        # Apply overrides
        current_values.update(overrides)

        # Create new config
        return LLMConfig(**current_values)

    def get_provider(self) -> Optional[str]:
        """Get the provider name from the model identifier.

        Returns
        -------
        Optional[str]
            Provider name (e.g., "openai", "anthropic", "hosted_vllm"), or None if model is not set.
        """
        if self.model is None:
            return None
        return self.model.split("/")[0]

    def get_model_name(self) -> Optional[str]:
        """Get the model name without provider prefix.

        Returns
        -------
        Optional[str]
            Model name (e.g., "gpt-4", "claude-3-sonnet-20240229"), or None if model is not set.
        """
        if self.model is None:
            return None
        parts = self.model.split("/", 1)
        return parts[1] if len(parts) > 1 else parts[0]

    def is_local_model(self) -> bool:
        """Check if this is a local model deployment.

        Returns
        -------
        bool
            True if the model is hosted locally (vLLM, Ollama, etc.).
        """
        provider = self.get_provider()
        if provider is None:
            return False
        local_providers = {"hosted_vllm", "ollama", "local", "vllm"}
        return provider.lower() in local_providers

    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"LLMConfig(model='{self.model}', provider='{self.get_provider()}')"

    def __repr__(self) -> str:
        """Detailed representation of the configuration."""
        return (
            f"LLMConfig(model='{self.model}', provider='{self.get_provider()}', "
            f"api_base='{self.api_base}', timeout={self.timeout}, "
            f"max_retries={self.max_retries})"
        )
