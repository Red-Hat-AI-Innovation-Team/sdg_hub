# SPDX-License-Identifier: Apache-2.0
"""Error handling system for LLM blocks supporting multiple providers."""

# Standard
from enum import Enum
from typing import Any, Optional

# Third Party
from litellm import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    UnprocessableEntityError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Local
from ...utils.logger_config import setup_logger

logger = setup_logger(__name__)


class ErrorCategory(Enum):
    """Categories of errors for different retry strategies."""

    RETRYABLE_RATE_LIMIT = "rate_limit"
    RETRYABLE_TIMEOUT = "timeout"
    RETRYABLE_CONNECTION = "connection"
    RETRYABLE_SERVER = "server_error"
    RETRYABLE_CONTENT_FILTER = "content_filter"

    NON_RETRYABLE_AUTH = "auth_error"
    NON_RETRYABLE_PERMISSION = "permission"
    NON_RETRYABLE_BAD_REQUEST = "bad_request"
    NON_RETRYABLE_NOT_FOUND = "not_found"
    NON_RETRYABLE_CONTEXT_LENGTH = "context_length"

    UNKNOWN = "unknown"


class LLMErrorHandler:
    """Centralized error handling for LLM operations across all providers.

    This class handles errors from multiple LLM providers through LiteLLM,
    which maps provider-specific errors to OpenAI-compatible exceptions.

    Parameters
    ----------
    max_retries : int, optional
        Maximum number of retry attempts, by default 6
    base_delay : float, optional
        Base delay between retries in seconds, by default 1.0
    max_delay : float, optional
        Maximum delay between retries in seconds, by default 60.0
    exponential_base : float, optional
        Base for exponential backoff, by default 2.0
    """

    def __init__(
        self,
        max_retries: int = 6,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

        # Error category mappings
        self.error_mappings = {
            # Rate limiting errors
            RateLimitError: ErrorCategory.RETRYABLE_RATE_LIMIT,
            # Connection errors
            APIConnectionError: ErrorCategory.RETRYABLE_CONNECTION,
            # Server errors (5xx)
            InternalServerError: ErrorCategory.RETRYABLE_SERVER,
            ServiceUnavailableError: ErrorCategory.RETRYABLE_SERVER,
            # Content filter errors (might be retryable with different input)
            ContentPolicyViolationError: ErrorCategory.RETRYABLE_CONTENT_FILTER,
            # Authentication errors (non-retryable)
            AuthenticationError: ErrorCategory.NON_RETRYABLE_AUTH,
            # Bad request errors (non-retryable)
            BadRequestError: ErrorCategory.NON_RETRYABLE_BAD_REQUEST,
            InvalidRequestError: ErrorCategory.NON_RETRYABLE_BAD_REQUEST,
            UnprocessableEntityError: ErrorCategory.NON_RETRYABLE_BAD_REQUEST,
            # Not found errors (non-retryable)
            NotFoundError: ErrorCategory.NON_RETRYABLE_NOT_FOUND,
            # Context length errors (non-retryable)
            ContextWindowExceededError: ErrorCategory.NON_RETRYABLE_CONTEXT_LENGTH,
        }

        # Retryable error types
        self.retryable_errors = {
            ErrorCategory.RETRYABLE_RATE_LIMIT,
            ErrorCategory.RETRYABLE_TIMEOUT,
            ErrorCategory.RETRYABLE_CONNECTION,
            ErrorCategory.RETRYABLE_SERVER,
            ErrorCategory.RETRYABLE_CONTENT_FILTER,
        }

    def classify_error(self, error: Exception) -> ErrorCategory:
        """Classify an error into a category for retry logic.

        Parameters
        ----------
        error : Exception
            The error to classify.

        Returns
        -------
        ErrorCategory
            The category of the error.
        """
        error_type = type(error)
        return self.error_mappings.get(error_type, ErrorCategory.UNKNOWN)

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an error should be retried.

        Parameters
        ----------
        error : Exception
            The error that occurred.
        attempt : int
            The current attempt number (1-based).

        Returns
        -------
        bool
            True if the error should be retried.
        """
        if attempt >= self.max_retries:
            return False

        category = self.classify_error(error)
        return category in self.retryable_errors

    def calculate_delay(self, error: Exception, attempt: int) -> float:
        """Calculate the delay before the next retry.

        Parameters
        ----------
        error : Exception
            The error that occurred.
        attempt : int
            The current attempt number (1-based).

        Returns
        -------
        float
            Delay in seconds before the next retry.
        """
        category = self.classify_error(error)

        if category == ErrorCategory.RETRYABLE_RATE_LIMIT:
            # Longer delays for rate limiting
            delay = min(
                self.base_delay * (self.exponential_base**attempt) * 2,
                self.max_delay * 2,
            )
        elif category == ErrorCategory.RETRYABLE_TIMEOUT:
            # Shorter delays for timeouts
            delay = min(
                self.base_delay * (self.exponential_base ** (attempt - 1)),
                self.max_delay * 0.5,
            )
        else:
            # Standard exponential backoff
            delay = min(
                self.base_delay * (self.exponential_base ** (attempt - 1)),
                self.max_delay,
            )

        return delay

    def log_error_context(
        self, error: Exception, context: dict[str, Any], attempt: int = 1
    ) -> None:
        """Log error with context information.

        Parameters
        ----------
        error : Exception
            The error that occurred.
        context : Dict[str, Any]
            Context information about the error.
        attempt : int, optional
            The current attempt number, by default 1.
        """
        category = self.classify_error(error)

        log_data = {
            "error_type": type(error).__name__,
            "error_category": category.value,
            "error_message": str(error),
            "attempt": attempt,
            "max_retries": self.max_retries,
            "retryable": category in self.retryable_errors,
            **context,
        }

        if category in self.retryable_errors and attempt < self.max_retries:
            delay = self.calculate_delay(error, attempt)
            log_data["retry_delay"] = delay
            logger.warning(
                f"Retryable error occurred (attempt {attempt}/{self.max_retries}). "
                f"Retrying in {delay:.1f}s: {error}",
                extra=log_data,
            )
        else:
            logger.error(
                f"Non-retryable error or max retries exceeded: {error}", extra=log_data
            )

    def create_retry_decorator(self, context: Optional[dict[str, Any]] = None):
        """Create a retry decorator for LLM operations.

        Parameters
        ----------
        context : Optional[Dict[str, Any]], optional
            Context information for logging, by default None.

        Returns
        -------
        Callable
            A retry decorator configured for LLM operations.
        """
        context = context or {}

        def retry_condition(retry_state):
            """Custom retry condition that logs errors."""
            if retry_state.outcome.failed:
                error = retry_state.outcome.exception()
                self.log_error_context(error, context, retry_state.attempt_number)
                return self.should_retry(error, retry_state.attempt_number)
            return False

        def wait_strategy(retry_state):
            """Custom wait strategy based on error type."""
            if retry_state.outcome.failed:
                error = retry_state.outcome.exception()
                return self.calculate_delay(error, retry_state.attempt_number)
            return 0

        return retry(
            retry=retry_condition,
            wait=wait_strategy,
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )

    def create_simple_retry_decorator(self):
        """Create a simple retry decorator using tenacity's built-in strategies.

        This is a simpler alternative when you don't need custom error handling logic.

        Returns
        -------
        Callable
            A simple retry decorator for LLM operations.
        """
        # Define retryable exception types
        retryable_exceptions = (
            RateLimitError,
            APIConnectionError,
            InternalServerError,
            ServiceUnavailableError,
            ContentPolicyViolationError,
        )

        return retry(
            retry=retry_if_exception_type(retryable_exceptions),
            wait=wait_exponential(
                multiplier=self.base_delay, min=self.base_delay, max=self.max_delay
            ),
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )

    def wrap_completion(
        self, completion_func, context: Optional[dict[str, Any]] = None
    ):
        """Wrap a completion function with error handling and retry logic.

        Parameters
        ----------
        completion_func : Callable
            The completion function to wrap.
        context : Optional[Dict[str, Any]], optional
            Context information for logging, by default None.

        Returns
        -------
        Callable
            The wrapped completion function with retry logic.
        """
        retry_decorator = self.create_retry_decorator(context)
        return retry_decorator(completion_func)

    def get_error_summary(self, error: Exception) -> dict[str, Any]:
        """Get a summary of error information.

        Parameters
        ----------
        error : Exception
            The error to summarize.

        Returns
        -------
        Dict[str, Any]
            Error summary information.
        """
        category = self.classify_error(error)

        return {
            "error_type": type(error).__name__,
            "error_category": category.value,
            "error_message": str(error),
            "retryable": category in self.retryable_errors,
            "provider_error": hasattr(error, "response") and error.response is not None,
        }

    def format_error_message(
        self, error: Exception, context: Optional[dict[str, Any]] = None
    ) -> str:
        """Format an error message for user display.

        Parameters
        ----------
        error : Exception
            The error to format.
        context : Optional[Dict[str, Any]], optional
            Additional context for the error, by default None.

        Returns
        -------
        str
            Formatted error message.
        """
        category = self.classify_error(error)
        context = context or {}

        base_msg = f"LLM operation failed: {error}"

        if category == ErrorCategory.NON_RETRYABLE_AUTH:
            return f"{base_msg}\nCheck your API key configuration."
        if category == ErrorCategory.NON_RETRYABLE_CONTEXT_LENGTH:
            return f"{base_msg}\nInput text is too long for the model."
        if category == ErrorCategory.RETRYABLE_RATE_LIMIT:
            return f"{base_msg}\nRate limit exceeded. Consider using a different model or reducing request frequency."
        if category == ErrorCategory.NON_RETRYABLE_NOT_FOUND:
            model = context.get("model", "unknown")
            return f"{base_msg}\nModel '{model}' not found. Check the model identifier."
        return base_msg
