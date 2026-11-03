"""Custom **exceptions** for LangChain."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage, RefusalContentBlock


class LangChainException(Exception):  # noqa: N818
    """General LangChain exception."""


class TracerException(LangChainException):
    """Base class for exceptions in tracers module."""


class OutputParserException(ValueError, LangChainException):  # noqa: N818
    """Exception that output parsers should raise to signify a parsing error.

    This exists to differentiate parsing errors from other code or execution errors
    that also may arise inside the output parser.

    `OutputParserException` will be available to catch and handle in ways to fix the
    parsing error, while other errors will be raised.
    """

    def __init__(
        self,
        error: Any,
        observation: str | None = None,
        llm_output: str | None = None,
        send_to_llm: bool = False,  # noqa: FBT001,FBT002
    ):
        """Create an `OutputParserException`.

        Args:
            error: The error that's being re-raised or an error message.
            observation: String explanation of error which can be passed to a model to
                try and remediate the issue.
            llm_output: String model output which is error-ing.

            send_to_llm: Whether to send the observation and llm_output back to an Agent
                after an `OutputParserException` has been raised.

                This gives the underlying model driving the agent the context that the
                previous output was improperly structured, in the hopes that it will
                update the output to the correct format.

        Raises:
            ValueError: If `send_to_llm` is `True` but either observation or
                `llm_output` are not provided.
        """
        if isinstance(error, str):
            error = create_message(
                message=error, error_code=ErrorCode.OUTPUT_PARSING_FAILURE
            )

        super().__init__(error)
        if send_to_llm and (observation is None or llm_output is None):
            msg = (
                "Arguments 'observation' & 'llm_output'"
                " are required if 'send_to_llm' is True"
            )
            raise ValueError(msg)
        self.observation = observation
        self.llm_output = llm_output
        self.send_to_llm = send_to_llm


class ModelError(LangChainException):
    """Base exception for failures related to model invocation.

    Subclasses correspond to conditions that model providers report consistently,
    so the same condition maps to the same exception type regardless of provider.
    Most are keyed to the HTTP status they surface it with; `ModelRefusalError`
    covers successful responses in which the model refused.

    Provider integrations raise HTTP error subclasses that also inherit from the
    provider SDK's own exception type, so code catching either continues to work.
    """

    is_retryable = False
    """Whether retrying the same model request may succeed."""


class ModelAuthenticationError(ModelError):
    """Exception raised when model provider authentication fails (HTTP 401)."""


class ModelPermissionDeniedError(ModelError):
    """Exception raised when credentials lack permission for a request (HTTP 403)."""


class ModelInvalidRequestError(ModelError):
    """Exception raised when a provider rejects a request as invalid (e.g. HTTP 400)."""


class ModelNotFoundError(ModelError):
    """Exception raised when the requested model cannot be found (HTTP 404)."""


class ModelRateLimitError(ModelError):
    """Exception raised when a model provider rate limit is exceeded (HTTP 429)."""

    is_retryable = True


class ModelAPIError(ModelError):
    """Exception raised when a model provider reports a server failure (HTTP 5xx)."""

    is_retryable = True


class ModelConnectionError(ModelError):
    """Exception raised when a model provider cannot be reached."""

    is_retryable = True


class ModelTimeoutError(ModelError):
    """Exception raised when a model request times out."""

    is_retryable = True


class ContextOverflowError(ModelError):
    """Exception raised when input exceeds the model's context limit.

    This exception is raised by chat models when the input tokens exceed
    the maximum context window supported by the model.
    """


class ModelRefusalError(ModelError):
    """Exception raised when a model declines to produce the requested output.

    Providers report a refusal as a successful response rather than an HTTP error
    (for example Anthropic's `stop_reason="refusal"`, or the `refusal` field in
    OpenAI Structured Outputs), so a plain model call returns the message as-is.
    Integrations raise this exception from paths that need the content the model
    withheld, such as `with_structured_output`, so callers can handle refusals the
    same way across providers instead of receiving a parsing error.

    Retrying the same request is unlikely to succeed.

    !!! version-added "Added in `langchain-core` 1.7.5"
    """

    def __init__(
        self,
        message: str = "Model refused to produce the requested output",
        *,
        raw_message: BaseMessage | None = None,
    ) -> None:
        """Create a `ModelRefusalError`.

        Args:
            message: Error message.
            raw_message: The model response that contained the refusal. Its
                `content_blocks` hold the refusal block, including any
                provider-specific details in `extras`.
        """
        self.raw_message = raw_message
        super().__init__(message)

    @classmethod
    def from_refusal(
        cls, refusal: RefusalContentBlock, *, raw_message: BaseMessage
    ) -> ModelRefusalError:
        """Create a `ModelRefusalError` from a standard refusal block.

        Args:
            refusal: The refusal block from the response's `content_blocks`.
            raw_message: The model response that contained the refusal.

        Returns:
            An error whose message includes the refusal's category and text.
        """
        message = "Model refused to produce the requested output"
        if category := refusal.get("extras", {}).get("category"):
            message += f" (category={category!r})"
        if text := refusal.get("refusal"):
            message += f": {text}"
        return cls(message, raw_message=raw_message)


class ErrorCode(Enum):
    """Error codes."""

    INVALID_PROMPT_INPUT = "INVALID_PROMPT_INPUT"
    INVALID_TOOL_RESULTS = "INVALID_TOOL_RESULTS"  # Used in JS; not Py (yet)
    MESSAGE_COERCION_FAILURE = "MESSAGE_COERCION_FAILURE"
    MODEL_AUTHENTICATION = "MODEL_AUTHENTICATION"  # Used in JS; not Py (yet)
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"  # Used in JS; not Py (yet)
    MODEL_RATE_LIMIT = "MODEL_RATE_LIMIT"  # Used in JS; not Py (yet)
    OUTPUT_PARSING_FAILURE = "OUTPUT_PARSING_FAILURE"


def create_message(*, message: str, error_code: ErrorCode) -> str:
    """Create a message with a link to the LangChain troubleshooting guide.

    Args:
        message: The message to display.
        error_code: The error code to display.

    Returns:
        The full message with the troubleshooting link.

    Example:
        ```python
        create_message(
            message="Failed to parse output",
            error_code=ErrorCode.OUTPUT_PARSING_FAILURE,
        )
        "Failed to parse output. For troubleshooting, visit: ..."
        ```
    """
    return (
        f"{message}\n"
        "For troubleshooting, visit: https://docs.langchain.com/oss/python/langchain"
        f"/errors/{error_code.value} "
    )
