"""Custom **exceptions** for LangChain."""

from enum import Enum
from typing import Any


class LangChainException(Exception):  # noqa: N818
    """General LangChain exception."""


class TracerException(LangChainException):
    """Base class for exceptions in tracers module."""


class OutputParserException(ValueError, LangChainException):  # noqa: N818
    """Exception that output parsers should raise to signify a parsing error.

    This exists to differentiate parsing errors from other code or execution errors
    that also may arise inside the output parser. `OutputParserException` will be
    available to catch and handle in ways to fix the parsing error, while other
    errors will be raised.
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
            observation: String explanation of error which can be passed to a
                model to try and remediate the issue.
            llm_output: String model output which is error-ing.

            send_to_llm: Whether to send the observation and llm_output back to an Agent
                after an `OutputParserException` has been raised.
                This gives the underlying model driving the agent the context that the
                previous output was improperly structured, in the hopes that it will
                update the output to the correct format.

        Raises:
            ValueError: If `send_to_llm` is True but either observation or
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


class ErrorCode(Enum):
    """Error codes."""

    INVALID_PROMPT_INPUT = "INVALID_PROMPT_INPUT"
    INVALID_TOOL_RESULTS = "INVALID_TOOL_RESULTS"
    MESSAGE_COERCION_FAILURE = "MESSAGE_COERCION_FAILURE"
    MODEL_AUTHENTICATION = "MODEL_AUTHENTICATION"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_RATE_LIMIT = "MODEL_RATE_LIMIT"
    OUTPUT_PARSING_FAILURE = "OUTPUT_PARSING_FAILURE"


def create_message(*, message: str, error_code: ErrorCode) -> str:
    """Create a message with a link to the LangChain troubleshooting guide.

    Args:
        message: The message to display.
        error_code: The error code to display.

    Returns:
        The full message with the troubleshooting link.
    """
    return (
        f"{message}\n"
        "For troubleshooting, visit: https://python.langchain.com/docs/"
        f"troubleshooting/errors/{error_code.value} "
    )
