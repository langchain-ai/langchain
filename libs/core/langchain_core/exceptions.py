"""Custom **exceptions** for LangChain."""

from enum import Enum
from typing import Any


class LangChainException(Exception):  # noqa: N818
    """General LangChain exception."""


class TracerException(LangChainException):
    """Base class for exceptions in tracers module."""


class ToolException(LangChainException):
    """Exception raised when a tool encounters an error during execution.

    This exception should be raised by tools when they fail to execute properly,
    allowing the calling code to handle tool failures gracefully and potentially
    retry or fall back to alternative approaches.
    """

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        *,
        original_error: Exception | None = None,
    ) -> None:
        """Create a ToolException.

        Args:
            message: Human-readable description of what went wrong.
            tool_name: Name of the tool that raised the exception.
            original_error: The underlying exception that caused this error.
        """
        full_message = f"Tool '{tool_name}' failed: {message}" if tool_name else message
        super().__init__(full_message)
        self.tool_name = tool_name
        self.original_error = original_error

    def __repr__(self) -> str:
        """Return a string representation of the exception."""
        return f"ToolException(tool_name={self.tool_name!r}, message={self.args[0]!r})"


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
            llm_output: String model output which is erroring.

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


class ErrorCode(Enum):
    """Error codes."""

    INVALID_PROMPT_INPUT = "INVALID_PROMPT_INPUT"
    INVALID_TOOL_RESULTS = "INVALID_TOOL_RESULTS"  # Used in JS; not Py (yet)
    MESSAGE_COERCION_FAILURE = "MESSAGE_COERCION_FAILURE"
    MODEL_AUTHENTICATION = "MODEL_AUTHENTICATION"  # Used in JS; not Py (yet)
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"  # Used in JS; not Py (yet)
    MODEL_RATE_LIMIT = "MODEL_RATE_LIMIT"  # Used in JS; not Py (yet)
    OUTPUT_PARSING_FAILURE = "OUTPUT_PARSING_FAILURE"
    TOOL_EXECUTION_FAILURE = "TOOL_EXECUTION_FAILURE"


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
        "For troubleshooting, visit: https://docs.langchain.com/oss/python/langchain"
        f"/errors/{error_code.value} "
    )
