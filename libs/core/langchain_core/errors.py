from enum import Enum


class ErrorCode(Enum):
    INVALID_PROMPT_INPUT = "INVALID_PROMPT_INPUT"
    INVALID_TOOL_RESULTS = "INVALID_TOOL_RESULTS"
    MESSAGE_COERCION_FAILURE = "MESSAGE_COERCION_FAILURE"
    MODEL_AUTHENTICATION = "MODEL_AUTHENTICATION"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_RATE_LIMIT = "MODEL_RATE_LIMIT"
    OUTPUT_PARSING_FAILURE = "OUTPUT_PARSING_FAILURE"


class LangChainException(Exception):
    """Base exception class for LangChain errors."""

    def __init__(self, message: str, error_code: ErrorCode):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self):
        return (
            f'{self.__class__.__name__}("{self.message}")\n'
            "For troubleshooting, visit: https://python.langchain.com/docs/"
            f"troubleshooting/errors/{self.error_code.value}"
        )


class InvalidPromptInputError(LangChainException):
    """Exception raised for invalid prompt input errors."""

    def __init__(self, message: str):
        super().__init__(message, ErrorCode.INVALID_PROMPT_INPUT)


class InvalidToolResultsError(LangChainException):
    """Exception raised for invalid tool results."""

    def __init__(self, message: str):
        super().__init__(message, ErrorCode.INVALID_TOOL_RESULTS)


class MessageCoercionFailureError(LangChainException):
    """Exception raised for message coercion failures."""

    def __init__(self, message: str):
        super().__init__(message, ErrorCode.MESSAGE_COERCION_FAILURE)


class ModelAuthenticationError(LangChainException):
    """Exception raised for model authentication errors."""

    def __init__(self, message: str):
        super().__init__(message, ErrorCode.MODEL_AUTHENTICATION)


class ModelNotFoundError(LangChainException):
    """Exception raised when a model is not found."""

    def __init__(self, message: str):
        super().__init__(message, ErrorCode.MODEL_NOT_FOUND)


class ModelRateLimitError(LangChainException):
    """Exception raised for model rate limit errors."""

    def __init__(self, message: str):
        super().__init__(message, ErrorCode.MODEL_RATE_LIMIT)


class OutputParsingFailureError(LangChainException):
    """Exception raised for output parsing failures."""

    def __init__(self, message: str):
        super().__init__(message, ErrorCode.OUTPUT_PARSING_FAILURE)
