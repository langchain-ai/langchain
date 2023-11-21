from langchain_core.schema import LangChainException


class InvalidKeyException(LangChainException):
    """Raised when a key is invalid; e.g., uses incorrect characters."""
