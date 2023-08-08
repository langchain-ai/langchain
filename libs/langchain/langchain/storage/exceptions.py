class LangChainException(Exception):
    """General LangChain exception."""


class InvalidKeyException(LangChainException):
    """Raised when a key is invalid; e.g., uses incorrect characters."""
