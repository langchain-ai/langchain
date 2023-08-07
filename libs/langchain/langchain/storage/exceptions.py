class LangConnectException(Exception):
    """General langconnect exception."""


class InvalidKeyException(LangConnectException):
    """Raised when a key is invalid; e.g., uses incorrect characters."""
