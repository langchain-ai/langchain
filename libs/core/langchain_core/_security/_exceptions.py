"""SSRF protection exceptions."""


class SSRFBlockedError(Exception):
    """Raised when a request is blocked by SSRF protection policy."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"SSRF blocked: {reason}")
