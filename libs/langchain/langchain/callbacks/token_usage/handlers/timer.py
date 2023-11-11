"""This module contains a helper class to measure LLM token response timing."""

import time
from typing import Optional


class TokenUsageTimer:
    """Helper class to measure LLM token response timing."""

    start_timestamp: Optional[float] = None
    first_token_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None

    def start(self) -> None:
        """Call this method when the LLM starts processing your request."""
        self.start_timestamp = time.perf_counter()

    def new_token(self) -> None:
        """Call this method when the LLM sends you a token output."""
        if self.first_token_timestamp is None:
            self.first_token_timestamp = time.perf_counter()

    def end(self) -> None:
        """Call this method when the LLM finishes processing your request."""
        self.end_timestamp = time.perf_counter()

    @property
    def first_token_elapsed(self) -> Optional[float]:
        """Returns the time elapsed until the first token was generated."""
        if self.first_token_timestamp is None or self.start_timestamp is None:
            return None
        return self.first_token_timestamp - self.start_timestamp

    @property
    def completion_elapsed(self) -> Optional[float]:
        """Returns the total time elapsed to elaborate the request."""
        if self.end_timestamp is None or self.start_timestamp is None:
            return None
        return self.end_timestamp - self.start_timestamp
