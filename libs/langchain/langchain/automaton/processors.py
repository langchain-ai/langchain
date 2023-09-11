"""Code for processing the working memory."""
from typing import List, Sequence, Callable, Protocol

from langchain.automaton.typedefs import MessageLike


class WorkingMemoryProcessor(Protocol):
    def update(self, messages: Sequence[MessageLike]) -> List[MessageLike]:
        """Process the working memory returning a potentially new working memory."""
        ...


class TokenBufferProcessor(WorkingMemoryProcessor):
    """Trim the working memory to a maximum context length (aka token limit)."""

    def __init__(
        self,
        max_token_limit: int,
        token_counter: Callable[[str], int],
        prompt_generator: Callable[[Sequence[MessageLike]], str],
    ) -> None:
        """Token counter."""
        self.token_counter = token_counter
        self.max_token_limit = max_token_limit

    def update(self, messages: Sequence[MessageLike]) -> List[MessageLike]:
        """Update the working memory with the given messages."""
        # Work backwards from the end of the buffer dropping messages until
        messages = list(messages)
        curr_buffer_length = self.token_counter(messages)
        while curr_buffer_length > self.max_token_limit:
            messages.pop(0)  # Drop the first message
            if not messages:
                raise AssertionError("No messages left in buffer")
            curr_buffer_length = self.token_counter(messages)
        return messages

    def count_across_messages(
        self, messages: Sequence[MessageLike]
    ) -> List[MessageLike]:
        """Count the number of tokens across messages."""
        return sum(self.token_counter(message) for message in messages)
