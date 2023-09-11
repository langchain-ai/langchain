"""Code for processing the working memory."""
import abc
from typing import List, Sequence, Callable

from langchain.schema.messages import get_buffer_string

from langchain.automaton.typedefs import MessageLike
from langchain.automaton.prompt_generator import PromptGenerator


class WorkingMemoryManager(abc.ABC):
    @abc.abstractmethod
    def process(self, messages: Sequence[MessageLike]) -> List[MessageLike]:
        """Process the working memory returning a potentially new working memory."""


def trim_messages(
    messages: Sequence[MessageLike],
    max_token_limit: int,
    token_counter: Callable[[str], int],
) -> List[MessageLike]:
    """Trim the working memory to a maximum context length (aka token limit)."""
    # Work backwards from the end of the buffer dropping messages until
    messages = list(messages)
    curr_buffer_length = sum(token_counter(message) for message in messages)
    while curr_buffer_length > max_token_limit:
        messages.pop(0)  # Drop the first message
        if not messages:
            raise AssertionError("No messages left in buffer")
        curr_buffer_length = sum(token_counter(message) for message in messages)
    return messages


class TokenBufferProcessor(WorkingMemoryManager):
    """Trim the working memory to a maximum context length (aka token limit)."""

    def __init__(
        self,
        max_token_limit: int,
        token_counter: Callable[[str], int],
        prompt_generator: PromptGenerator,
    ) -> None:
        """Token counter."""
        self.token_counter = token_counter
        self.max_token_limit = max_token_limit
        self.prompt_generator = prompt_generator

    def process(self, messages: Sequence[MessageLike]) -> List[MessageLike]:
        """Update the working memory with the given messages."""
        # Work backwards from the end of the buffer dropping messages until
        messages = list(messages)
        curr_buffer_length = self.count_across_messages(messages)
        while curr_buffer_length > self.max_token_limit:
            messages.pop(0)  # Drop the first message
            if not messages:
                raise AssertionError("No messages left in buffer")
            curr_buffer_length = self.count_across_messages(messages)
        return messages

    def count_across_messages(self, messages: Sequence[MessageLike]) -> int:
        """Count the number of tokens across messages."""
        base_messages = self.prompt_generator.to_messages(messages)
        buffer_string = get_buffer_string(base_messages)
        return self.token_counter(buffer_string)
