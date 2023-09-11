"""Code for processing the working memory."""
import abc
from typing import List, Sequence, Callable

from langchain.schema.messages import get_buffer_string, SystemMessage

from langchain.automaton.typedefs import MessageLike
from langchain.automaton.prompt_generator import PromptGenerator


class WorkingMemoryManager(abc.ABC):
    @abc.abstractmethod
    def process(self, messages: Sequence[MessageLike]) -> List[MessageLike]:
        """Process the working memory returning a potentially new working memory."""


class TokenBufferProcessor(WorkingMemoryManager):
    """Trim the working memory to a maximum context length (aka token limit)."""

    def __init__(
        self,
        max_token_limit: int,
        token_counter: Callable[[str], int],
        prompt_generator: PromptGenerator,
        skip_system_messages: bool = True,
    ) -> None:
        """Token counter."""
        self.token_counter = token_counter
        self.max_token_limit = max_token_limit
        self.prompt_generator = prompt_generator
        self.skip_system_messages = skip_system_messages

    def process(self, messages: Sequence[MessageLike]) -> List[MessageLike]:
        """Update the working memory with the given messages."""
        # Work backwards from the end of the buffer dropping messages until
        messages = list(messages)
        curr_buffer_length = self.count_across_messages(messages)

        idx = 0

        while curr_buffer_length > self.max_token_limit:
            if idx >= len(messages):
                raise AssertionError("No messages left in buffer")

            if isinstance(messages[idx], SystemMessage):
                idx += 1
                continue

            messages.pop(idx)  # Drop the first message
            if not messages:
                raise AssertionError("No messages left in buffer")
            curr_buffer_length = self.count_across_messages(messages)
        return messages

    def count_across_messages(self, messages: Sequence[MessageLike]) -> int:
        """Count the number of tokens across messages."""
        if isinstance(self.prompt_generator, PromptGenerator):
            base_messages = self.prompt_generator.to_messages(messages)
        else:
            base_messages = self.prompt_generator(messages)
        buffer_string = get_buffer_string(base_messages)
        return self.token_counter(buffer_string)
