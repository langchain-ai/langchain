"""Prompt generation for the automaton."""
from __future__ import annotations

import abc
from typing import Mapping, Any, Callable, List, Sequence, Optional, Union

from langchain.automaton.typedefs import MessageLike
from langchain.schema import BaseMessage, PromptValue


class BoundPromptValue(PromptValue):
    """A prompt value that is bound to a specific value."""

    as_string: Callable[[], str]
    as_messages: Callable[[], List[BaseMessage]]

    def to_string(self) -> str:
        """Return prompt value as string."""
        return self.as_string()

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of Messages."""
        return self.as_messages()


class PromptGenerator(abc.ABC):
    @abc.abstractmethod
    def to_messages(
        self, original_messages: Sequence[MessageLike]
    ) -> List[BaseMessage]:
        """Generate a prompt from message like objects."""

    @abc.abstractmethod
    def to_string(self, original_messages: Sequence[MessageLike]) -> str:
        """Generate a prompt from message like objects."""

    def to_prompt_value(self, original_messages: Sequence[MessageLike]) -> PromptValue:
        """Generate a prompt from message like objects."""
        return BoundPromptValue(
            as_string=lambda: self.to_string(original_messages),
            as_messages=lambda: self.to_messages(original_messages),
        )


class AdapterBasedGenerator(PromptGenerator):
    def __init__(
        self,
        *,
        msg_adapters: Optional[
            Mapping[Any, Callable[[MessageLike], Union[BaseMessage, List[BaseMessage]]]]
        ] = None,
        str_adapters: Optional[Mapping[Any, Callable[[MessageLike], str]]] = None,
    ) -> None:
        """Initialize the adapter based generator."""
        self.msg_adapters = msg_adapters or {}
        self.str_adapters = str_adapters or {}

    def to_messages(self, messages: Sequence[MessageLike]) -> List[BaseMessage]:
        """Generate a prompt from message like objects."""
        new_messages = []

        for original_message in messages:
            adapter = self.msg_adapters.get(type(original_message), None)
            if adapter:
                translated = adapter(original_message)
                if isinstance(translated, BaseMessage):
                    new_messages.append(translated)
                else:
                    new_messages.extend(translated)
                continue

            if isinstance(original_message, BaseMessage):
                # Only adds BaseMessages by default,
                # internal messages are ignored
                new_messages.append(original_message)

        return new_messages

    def to_string(self, messages: Sequence[MessageLike]) -> str:
        """Generate a prompt from message like objects."""
        string_prompts = []
        for original_message in messages:
            adapter = self.str_adapters.get(type(original_message), None)
            if adapter:
                string_prompts.extend(adapter(original_message))
                continue
            else:
                raise RuntimeError(
                    f"String adapter not found for {type(original_message)}"
                )

        return "\n".join(string_prompts)
