"""Prompt generation for the automaton."""
from __future__ import annotations

import abc
from typing import Mapping, Any, Callable, List, Sequence, Optional

from langchain.automaton.typedefs import MessageLike
from langchain.schema import BaseMessage, PromptValue


class BoundPromptValue(PromptValue):
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
        adapters: Optional[
            Mapping[Any, Callable[[MessageLike], List[BaseMessage]]]
        ] = None,
        str_adapters: Optional[Mapping[Any, Callable[[MessageLike], str]]] = None,
        pass_through_base_messages: bool = True,
    ) -> None:
        """Initialize the adapter based generator."""
        self.adapters = adapters
        self.str_adapters = str_adapters
        self.pass_through_base_messages = pass_through_base_messages

    def to_messages(self, messages: Sequence[MessageLike]) -> List[BaseMessage]:
        """Generate a prompt from message like objects."""
        new_messages = []

        for original_message in messages:
            adapter = self.adapters.get(type(original_message), None)
            if adapter:
                new_messages.extend(adapter(original_message))
                continue

            if self.pass_through_base_messages:
                new_messages.append(original_message)
            else:
                raise RuntimeError(f"Adapter not found for {type(original_message)}")
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


#
# def create_standard_generator(overrided_adapters: Mapping[]) -> AdapterBasedGenerator:
#     """Create the standard prompt generator."""
#     return AdapterBasedGenerator(
#         adapters={
#             FunctionResult: (lambda result: [HumanMessage(content=f"Observation: result.content)])
#         },
#         str_adapters={
#             BaseMessage: lambda message: message.to_string(),
#         },
#         pass_through_base_messages=True,
#     )
