import warnings
from abc import ABC
from typing import Any

from langchain_core._api import deprecated
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from langchain_classic.base_memory import BaseMemory
from langchain_classic.memory.utils import get_prompt_input_key


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class BaseChatMemory(BaseMemory, ABC):
    """Abstract base class for chat memory.

    **ATTENTION** This abstraction was created prior to when chat models had
        native tool calling capabilities.
        It does **NOT** support native tool calling capabilities for chat models and
        will fail SILENTLY if used with a chat model that has native tool calling.

    DO NOT USE THIS ABSTRACTION FOR NEW CODE.
    """

    chat_memory: BaseChatMessageHistory = Field(
        default_factory=InMemoryChatMessageHistory,
    )
    output_key: str | None = None
    input_key: str | None = None
    return_messages: bool = False

    def _get_input_output(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) == 1:
                output_key = next(iter(outputs.keys()))
            elif "output" in outputs:
                output_key = "output"
                warnings.warn(
                    f"'{self.__class__.__name__}' got multiple output keys:"
                    f" {outputs.keys()}. The default 'output' key is being used."
                    f" If this is not desired, please manually set 'output_key'.",
                    stacklevel=3,
                )
            else:
                msg = (
                    f"Got multiple output keys: {outputs.keys()}, cannot "
                    f"determine which to store in memory. Please set the "
                    f"'output_key' explicitly."
                )
                raise ValueError(msg)
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_messages(
            [
                HumanMessage(content=input_str),
                AIMessage(content=output_str),
            ],
        )

    async def asave_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        await self.chat_memory.aadd_messages(
            [
                HumanMessage(content=input_str),
                AIMessage(content=output_str),
            ],
        )

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()

    async def aclear(self) -> None:
        """Clear memory contents."""
        await self.chat_memory.aclear()
