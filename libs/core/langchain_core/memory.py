"""**Memory** maintains Chain state, incorporating context from past runs.

**Class hierarchy for Memory:**

.. code-block::

    BaseMemory --> <name>Memory --> <name>Memory  # Examples: BaseChatMemory -> MotorheadMemory

"""  # noqa: E501
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    get_buffer_string,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import run_in_executor


class BaseMemory(Serializable, ABC):
    """Abstract base class for memory in Chains.

    Memory refers to state in Chains. Memory can be used to store information about
        past executions of a Chain and inject that information into the inputs of
        future executions of the Chain. For example, for conversational Chains Memory
        can be used to store conversations and automatically add them to future model
        prompts so that the model has the necessary context to respond coherently to
        the latest input.

    Example:
        .. code-block:: python

            class SimpleMemory(BaseMemory):
                memories: Dict[str, Any] = dict()

                @property
                def memory_variables(self) -> List[str]:
                    return list(self.memories.keys())

                def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
                    return self.memories

                def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
                    pass

                def clear(self) -> None:
                    pass
    """  # noqa: E501

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """The string keys this memory class will add to chain inputs."""

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain."""

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain."""
        return await run_in_executor(None, self.load_memory_variables, inputs)

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this chain run to memory."""

    async def asave_context(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> None:
        """Save the context of this chain run to memory."""
        await run_in_executor(None, self.save_context, inputs, outputs)

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""

    async def aclear(self) -> None:
        """Clear memory contents."""
        await run_in_executor(None, self.clear)


class ChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history.

    Stores messages in an in memory list.
    """

    messages: List[BaseMessage] = Field(default_factory=list)

    async def aget_messages(self) -> List[BaseMessage]:
        return self.messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the store"""
        self.add_messages(messages)

    def clear(self) -> None:
        self.messages = []

    async def aclear(self) -> None:
        self.clear()


class BaseChatMemory(BaseMemory, ABC):
    """Abstract base class for chat memory."""

    chat_memory: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    return_messages: bool = False

    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_messages(
            [HumanMessage(content=input_str), AIMessage(content=output_str)]
        )

    async def asave_context(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        await self.chat_memory.aadd_messages(
            [HumanMessage(content=input_str), AIMessage(content=output_str)]
        )

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()

    async def aclear(self) -> None:
        """Clear memory contents."""
        await self.chat_memory.aclear()


class ConversationBufferMemory(BaseChatMemory):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:

    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    async def abuffer(self) -> Any:
        """String buffer of memory."""
        return (
            await self.abuffer_as_messages()
            if self.return_messages
            else await self.abuffer_as_str()
        )

    def _buffer_as_str(self, messages: List[BaseMessage]) -> str:
        return get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is True."""
        return self._buffer_as_str(self.chat_memory.messages)

    async def abuffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is True."""
        messages = await self.chat_memory.aget_messages()
        return self._buffer_as_str(messages)

    @property
    def buffer_as_messages(self) -> List[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        return self.chat_memory.messages

    async def abuffer_as_messages(self) -> List[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        return await self.chat_memory.aget_messages()

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain."""
        buffer = await self.abuffer()
        return {self.memory_key: buffer}


def get_prompt_input_key(inputs: Dict[str, Any], memory_variables: List[str]) -> str:
    """
    Get the prompt input key.
    Args:
        inputs: Dict[str, Any]
        memory_variables: List[str]
    Returns:
        A prompt input key.
    """
    # "stop" is a special key that can be passed as input but is not used to
    # format the prompt.
    prompt_input_keys = list(set(inputs).difference(memory_variables + ["stop"]))
    if len(prompt_input_keys) != 1:
        raise ValueError(f"One input key expected got {prompt_input_keys}")
    return prompt_input_keys[0]
