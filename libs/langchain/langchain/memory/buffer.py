from typing import Any, Optional

from langchain_core._api import deprecated
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils import pre_init

from langchain.memory.chat_memory import BaseChatMemory, BaseMemory
from langchain.memory.utils import get_prompt_input_key


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationBufferMemory(BaseChatMemory):
    """A basic memory implementation that simply stores the conversation history.

    This stores the entire conversation history in memory without any
    additional processing.

    Note that additional processing may be required in some situations when the
    conversation history is too large to fit in the context window of the model.
    """

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

    def _buffer_as_str(self, messages: list[BaseMessage]) -> str:
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
    def buffer_as_messages(self) -> list[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        return self.chat_memory.messages

    async def abuffer_as_messages(self) -> list[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        return await self.chat_memory.aget_messages()

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return key-value pairs given the text input to the chain."""
        buffer = await self.abuffer()
        return {self.memory_key: buffer}


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationStringBufferMemory(BaseMemory):
    """A basic memory implementation that simply stores the conversation history.

    This stores the entire conversation history in memory without any
    additional processing.

    Equivalent to ConversationBufferMemory but tailored more specifically
    for string-based conversations rather than chat models.

    Note that additional processing may be required in some situations when the
    conversation history is too large to fit in the context window of the model.
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """Prefix to use for AI generated responses."""
    buffer: str = ""
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    memory_key: str = "history"  #: :meta private:

    @pre_init
    def validate_chains(cls, values: dict) -> dict:
        """Validate that return messages is not True."""
        if values.get("return_messages", False):
            raise ValueError(
                "return_messages must be False for ConversationStringBufferMemory"
            )
        return values

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables.
        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Return history buffer."""
        return self.load_memory_variables(inputs)

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
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
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        self.buffer += "\n" + "\n".join([human, ai])

    async def asave_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        """Save context from this conversation to buffer."""
        return self.save_context(inputs, outputs)

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = ""

    async def aclear(self) -> None:
        self.clear()
