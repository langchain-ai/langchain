from typing import Any, Dict, List, Union

from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.pydantic_v1 import root_validator

from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.summary import SummarizerMixin


class ConversationSummaryBufferMemory(BaseChatMemory, SummarizerMixin):
    """Buffer with summarizer for storing conversation memory."""

    max_token_limit: int = 2000
    moving_summary_buffer: str = ""
    memory_key: str = "history"

    @property
    def buffer(self) -> Union[str, List[BaseMessage]]:
        """String buffer of memory."""
        return self.load_memory_variables({})[self.memory_key]

    async def abuffer(self) -> Any:
        """String buffer of memory asynchronously."""
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
        
    async def abuffer_as_str(self) -> str:
        """Asynchronously exposes the buffer as a string in case return_messages is True."""
        messages = await self.chat_memory.aget_messages()
        return self._buffer_as_str(messages)        

    @property
    def buffer_as_messages(self) -> List[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        return self.chat_memory.messages

    async def abuffer_as_messages(self) -> List[BaseMessage]:
        """Asynchronously exposes the buffer as a list of messages in case return_messages is False."""
        return await self.chat_memory.aget_messages()

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        buffer = self.chat_memory.messages
        if self.moving_summary_buffer != "":
            first_messages: List[BaseMessage] = [
                self.summary_message_cls(content=self.moving_summary_buffer)
            ]
            buffer = first_messages + buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer, human_prefix=self.human_prefix, ai_prefix=self.ai_prefix
            )
        return {self.memory_key: final_buffer}

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously return key-value pairs given the text input to the chain."""
        buffer = await self.abuffer()
        if self.moving_summary_buffer != "":
            first_messages: List[BaseMessage] = [
                self.summary_message_cls(content=self.moving_summary_buffer)
            ]
            buffer = first_messages + buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer, human_prefix=self.human_prefix, ai_prefix=self.ai_prefix
            )
        return {self.memory_key: final_buffer}

    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but it should have {expected_keys}."
            )
        return values

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        self.prune()

    async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Asynchronously save context from this conversation to buffer."""
        await super().asave_context(inputs, outputs)
        await self.aprune()

    def prune(self) -> None:
        """Prune buffer if it exceeds max token limit"""
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self.moving_summary_buffer = self.predict_new_summary(
                pruned_memory, self.moving_summary_buffer
            )


    async def aprune(self) -> None:
        """Asynchronously prune buffer if it exceeds max token limit"""
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self.moving_summary_buffer = await self.apredict_new_summary(
                pruned_memory, self.moving_summary_buffer
            )

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.moving_summary_buffer = ""

    async def aclear(self) -> None:
        """Asynchronously clear memory contents."""
        await super().aclear()
        self.moving_summary_buffer = ""