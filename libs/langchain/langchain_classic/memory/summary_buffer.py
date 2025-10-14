from typing import Any

from langchain_core._api import deprecated
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils import pre_init
from typing_extensions import override

from langchain_classic.memory.chat_memory import BaseChatMemory
from langchain_classic.memory.summary import SummarizerMixin


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationSummaryBufferMemory(BaseChatMemory, SummarizerMixin):
    """Buffer with summarizer for storing conversation memory.

    Provides a running summary of the conversation together with the most recent
    messages in the conversation under the constraint that the total number of
    tokens in the conversation does not exceed a certain limit.
    """

    max_token_limit: int = 2000
    moving_summary_buffer: str = ""
    memory_key: str = "history"

    @property
    def buffer(self) -> str | list[BaseMessage]:
        """String buffer of memory."""
        return self.load_memory_variables({})[self.memory_key]

    async def abuffer(self) -> str | list[BaseMessage]:
        """Async memory buffer."""
        memory_variables = await self.aload_memory_variables({})
        return memory_variables[self.memory_key]

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return history buffer."""
        buffer = self.chat_memory.messages
        if self.moving_summary_buffer != "":
            first_messages: list[BaseMessage] = [
                self.summary_message_cls(content=self.moving_summary_buffer),
            ]
            buffer = first_messages + buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    @override
    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Asynchronously return key-value pairs given the text input to the chain."""
        buffer = await self.chat_memory.aget_messages()
        if self.moving_summary_buffer != "":
            first_messages: list[BaseMessage] = [
                self.summary_message_cls(content=self.moving_summary_buffer),
            ]
            buffer = first_messages + buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    @pre_init
    def validate_prompt_input_variables(cls, values: dict) -> dict:
        """Validate that prompt input variables are consistent."""
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            msg = (
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but it should have {expected_keys}."
            )
            raise ValueError(msg)
        return values

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        self.prune()

    async def asave_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        """Asynchronously save context from this conversation to buffer."""
        await super().asave_context(inputs, outputs)
        await self.aprune()

    def prune(self) -> None:
        """Prune buffer if it exceeds max token limit."""
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self.moving_summary_buffer = self.predict_new_summary(
                pruned_memory,
                self.moving_summary_buffer,
            )

    async def aprune(self) -> None:
        """Asynchronously prune buffer if it exceeds max token limit."""
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self.moving_summary_buffer = await self.apredict_new_summary(
                pruned_memory,
                self.moving_summary_buffer,
            )

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.moving_summary_buffer = ""

    async def aclear(self) -> None:
        """Asynchronously clear memory contents."""
        await super().aclear()
        self.moving_summary_buffer = ""
