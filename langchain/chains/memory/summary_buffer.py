from typing import Any, Dict, List

from pydantic import BaseModel, root_validator

from langchain.chains.llm import LLMChain
from langchain.chains.memory.prompt import SUMMARY_PROMPT
from langchain.llms.base import BaseLLM
from langchain.memory.chat_memory import ChatMemoryMixin
from langchain.memory.utils import get_buffer_string
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseMessage


class ConversationSummaryBufferMemory(ChatMemoryMixin, BaseModel):
    """Buffer with summarizer for storing conversation memory."""

    max_token_limit: int = 2000
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    moving_summary_buffer: str = ""
    llm: BaseLLM
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    memory_key: str = "history"

    @property
    def buffer(self) -> List[BaseMessage]:
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        buffer_string = get_buffer_string(
            self.buffer, human_prefix=self.human_prefix, ai_prefix=self.ai_prefix
        )

        if self.moving_summary_buffer == "":
            return {self.memory_key: buffer_string}
        memory_val = self.moving_summary_buffer + "\n" + buffer_string
        return {self.memory_key: memory_val}

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

    def get_num_tokens_list(self, arr: List[BaseMessage]) -> List[int]:
        """Get list of number of tokens in each string in the input array."""
        return [self.llm.get_num_tokens(get_buffer_string([x])) for x in arr]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        # Prune buffer if it exceeds max token limit
        buffer = self.chat_memory.messages
        curr_buffer_length = sum(self.get_num_tokens_list(buffer))
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = sum(self.get_num_tokens_list(buffer))
            chain = LLMChain(llm=self.llm, prompt=self.prompt)
            self.moving_summary_buffer = chain.predict(
                summary=self.moving_summary_buffer,
                new_lines=(
                    get_buffer_string(
                        pruned_memory,
                        human_prefix=self.human_prefix,
                        ai_prefix=self.ai_prefix,
                    )
                ),
            )

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.moving_summary_buffer = ""
