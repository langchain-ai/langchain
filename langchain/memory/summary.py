from typing import Any, Dict, List

from pydantic import BaseModel, root_validator

from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.memory.chat_memory import ChatMemoryMixin
from langchain.memory.prompt import SUMMARY_PROMPT
from langchain.memory.utils import get_buffer_string
from langchain.prompts.base import BasePromptTemplate


class ConversationSummaryMemory(ChatMemoryMixin, BaseModel):
    """Conversation summarizer to memory."""

    buffer: str = ""
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLLM
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    memory_key: str = "history"  #: :meta private:

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

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
        new_lines = get_buffer_string(
            self.chat_memory.messages[-2:],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.buffer = chain.predict(summary=self.buffer, new_lines=new_lines)

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.buffer = ""
