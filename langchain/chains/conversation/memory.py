"""Memory modules for conversation prompts."""
from typing import Any, Dict, List

from pydantic import BaseModel

from langchain.chains.base import Memory
from langchain.chains.conversation.prompt import SUMMARY_PROMPT
from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from langchain.prompts.base import BasePromptTemplate


class ConversationBufferMemory(Memory, BaseModel):
    """Buffer for storing conversation memory."""

    buffer: str = ""
    dynamic_key: str = "history"  #: :meta private:

    @property
    def dynamic_keys(self) -> List[str]:
        """Will always return list of dynamic keys.

        :meta private:
        """
        return [self.dynamic_key]

    def _load_dynamic_keys(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        return {self.dynamic_key: self.buffer}

    def _save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        prompt_input_keys = list(set(inputs.keys()) - set(self.dynamic_keys))
        assert len(prompt_input_keys) == 1, "One input key expected"
        assert len(outputs.keys()) == 1, "One output key expected"
        human = "Human: " + inputs[prompt_input_keys[0]]
        ai = "AI: " + outputs[list(outputs.keys())[0]]
        self.buffer += "\n" + "\n".join([human, ai])


class ConversationSummaryMemory(Memory, BaseModel):
    """Conversation summarizer to memory."""

    buffer: str = ""
    llm: LLM
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    dynamic_key: str = "history"  #: :meta private:

    @property
    def dynamic_keys(self) -> List[str]:
        """Will always return list of dynamic keys.

        :meta private:
        """
        return [self.dynamic_key]

    def _load_dynamic_keys(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        return {self.dynamic_key: self.buffer}

    def _save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        prompt_input_keys = list(set(inputs.keys()) - set(self.dynamic_keys))
        assert len(prompt_input_keys) == 1, "One input key expected"
        assert len(outputs.keys()) == 1, "One output key expected"
        assert self.prompt.input_variables == [
            "summary",
            "new_lines",
        ], "Expected different input keys"
        human = "Human: " + inputs[prompt_input_keys[0]]
        ai = "AI: " + outputs[list(outputs.keys())[0]]
        new_lines = "\n".join([human, ai])
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.buffer = chain.predict(summary=self.buffer, new_lines=new_lines)
