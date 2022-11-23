"""Chain that carries on a conversation and calls an LLM."""
from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Memory
from langchain.chains.conversation.prompt import PROMPT
from langchain.chains.llm import LLMChain
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
        return {self.dynamic_keys[0]: self.buffer}

    def _save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        self.buffer += "\nHuman: " + inputs[list(inputs.keys())[0]]
        self.buffer += "\nAI: " + outputs[list(outputs.keys())[0]]


class ConversationChain(LLMChain, BaseModel):
    """Chain to have a conversation and load context from memory.

    Example:
        .. code-block:: python

            from langchain import ConversationChain, OpenAI
            conversation = ConversationChain(llm=OpenAI())
    """

    memory: Memory = ConversationBufferMemory()
    """Default memory store."""
    prompt: BasePromptTemplate = PROMPT
    """Default conversation prompt to use."""

    input_key: str = "input"  #: :meta private:
    output_key: str = "response"  #: :meta private:
    buffer: str = ""  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
