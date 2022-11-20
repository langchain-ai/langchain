"""Chain that just formats a prompt and calls an LLM."""
from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.memory import MemoryChain
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.input import print_text
from langchain.llms.base import LLM
from langchain.prompts.base import BasePromptTemplate


class ConversationChain(MemoryChain, BaseModel):
    """Chain to have a conversation and load context from memory.

    Example:
        .. code-block:: python

            from langchain import ConversationChain, OpenAI
            react = ConversationChain(llm=OpenAI())
    """

    llm: LLM
    """LLM wrapper to use."""
    docstore: Docstore
    """Docstore to use."""
    input_key: str = "input"  #: :meta private:
    output_key: str = "response"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _fetch_history_from_docstore(self, inputs):
        return '\n'.join(self.docstore.all())

    def _format_inputs_for_docstore(self, inputs):
        return "Human: " + inputs[self.input_key]
