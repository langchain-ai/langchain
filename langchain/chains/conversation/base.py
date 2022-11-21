"""Chain that carries on a conversation and calls an LLM."""
from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.memory import MemoryChain
from langchain.docstore.in_memory import Docstore
from langchain.llms.base import LLM


class ConversationChain(MemoryChain, BaseModel):
    """Chain to have a conversation and load context from memory.

    Example:
        .. code-block:: python

            from langchain import ConversationChain, InMemoryDocstore, OpenAI
            conversation = ConversationChain(llm=OpenAI(), docstore=InMemoryDocstore())
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

    def _format_inputs_for_docstore(self, inputs: Dict[str, Any]) -> str:
        return "Human: " + inputs[self.input_key]

    def _format_output_for_docstore(self, output: str) -> str:
        return "AI: {output}".format(output=output)
