from typing import Any, Dict, List

from langchain.memory.chat_memory import BaseChatMemory, get_prompt_input_key
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import Field


class AutoGPTMemory(BaseChatMemory):
    retriever: VectorStoreRetriever = Field(exclude=True)
    """VectorStoreRetriever object to connect to."""

    @property
    def memory_variables(self) -> List[str]:
        return ["chat_history", "relevant_context"]

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        """Get the input key for the prompt."""
        if self.input_key is None:
            return get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = self.retriever.get_relevant_documents(query)
        return {
            "chat_history": self.chat_memory.messages[-10:],
            "relevant_context": docs,
        }
