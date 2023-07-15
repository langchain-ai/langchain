from typing import Any, Dict, List

from pydantic import Field

from langchain.memory.chat_memory import BaseChatMemory
from langchain.vectorstores.base import VectorStoreRetriever


class AutoGPTMemory(BaseChatMemory):
    retriever: VectorStoreRetriever = Field(exclude=True)
    """VectorStoreRetriever object to connect to."""

    @property
    def memory_variables(self) -> List[str]:
        return ["chat_history", "relevant_context"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_key = self.get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = self.retriever.get_relevant_documents(query)
        return {
            "chat_history": self.chat_memory.messages[-10:],
            "relevant_context": docs,
        }
