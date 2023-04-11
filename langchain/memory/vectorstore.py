"""Class for a VectorStore-backed memory object."""

from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from langchain.memory.chat_memory import BaseMemory
from langchain.memory.utils import get_prompt_input_key
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore


class VectorStoreMemory(BaseMemory):
    """Class for a VectorStore-backed memory object."""

    vectorstore: VectorStore = Field(exclude=True)
    """Vector Database to connect to."""

    k: int = 4
    """Number of results to attempt to return from store."""

    memory_key: str = "history"  #: :meta private:
    """Key name to locate the memories in the result of load_memory_variables."""

    input_key: Optional[str] = None
    """Key name to index the inputs to load_memory_variables."""

    search_type: str = "similarity"
    """Method to use for searching."""

    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Extra search args."""

    return_docs: bool = True
    """Whether or not to return the result of querying the database directly."""

    @property
    def memory_variables(self) -> List[str]:
        """The list of keys emitted from the load_memory_variables method."""
        return [self.memory_key]

    def _similarity_search(self, query: str, k: int) -> List[Document]:
        """Search the database for similar documents."""
        # TODO: Extract vectorstores with similarity_search_with_score classes
        # to a separate mixin and check to attempt to guarantee orderedness
        # of the output.
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(
                query, k=self.k, **self.search_kwargs
            )
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, k=self.k, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        """Get the input key for the prompt."""
        if self.input_key is None:
            return get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key

    def load_memory_variables(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, Union[List[Document], str]]:
        """Return history buffer."""
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = self._similarity_search(query, k=self.k)
        result: Union[List[Document], str]
        if not self.return_docs:
            result = "\n".join([doc.page_content for doc in docs])
        else:
            result = docs
        return {self.memory_key: result}

    def _format_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> str:
        """Format context from this conversation to buffer."""
        # Each document should only include the current turn, not the chat history
        filtered_inputs = {k: v for k, v in inputs.items() if k != self.memory_key}
        texts = [
            f"{k}: {v}"
            for k, v in list(filtered_inputs.items()) + list(outputs.items())
        ]
        return "\n".join(texts)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        formatted_text = self._format_context(inputs, outputs)
        self.vectorstore.add_texts([formatted_text])

    def clear(self) -> None:
        """Nothing to clear."""
