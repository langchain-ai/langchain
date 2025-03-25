"""Class for a VectorStore-backed memory object."""

from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import Field

from langchain.memory.chat_memory import BaseMemory
from langchain.memory.utils import get_prompt_input_key


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class VectorStoreRetrieverMemory(BaseMemory):
    """Store the conversation history in a vector store and retrieves the relevant
    parts of past conversation based on the input.
    """

    retriever: VectorStoreRetriever = Field(exclude=True)
    """VectorStoreRetriever object to connect to."""

    memory_key: str = "history"  #: :meta private:
    """Key name to locate the memories in the result of load_memory_variables."""

    input_key: Optional[str] = None
    """Key name to index the inputs to load_memory_variables."""

    return_docs: bool = False
    """Whether or not to return the result of querying the database directly."""

    exclude_input_keys: Sequence[str] = Field(default_factory=tuple)
    """Input keys to exclude in addition to memory key when constructing the document"""

    @property
    def memory_variables(self) -> List[str]:
        """The list of keys emitted from the load_memory_variables method."""
        return [self.memory_key]

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        """Get the input key for the prompt."""
        if self.input_key is None:
            return get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key

    def _documents_to_memory_variables(
        self, docs: List[Document]
    ) -> Dict[str, Union[List[Document], str]]:
        result: Union[List[Document], str]
        if not self.return_docs:
            result = "\n".join([doc.page_content for doc in docs])
        else:
            result = docs
        return {self.memory_key: result}

    def load_memory_variables(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, Union[List[Document], str]]:
        """Return history buffer."""
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = self.retriever.invoke(query)
        return self._documents_to_memory_variables(docs)

    async def aload_memory_variables(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, Union[List[Document], str]]:
        """Return history buffer."""
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = await self.retriever.ainvoke(query)
        return self._documents_to_memory_variables(docs)

    def _form_documents(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        """Format context from this conversation to buffer."""
        # Each document should only include the current turn, not the chat history
        exclude = set(self.exclude_input_keys)
        exclude.add(self.memory_key)
        filtered_inputs = {k: v for k, v in inputs.items() if k not in exclude}
        texts = [
            f"{k}: {v}"
            for k, v in list(filtered_inputs.items()) + list(outputs.items())
        ]
        page_content = "\n".join(texts)
        return [Document(page_content=page_content)]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        documents = self._form_documents(inputs, outputs)
        self.retriever.add_documents(documents)

    async def asave_context(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> None:
        """Save context from this conversation to buffer."""
        documents = self._form_documents(inputs, outputs)
        await self.retriever.aadd_documents(documents)

    def clear(self) -> None:
        """Nothing to clear."""

    async def aclear(self) -> None:
        """Nothing to clear."""
