"""Class for a VectorStore-backed memory object."""

import json
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStoreRetriever

from langchain.load.serializable import Serializable
from langchain.memory.chat_memory import BaseMemory
from langchain.memory.utils import get_prompt_input_key


class VectorStoreRetrieverMemory(BaseMemory):
    """VectorStoreRetriever-backed memory."""

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

    def load_memory_variables(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, Union[List[Document], str]]:
        """Return history buffer."""
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = self.retriever.get_relevant_documents(query)
        result: Union[List[Document], str]
        if not self.return_docs:
            result = "\n".join([doc.page_content for doc in docs])
        else:
            result = docs
        return {self.memory_key: result}

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

    def clear(self) -> None:
        """Nothing to clear."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Is this class serializable?"""
        return True

    def to_json(self) -> Dict[str, Any]:
        serialized = super().to_json()

        self_dict = {key: value for key, value in vars(self).items()}

        serialized["obj"] = json.loads(
            json.dumps(
                self_dict,
                default=lambda o: o.to_json()
                if isinstance(o, Serializable)
                else o.__dict__,
                sort_keys=True,
                indent=4,
            )
        )
        return serialized
