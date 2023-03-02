"""Chain for question-answering against a vector database."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.chat.base import BaseChatChain
from langchain.chat.question_answering import QAChain
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatMessage
from langchain.vectorstores.base import VectorStore


class VectorDBQA(BaseChatChain, BaseModel):
    """Chain for question-answering against a vector database.

    Example:
        .. code-block:: python

            from langchain import OpenAI, VectorDBQA
            from langchain.faiss import FAISS
            vectordb = FAISS(...)
            vectordbQA = VectorDBQA(llm=OpenAI(), vectorstore=vectordb)

    """

    vectorstore: VectorStore = Field(exclude=True)
    """Vector Database to connect to."""
    k: int = 4
    """Number of documents to query for."""
    qa_chain: QAChain
    """Chain to use to combine the documents."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_source_documents: bool = False
    """Return the source documents."""
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Extra search args."""
    search_type: str = "similarity"
    """Search type to use over vectorstore. `similarity` or `mmr`."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"]
        return _output_keys

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in ("similarity", "mmr"):
                raise ValueError(f"search_type of {search_type} not allowed.")
        return values

    @classmethod
    def from_model(
        cls,
        model: BaseChatModel,
        starter_messages: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> VectorDBQA:
        """Initialize from LLM."""
        qa_chain = QAChain.from_model(model, starter_messages=starter_messages)

        return cls(qa_chain=qa_chain, **kwargs)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Run similarity search and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = vectordbqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        question = inputs[self.input_key]

        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(
                question, k=self.k, **self.search_kwargs
            )
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                question, k=self.k, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        args = {self.qa_chain.documents_key: docs, self.qa_chain.question_key: question}

        result = self.qa_chain(args)
        answer = result[self.qa_chain.output_key]

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}
