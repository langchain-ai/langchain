"""Question-answering with sources over a vector database."""
from typing import Any, Dict, List

from pydantic import BaseModel

from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore


class VectorDBQAWithSourcesChain(BaseQAWithSourcesChain, BaseModel):
    """Question-answering with sources over a vector database."""

    vectorstore: VectorStore
    """Vector Database to connect to."""
    k: int = 4

    def _get_docs(self, inputs: Dict[str, Any]) -> List[Document]:
        question = inputs[self.question_key]
        return self.vectorstore.similarity_search(question, k=self.k)
