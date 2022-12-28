"""Chain for question-answering against multiple vector databases."""
from __future__ import annotations

from typing import Dict, List, Optional

from langchain.chains.vector_db_qa.base import VectorDBQA
from langchain.vectorstores.base import VectorStore


class MultiVectorDBQA(VectorDBQA):
    """Chain for question-answering against multiple vector database.

    Example:
        .. code-block:: python

            from langchain import OpenAI, VectorDBQA
            from langchain.faiss import FAISS
            vectordb1 = FAISS(...)
            vectordb2 = FAISS(...)
            vectordbQA = VectorDBQA(llm=OpenAI(), vectorstore=[vectordb1, vectordb2])

    """

    # This is just to remove the singular field from base class
    vectorstore: Optional[VectorStore]  # type: ignore

    vectorstores: List[VectorStore]
    """Vector Databases to connect to."""

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Call all vectorstores to get relevant documents before searching."""
        question = inputs[self.input_key]

        # Get docs from all vectorstores
        n = len(self.vectorstores)
        k = self.k // n
        docs = []

        for vectorstore in self.vectorstores:
            docs.extend(vectorstore.similarity_search(question, k=k))

        # Get answer
        answer, _ = self.combine_documents_chain.combine_docs(docs, question=question)

        return {self.output_key: answer}
