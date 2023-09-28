"""Question-answering with sources over a vector database."""

import warnings
from typing import Any, Dict, List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.docstore.document import Document
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema.vectorstore import VectorStore


class VectorDBQAWithSourcesChain(BaseQAWithSourcesChain):
    """Question-answering with sources over a vector database."""

    vectorstore: VectorStore = Field(exclude=True)
    """Vector Database to connect to."""
    k: int = 4
    """Number of results to return from store"""
    reduce_k_below_max_tokens: bool = False
    """Reduce the number of results to return from store based on tokens limit"""
    max_tokens_limit: int = 3375
    """Restrict the docs to return from store based on tokens,
    enforced only for StuffDocumentChain and if reduce_k_below_max_tokens is to true"""
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Extra search args."""

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        if self.reduce_k_below_max_tokens and isinstance(
            self.combine_documents_chain, StuffDocumentsChain
        ):
            tokens = [
                self.combine_documents_chain.llm_chain.llm.get_num_tokens(
                    doc.page_content
                )
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(
        self, inputs: Dict[str, Any], *, run_manager: CallbackManagerForChainRun
    ) -> List[Document]:
        question = inputs[self.question_key]
        docs = self.vectorstore.similarity_search(
            question, k=self.k, **self.search_kwargs
        )
        return self._reduce_tokens_below_limit(docs)

    async def _aget_docs(
        self, inputs: Dict[str, Any], *, run_manager: AsyncCallbackManagerForChainRun
    ) -> List[Document]:
        raise NotImplementedError("VectorDBQAWithSourcesChain does not support async")

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        warnings.warn(
            "`VectorDBQAWithSourcesChain` is deprecated - "
            "please use `from langchain.chains import RetrievalQAWithSourcesChain`"
        )
        return values

    @property
    def _chain_type(self) -> str:
        return "vector_db_qa_with_sources_chain"
