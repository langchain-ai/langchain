"""Question-answering with sources over a vector database."""

import warnings
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pydantic import Field, model_validator
from typing_extensions import override

from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.qa_with_sources.base import BaseQAWithSourcesChain


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
    search_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra search args."""

    def _reduce_tokens_below_limit(self, docs: list[Document]) -> list[Document]:
        num_docs = len(docs)

        if self.reduce_k_below_max_tokens and isinstance(
            self.combine_documents_chain,
            StuffDocumentsChain,
        ):
            tokens = [
                self.combine_documents_chain.llm_chain._get_num_tokens(doc.page_content)  # noqa: SLF001
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    @override
    def _get_docs(
        self,
        inputs: dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        question = inputs[self.question_key]
        docs = self.vectorstore.similarity_search(
            question,
            k=self.k,
            **self.search_kwargs,
        )
        return self._reduce_tokens_below_limit(docs)

    async def _aget_docs(
        self,
        inputs: dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> list[Document]:
        msg = "VectorDBQAWithSourcesChain does not support async"
        raise NotImplementedError(msg)

    @model_validator(mode="before")
    @classmethod
    def _raise_deprecation(cls, values: dict) -> Any:
        warnings.warn(
            "`VectorDBQAWithSourcesChain` is deprecated - "
            "please use `from langchain_classic.chains import "
            "RetrievalQAWithSourcesChain`",
            stacklevel=5,
        )
        return values

    @property
    def _chain_type(self) -> str:
        return "vector_db_qa_with_sources_chain"
