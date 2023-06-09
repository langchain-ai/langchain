"""Question-answering with sources over an index."""

from typing import Any, Dict, List, Optional

from pydantic import Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.docstore.document import Document
from langchain.schema.retriever import BaseRetriever


class RetrievalQAWithSourcesChain(BaseQAWithSourcesChain):
    """Question-answering with sources over an index."""

    retriever: BaseRetriever = Field(exclude=True)
    """Index to connect to."""
    reduce_k_below_max_tokens: bool = False
    """Reduce the number of results to return from store based on tokens limit"""
    max_tokens_limit: int = 3375
    """Restrict the docs to return from store based on tokens,
    enforced only for StuffDocumentChain and if reduce_k_below_max_tokens is to true"""

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
        self,
        inputs: Dict[str, Any],
        *,
        run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> List[Document]:
        run_manager_ = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.question_key]
        docs = self.retriever.retrieve(question, callbacks=run_manager_.get_child())
        return self._reduce_tokens_below_limit(docs)

    async def _aget_docs(
        self,
        inputs: Dict[str, Any],
        *,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None
    ) -> List[Document]:
        run_manager_ = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.question_key]
        docs = await self.retriever.aretrieve(
            question, callbacks=run_manager_.get_child()
        )
        return self._reduce_tokens_below_limit(docs)
