"""Chain for question-answering against a vector database based on the LLMs token Window."""  # noqa: E501
from __future__ import annotations

from typing import List

from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document


# This class extends the RetrievalQA class and adds a token limit to the documents retrieved. # noqa: E501
class RetrievalQAWithTokenLimit(RetrievalQA):
    # If True, the number of documents will be reduced to stay below the max token limit. # noqa: E501
    reduce_k_below_max_tokens: bool = True
    # The maximum number of tokens allowed in the documents.
    max_tokens_limit: int = 3375

    # This method reduces the number of documents so that the total number of tokens stays below the limit. # noqa: E501
    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        # If reduce_k_below_max_tokens is True and the combine_documents_chain is an instance of StuffDocumentsChain, # noqa: E501
        # calculate the total number of tokens in the documents.
        if self.reduce_k_below_max_tokens and isinstance(
            self.combine_documents_chain, StuffDocumentsChain
        ):
            tokens = [
                # Get the number of tokens in each document.
                self.combine_documents_chain.llm_chain._get_num_tokens(doc.page_content)
                for doc in docs
            ]
            # Calculate the total number of tokens.
            token_count = sum(tokens[:num_docs])
            # While the total number of tokens is greater than the limit, reduce the number of documents. # noqa: E501
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        # Return the reduced list of documents.
        return docs[:num_docs]

    # This method retrieves the relevant documents for a given question and reduces the number of documents to stay below the token limit. # noqa: E501
    def _get_docs(
        self, question: str, *, run_manager: CallbackManagerForChainRun
    ) -> List[Document]:
        # Retrieve the relevant documents.
        docs = self.retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        # Reduce the number of documents to stay below the token limit.
        return self._reduce_tokens_below_limit(docs)

    # This method is the asynchronous version of _get_docs.
    async def _aget_docs(
        self, question: str, *, run_manager: AsyncCallbackManagerForChainRun
    ) -> List[Document]:
        # Asynchronously retrieve the relevant documents.
        docs = await self.retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        # Reduce the number of documents to stay below the token limit.
        return self._reduce_tokens_below_limit(docs)

    # This property returns the type of the chain.
    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "retrieval_qa_with_token_limit"
