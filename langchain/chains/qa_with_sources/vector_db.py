"""Question-answering with sources over a vector database."""
import sys

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore


class VectorDBQAWithSourcesChain(BaseQAWithSourcesChain, BaseModel):
    """Question-answering with sources over a vector database."""

    vectorstore: VectorStore
    """Vector Database to connect to."""
    k: int = 4
    """Number of results to return from store"""
    reduce_k_below_max_tokens: bool = False
    """Reduce the number of results to return from store based on tokens limit"""
    max_tokens_limit: int = 3375
    """Restrict the docs to return from store based on tokens, only applicable if reduce_k_below_max_tokens set to true"""
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Extra search args."""

    def _get_num_tokens(self, text: str) -> int:
        """Calculate num tokens with tiktoken package."""
        # tiktoken NOT supported for Python 3.8 or below
        if sys.version_info[1] <= 8:
            return super().get_num_tokens(text)
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please it install it with `pip install tiktoken`."
            )
        # create a GPT-3 encoder instance
        enc = tiktoken.get_encoding("gpt2")

        # encode the text using the GPT-3 encoder
        tokenized_text = enc.encode(text)

        # calculate the number of tokens in the encoded text
        return len(tokenized_text)

    def _page_content(self, doc: Document) -> str:
        return doc.page_content

    def _reduce_tokens_below_limit(self, docs) -> List[Document]:
        tokens = len(self._get_num_tokens("".join(map(self._page_content, docs))))
        return docs if (tokens <= self.max_tokens_limit) else self._reduce_tokens_below_limit(docs[:-1])

    def _get_docs(self, inputs: Dict[str, Any]) -> List[Document]:
        question = inputs[self.question_key]
        docs = self.vectorstore.similarity_search(
            question, k=self.k, **self.search_kwargs
        )
        return self._reduce_tokens_below_limit(docs)
