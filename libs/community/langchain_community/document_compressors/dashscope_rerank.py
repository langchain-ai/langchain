from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env


class DashScopeRerank(BaseDocumentCompressor):
    """Document compressor that uses `DashScope Rerank API`."""

    client: Any = None
    """DashScope client to use for compressing documents."""

    model: Optional[str] = None
    """Model to use for reranking."""

    top_n: Optional[int] = 3
    """Number of documents to return."""

    dashscope_api_key: Optional[str] = Field(None, alias="api_key")
    """DashScope API key. Must be specified directly or via environment variable 
        DASHSCOPE_API_KEY."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""

        if not values.get("client"):
            try:
                import dashscope
            except ImportError:
                raise ImportError(
                    "Could not import dashscope python package. "
                    "Please install it with `pip install dashscope`."
                )

            values["client"] = dashscope.TextReRank
            values["dashscope_api_key"] = get_from_dict_or_env(
                values, "dashscope_api_key", "DASHSCOPE_API_KEY"
            )
            values["model"] = dashscope.TextReRank.Models.gte_rerank

        return values

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        top_n: Optional[int] = -1,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            top_n : The number of results to return. If None returns all results.
                Defaults to self.top_n.
        """  # noqa: E501

        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]

        top_n = top_n if (top_n is None or top_n > 0) else self.top_n

        results = self.client.call(
            model=self.model,
            query=query,
            documents=docs,
            top_n=top_n,
            return_documents=False,
            api_key=self.dashscope_api_key,
        )

        result_dicts = []
        for res in results.output.results:
            result_dicts.append(
                {"index": res.index, "relevance_score": res.relevance_score}
            )
        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using DashScope's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed
