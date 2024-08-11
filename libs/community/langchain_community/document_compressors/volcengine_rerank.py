from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env


class VolcengineRerank(BaseDocumentCompressor):
    """Document compressor that uses `Volcengine Rerank API`."""

    client: Any = None
    """Volcengine client to use for compressing documents."""

    ak: Optional[str] = None
    """Access Key ID. 
    https://www.volcengine.com/docs/84313/1254553"""

    sk: Optional[str] = None
    """Secret Access Key. 
    https://www.volcengine.com/docs/84313/1254553"""

    region: str = "api-vikingdb.volces.com"
    """https://www.volcengine.com/docs/84313/1254488. """

    host: str = "cn-beijing"
    """https://www.volcengine.com/docs/84313/1254488. """

    top_n: Optional[int] = 3
    """Number of documents to return."""

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        extra = "forbid"

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""

        if not values.get("client"):
            try:
                from volcengine.viking_db import VikingDBService
            except ImportError:
                raise ImportError(
                    "Could not import volcengine python package. "
                    "Please install it with `pip install volcengine` "
                    "or `pip install --user volcengine`."
                )

            values["ak"] = get_from_dict_or_env(values, "ak", "VOLC_API_AK")
            values["sk"] = get_from_dict_or_env(values, "sk", "VOLC_API_SK")

            values["client"] = VikingDBService(
                host="api-vikingdb.volces.com",
                region="cn-beijing",
                scheme="https",
                connection_timeout=30,
                socket_timeout=30,
                ak=values["ak"],
                sk=values["sk"],
            )

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
            {
                "query": query,
                "content": doc.page_content if isinstance(doc, Document) else doc,
            }
            for doc in documents
        ]

        from volcengine.viking_db import VikingDBService

        client: VikingDBService = self.client
        results = client.batch_rerank(docs)

        result_dicts = []
        for index, score in enumerate(results):
            result_dicts.append({"index": index, "relevance_score": score})

        result_dicts.sort(key=lambda x: x["relevance_score"], reverse=True)
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n

        return result_dicts[:top_n]

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Volcengine's rerank API.

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
