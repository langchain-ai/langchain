from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils.pydantic import get_fields

from langchain_community.vectorstores.utils import maximal_marginal_relevance


class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    mmr = "mmr"


class DocArrayRetriever(BaseRetriever):
    """`DocArray Document Indices` retriever.

    Currently, it supports 5 backends:
    InMemoryExactNNIndex, HnswDocumentIndex, QdrantDocumentIndex,
    ElasticDocIndex, and WeaviateDocumentIndex.

    Args:
        index: One of the above-mentioned index instances
        embeddings: Embedding model to represent text as vectors
        search_field: Field to consider for searching in the documents.
            Should be an embedding/vector/tensor.
        content_field: Field that represents the main content in your document schema.
            Will be used as a `page_content`. Everything else will go into `metadata`.
        search_type: Type of search to perform (similarity / mmr)
        filters: Filters applied for document retrieval.
        top_k: Number of documents to return
    """

    index: Any
    embeddings: Embeddings
    search_field: str
    content_field: str
    search_type: SearchType = SearchType.similarity
    top_k: int = 1
    filters: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        query_emb = np.array(self.embeddings.embed_query(query))

        if self.search_type == SearchType.similarity:
            results = self._similarity_search(query_emb)
        elif self.search_type == SearchType.mmr:
            results = self._mmr_search(query_emb)
        else:
            raise ValueError(
                f"Search type {self.search_type} does not exist. "
                f"Choose either 'similarity' or 'mmr'."
            )

        return results

    def _search(
        self, query_emb: np.ndarray, top_k: int
    ) -> List[Union[Dict[str, Any], Any]]:
        """
        Perform a search using the query embedding and return top_k documents.

        Args:
            query_emb: Query represented as an embedding
            top_k: Number of documents to return

        Returns:
            A list of top_k documents matching the query
        """

        from docarray.index import ElasticDocIndex, WeaviateDocumentIndex

        filter_args = {}
        search_field = self.search_field
        if isinstance(self.index, WeaviateDocumentIndex):
            filter_args["where_filter"] = self.filters
            search_field = ""
        elif isinstance(self.index, ElasticDocIndex):
            filter_args["query"] = self.filters
        else:
            filter_args["filter_query"] = self.filters

        if self.filters:
            query = (
                self.index.build_query()  # get empty query object
                .find(
                    query=query_emb, search_field=search_field
                )  # add vector similarity search
                .filter(**filter_args)  # add filter search
                .build(limit=top_k)  # build the query
            )
            # execute the combined query and return the results
            docs = self.index.execute_query(query)
            if hasattr(docs, "documents"):
                docs = docs.documents
            docs = docs[:top_k]
        else:
            docs = self.index.find(
                query=query_emb, search_field=search_field, limit=top_k
            ).documents
        return docs

    def _similarity_search(self, query_emb: np.ndarray) -> List[Document]:
        """
        Perform a similarity search.

        Args:
            query_emb: Query represented as an embedding

        Returns:
            A list of documents most similar to the query
        """
        docs = self._search(query_emb=query_emb, top_k=self.top_k)
        results = [self._docarray_to_langchain_doc(doc) for doc in docs]
        return results

    def _mmr_search(self, query_emb: np.ndarray) -> List[Document]:
        """
        Perform a maximal marginal relevance (mmr) search.

        Args:
            query_emb: Query represented as an embedding

        Returns:
            A list of diverse documents related to the query
        """
        docs = self._search(query_emb=query_emb, top_k=20)

        mmr_selected = maximal_marginal_relevance(
            query_emb,
            [
                doc[self.search_field]
                if isinstance(doc, dict)
                else getattr(doc, self.search_field)
                for doc in docs
            ],
            k=self.top_k,
        )
        results = [self._docarray_to_langchain_doc(docs[idx]) for idx in mmr_selected]
        return results

    def _docarray_to_langchain_doc(self, doc: Union[Dict[str, Any], Any]) -> Document:
        """
        Convert a DocArray document (which also might be a dict)
        to a langchain document format.

        DocArray document can contain arbitrary fields, so the mapping is done
        in the following way:

        page_content <-> content_field
        metadata <-> all other fields excluding
            tensors and embeddings (so float, int, string)

        Args:
            doc: DocArray document

        Returns:
            Document in langchain format

        Raises:
            ValueError: If the document doesn't contain the content field
        """

        fields = doc.keys() if isinstance(doc, dict) else get_fields(doc)

        if self.content_field not in fields:
            raise ValueError(
                f"Document does not contain the content field - {self.content_field}."
            )
        lc_doc = Document(
            page_content=doc[self.content_field]
            if isinstance(doc, dict)
            else getattr(doc, self.content_field)
        )

        for name in fields:
            value = doc[name] if isinstance(doc, dict) else getattr(doc, name)
            if (
                isinstance(value, (str, int, float, bool))
                and name != self.content_field
            ):
                lc_doc.metadata[name] = value

        return lc_doc
