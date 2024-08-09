from __future__ import annotations
from ast import Return
import base64

import json
import logging
from math import e
import os
from dataclasses import dataclass, field
from hashlib import md5
import re
from typing import (
    Any,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

try:
    import trieve_client
    from trieve_client.api.chunk import delete_chunk, create_chunk
    from trieve_client.api.file import upload_file_handler
    from trieve_client.api.chunk import search_chunk
    from trieve_client.models import (
        ErrorResponseBody,
        CreateChunkData,
        UploadFileData,
        UploadFileResult,
        ReturnCreatedChunk,
        SearchChunkData,
        SearchChunkQueryResponseBody,
    )
except ImportError:
    raise ImportError(
        "Could not import trieve_client python package. "
        "Please install it with `pip install trieve_client_py` "
    )

logger = logging.getLogger(__name__)


class Trieve(VectorStore):
    """`Trieve API` vector store.

     See (https://trieve.ai).

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import Trieve

            vectorstore = Trieve(
                trieve_dataset_id=trieve_dataset_id,
                trieve_api_key=trieve_api_key,
            )
    """

    def __init__(
        self,
        trieve_dataset_id: Optional[str] = None,
        trieve_api_key: Optional[str] = None,
        base_url: Optional[str] = "https://api.trieve.ai",
    ):
        """Initialize with Trieve API."""
        self._trieve_dataset_id = trieve_dataset_id or os.environ.get(
            "TRIEVE_DATASET_ID"
        )
        self._trieve_api_key = trieve_api_key or os.environ.get("TRIEVE_API_KEY")

        if self._trieve_dataset_id is None or self._trieve_api_key is None:
            logger.warning(
                "Can't find Trieve credentials or dataset_id in " "environment."
            )
            raise ValueError(
                "Can't find Trieve credentials or dataset_id in " "environment."
            )
        else:
            logger.debug(f"Using dataset id {self._trieve_dataset_id}")

        self.client = trieve_client.AuthenticatedClient(
            prefix="", _base_url=base_url, token=self._trieve_api_key
        ).with_headers(
            {
                "TR-Dataset": self._trieve_dataset_id,
            }
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return None

    def _get_post_headers(self) -> dict:
        """Returns headers that should be attached to each post request."""
        return {
            "Authentication": self._trieve_api_key,
            "TR-Dataset": self._trieve_dataset_id,
            "Content-Type": "application/json",
        }

    def _delete_doc(self, doc_id: str) -> bool:
        """
        Delete a document from the Trieve dataset.

        Args:
            doc_id (str): ID of the document to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """

        delete_request = delete_chunk.sync(
            chunk_id=doc_id, client=self.client, tr_dataset=self._trieve_dataset_id
        )
        if type(delete_request) == ErrorResponseBody:
            logger.error(f"Delete request failed for doc_id = {doc_id}")
            return False
        return True

    def _index_doc(
        self, doc: CreateChunkData
    ) -> ReturnCreatedChunk | ErrorResponseBody | None:
        created_chunk = create_chunk.sync(
            client=self.client, body=doc, tr_dataset=self._trieve_dataset_id
        )

        return created_chunk

    def add_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Trieve provides a way to add documents directly via our API where
        pre-processing and chunking occurs internally in an optimal way
        This method provides a way to use that API in LangChain

        Args:
            files_list: Iterable of strings, each representing a local file path.
                    Files could be text, HTML, PDF, markdown, doc/docx, ppt/pptx, etc.
                    see API docs for full list
            metadatas: Optional list of metadatas associated with each file

        Returns:
            List of ids associated with each of the files indexed
        """
        doc_ids = []
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist, skipping")
        md = metadata if metadata else {}
        with open(file_path, "rb") as f:
            file_data = UploadFileData(
                base64_file=base64.urlsafe_b64decode(f.read()).decode("utf-8"),
                file_name=os.path.splitext(file_path)[0],
                metadata=json.dumps(md),
                file_mime_type=os.path.splitext(file_path)[1],
            )
            file = upload_file_handler.sync(
                client=self.client,
                body=file_data,
                tr_dataset=self._trieve_dataset_id,
            )
            if type(file) == ErrorResponseBody:
                logger.error(
                    f"Failed to upload file {file_data.file_name}, because {file.message}"
                )
            elif type(file) == UploadFileResult:
                doc_ids.append(file.file_metadata.id)
        return doc_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
                - group_ids: list of group ids to associate with the document
                - link: link to the document
                - tag_set: list of tags to associate with the document
                - time_stamp: time stamp of the document
                - tracking_id: tracking id of the document
                - weight: weight of the document
                - metadata: metadata for the document

        This function indexes all the input text strings in the Trieve dataset as a
        single Trieve document, where each input text is considered a "section" and the
        metadata are associated with each section.
        if 'doc_metadata' is provided, it is associated with the Trieve document.

        Returns:
            document ID of the document added

        """
        results = []
        for i, text in enumerate(texts):
            doc = CreateChunkData(
                chunk_html=text,
                metadata=metadatas[i]["metadata"] if metadatas else {},
                group_ids=metadatas[i]["group_ids"] if metadatas else [],
                link=metadatas[i]["link"] if metadatas else "",
                tag_set=metadatas[i]["tag_set"] if metadatas else [],
                time_stamp=metadatas[i]["time_stamp"] if metadatas else "",
                tracking_id=metadatas[i]["tracking_id"] if metadatas else "",
                weight=metadatas[i]["weight"] if metadatas else 1.0,
            )

            created_chunk = self._index_doc(doc)

            if type(created_chunk) == ErrorResponseBody:
                print(
                    f"Failed to add document to Trieve. Reason: {created_chunk.message}"
                )
            elif type(created_chunk) == ReturnCreatedChunk:
                results.append(created_chunk.chunk_metadata.id)

        return results

    def add_chunk(
        self,
        chunk: str,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> str:
        """
        Trieve provides a way to add documents directly via our API where
        pre-processing and chunking occurs internally in an optimal way
        This method provides a way to use that API in LangChain

        Args:
            chunk: Text to add to the vectorstore.
            metadata: Optional metadata associated with the chunk
                - group_ids: list of group ids to associate with the document
                - link: link to the document
                - tag_set: list of tags to associate with the document
                - time_stamp: time stamp of the document
                - tracking_id: tracking id of the document
                - weight: weight of the document
                - metadata: metadata for the document

        Returns:
            ID associated with the chunk indexed
        """

        md = metadata if metadata else {}
        doc = CreateChunkData(
            chunk_html=chunk,
            metadata=md["metadata"] if md else {},
            group_ids=md["group_ids"] if md else [],
            link=md["link"] if md else "",
            tag_set=md["tag_set"] if md else [],
            time_stamp=md["time_stamp"] if md else "",
            tracking_id=md["tracking_id"] if md else "",
            weight=md["weight"] if md else 1.0,
        )
        created_chunk = self._index_doc(doc)

        if type(created_chunk) == ErrorResponseBody:
            print(f"Failed to add document to Trieve. Reason: {created_chunk.message}")
            return ""
        elif type(created_chunk) == ReturnCreatedChunk:
            return created_chunk.chunk_metadata.id
        return ""

    def trieve_query(
        self,
        query: str,
        search_type: str = "hybrid",
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run a Trieve query

        Args:
            query: Text to look up documents similar to.
            search_type: Type of search to perform. Options are "hybrid", "semantic", or "fulltext".
            kwargs:
                - date_bias: bias towards more recent documents
                - filters: filter based on key value pairs in metadata
                - get_collisions: return documents that are similar to the doc but not the same
                - highlight_delimiters: delimiters for highlighting the matching segments
                - highlight_results: whether to highlight the matching segments
                - link: filter based on the link of the doc
                - page: page number of the results
                - tag_set: filter based on the tag_set of the doc
                - time_range: filter based on the time_stamp of the doc
            for more details see https://api.trieve.ai/redoc#tag/chunk/operation/search_chunk
        Returns:
            A list of k Documents matching the given query
            If summary is enabled, last document is the summary text with 'summary'=True
        """
        res = []
        data = SearchChunkData(query=query, search_type=search_type, **kwargs)
        search = search_chunk.sync(
            body=data,
            client=self.client,
            tr_dataset=self._trieve_dataset_id,
        )

        if type(search) == ErrorResponseBody:
            logger.error(f"Search request failed for query = {query}")
            return res
        elif type(search) == SearchChunkQueryResponseBody:
            for doc in search.score_chunks:
                res.append(
                    (
                        Document(doc.metadata[0].chunk_html, metadata=doc.metadata),
                        doc.score,
                    )
                )

        return res

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return Trieve documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 10.
            kwargs:
                - date_bias: bias towards more recent documents
                - filters: filter based on key value pairs in metadata
                - get_collisions: return documents that are similar to the doc but not the same
                - highlight_delimiters: delimiters for highlighting the matching segments
                - highlight_results: whether to highlight the matching segments
                - link: filter based on the link of the doc
                - page: page number of the results
                - tag_set: filter based on the tag_set of the doc
                - time_range: filter based on the time_stamp of the doc
            for more details see https://api.trieve.ai/redoc#tag/chunk/operation/search_chunk
        Returns:
            List of Documents most similar to the query and score for each.
        """
        docs = self.trieve_query(query, search_type="hybrid", **kwargs)
        result = [doc[0] for i, doc in enumerate(docs) if i < k]
        return result

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return Trieve documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 10.
            kwargs:
                - date_bias: bias towards more recent documents
                - filters: filter based on key value pairs in metadata
                - get_collisions: return documents that are similar to the doc but not the same
                - highlight_delimiters: delimiters for highlighting the matching segments
                - highlight_results: whether to highlight the matching segments
                - link: filter based on the link of the doc
                - page: page number of the results
                - tag_set: filter based on the tag_set of the doc
                - time_range: filter based on the time_stamp of the doc
            for more details see https://api.trieve.ai/redoc#tag/chunk/operation/search_chunk
        Returns:
            List of Documents most similar to the query and score for each.
        """
        docs = self.trieve_query(query, search_type="hybrid", **kwargs)
        return docs

    def full_text_search(  # type: ignore[override]
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return Trieve documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            any other querying variable in TrieveQueryConfig

        Returns:
            List of Documents most similar to the query
        """
        return self.trieve_query(query, search_type="fulltext", **kwargs)

    def semantic_search(  # type: ignore[override]
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return Trieve documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            any other querying variable in TrieveQueryConfig

        Returns:
            List of Documents most similar to the query
        """
        return self.trieve_query(query, search_type="semantic", **kwargs)

    def hybrid_search(  # type: ignore[override]
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return Trieve documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            any other querying variable in TrieveQueryConfig

        Returns:
            List of Documents most similar to the query
        """
        return self.trieve_query(query, search_type="hybrid", **kwargs)

    @classmethod
    def from_texts(
        cls: Type[Trieve],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Trieve:
        """Construct Trieve wrapper from raw documents.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python

                from langchain_community.vectorstores import Trieve
                trieve = Trieve.from_texts(
                    texts,
                    trieve_dataset_id=dataset_id,
                    trieve_api_key=api_key,
                )
        """
        # Notes:
        # * Trieve generates its own embeddings, so we ignore the provided
        #   embeddings (required by interface)
        # * when metadatas[] are provided they are associated with each "part"
        #   in Trieve. doc_metadata can be used to provide additional metadata
        #   for the document itself (applies to all "texts" in this call)
        trieve = cls(**kwargs)
        trieve.add_texts(
            texts,
            metadatas=metadatas,
        )
        return trieve

    @classmethod
    def from_file(
        cls: Type[Trieve],
        file: str,
        embedding: Optional[Embeddings] = None,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> Trieve:
        """Construct Trieve wrapper from raw documents.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python

                from langchain_community.vectorstores import Trieve
                trieve = Trieve.from_files(
                    files_list,
                    trieve_dataset_id=dataset_id,
                    trieve_api_key=api_key,
                )
        """
        # Note: Trieve generates its own embeddings, so we ignore the provided
        # embeddings (required by interface)
        trieve = cls(**kwargs)
        trieve.add_file(file, metadata)
        return trieve


class TrieveRetriever(VectorStoreRetriever):
    """Retriever class for `Trieve`."""

    vectorstore: Trieve
    """Trieve vectorstore."""
    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "fulltext",
        "semantic",
        "hybrid",
    )

    """Search params.
        k: Number of Documents to return. Defaults to 5.
        search_type: Type of search to perform. Options are "hybrid", "semantic", or "fulltext".
        search_kwargs: Keyword arguments to pass to the search function.
            - date_bias: bias towards more recent documents
            - filters: filter based on key value pairs in metadata
            - get_collisions: return documents that are similar to the doc but not the same
            - highlight_delimiters: delimiters for highlighting the matching segments
            - highlight_results: whether to highlight the matching segments
            - link: filter based on the link of the doc
            - page: page number of the results
            - tag_set: filter based on the tag_set of the doc
            - time_range: filter based on the time_stamp of the doc
        for more details see https://api.trieve.ai/redoc#tag/chunk/operation/search_chunk
    """

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        return self.vectorstore.add_texts(
            [doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )

    def add_file(
        self, file_path: str, metadata: Optional[dict] = None, **kwargs: Any
    ) -> List[str]:
        return self.vectorstore.add_file(file_path, metadata)

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs: Any
    ) -> List[str]:
        return self.vectorstore.add_texts(texts, metadatas)

    def get_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Any = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        run_name: str | None = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return Trieve documents most similar to query"""
        if self.search_type not in self.allowed_search_types:
            raise ValueError(
                f"search_type must be one of {self.allowed_search_types}, "
                f"not {self.search_type}"
            )
        result = []
        if self.search_type == "similarity":
            result = self.vectorstore.similarity_search(query, **kwargs)
        elif self.search_type == "fulltext":
            result = [
                doc[0] for doc in self.vectorstore.full_text_search(query, **kwargs)
            ]
        elif self.search_type == "semantic":
            result = [
                doc[0] for doc in self.vectorstore.semantic_search(query, **kwargs)
            ]
        elif self.search_type == "hybrid":
            result = [doc[0] for doc in self.vectorstore.hybrid_search(query, **kwargs)]
        return result
