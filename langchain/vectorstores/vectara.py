"""Wrapper around Vectara vector database."""
from __future__ import annotations

import json
import logging
from hashlib import md5
from typing import Any, Iterable, List, Optional, Tuple, Type

import requests

from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.base import VectorStore


class Vectara(VectorStore):
    """Implementation of Vector Store using Vectara (https://vectara.com)
    Example:
        .. code-block:: python

            from langchain.vectorstores import Vectara
            vectorstore = Vectara(customer_id, corpus_id, api_key)
    """

    def __init__(
        self, vectara_customer_id: int, vectara_corpus_id: int, vectara_api_key: str
    ):
        """Initialize with Vectara API."""
        self._vectara_customer_id = vectara_customer_id
        self._vectara_corpus_id = vectara_corpus_id
        self._vectara_api_key = vectara_api_key
        self._session = requests.Session()  # to resuse connections
        logging.debug("Using corpus id %s", self._vectara_corpus_id)

    def _get_post_headers(self) -> dict:
        """Returns headers that should be attached to each post request."""
        return {
            "x-api-key": self._vectara_api_key,
            "customer-id": str(self._vectara_customer_id),
            "Content-Type": "application/json",
        }

    def _delete_doc(self, doc_id: str) -> bool:
        """
        Delete a document from the Vectara corpus.

        Args:
            url (str): URL of the page to delete.
            doc_id (str): ID of the document to delete.

        Returns:
            bool: True if the delete was successful, False otherwise.
        """
        body = {
            "customer_id": self._vectara_customer_id,
            "corpus_id": self._vectara_corpus_id,
            "document_id": doc_id,
        }
        post_headers = {
            "x-api-key": self._vectara_api_key,
            "customer-id": str(self._vectara_customer_id),
        }
        response = requests.post(
            "https://api.vectara.io/v1/delete-doc",
            data=json.dumps(body),
            verify=True,
            headers=post_headers,
        )
        if response.status_code != 200:
            logging.error(
                f"Delete request failed for doc_id = {doc_id} with status code "
                f"{response.status_code}, reason {response.reason}, text "
                f"{response.text}"
            )
            return False
        return True

    def _index_doc(self, doc_id: str, text: str, metadata: dict) -> bool:
        request: dict[str, Any] = {}
        request["customer_id"] = str(self._vectara_customer_id)
        request["corpus_id"] = str(self._vectara_corpus_id)
        request["document"] = {
            "document_id": doc_id,
            "metadataJson": json.dumps(metadata),
            "section": [{"text": text, "metadataJson": json.dumps(metadata)}],
        }

        response = self._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/index",
            data=json.dumps(request),
            timeout=30,
            verify=True,
        )

        status_code = response.status_code

        result = response.json()
        status_str = result["status"]["code"] if "status" in result else None
        if status_code == 409 or (status_str and status_str == "ALREADY_EXISTS"):
            return False
        else:
            return True

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

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        ids = [md5(text.encode("utf-8")).hexdigest() for text in texts]
        for i, doc in enumerate(texts):
            doc_id = ids[i]
            metadata = metadatas[i] if metadatas else {}
            succeeded = self._index_doc(doc_id, doc, metadata)
            if not succeeded:
                self._delete_doc(doc_id)
                self._index_doc(doc_id, doc, metadata)
        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.025,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return Vectara documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 5.
            alpha: parameter for hybrid search (called "lambda" in Vectara
                documentation).
            filter: Dictionary of argument(s) to filter on metadata. For example a
                filter can be "doc.rating > 3.0 and part.lang = 'deu'"} see
                https://docs.vectara.com/docs/search-apis/sql/filter-overview
                for more details.

        Returns:
            List of Documents most similar to the query and score for each.
        """
        response = self._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/query",
            data=json.dumps(
                {
                    "query": [
                        {
                            "query": query,
                            "start": 0,
                            "num_results": k,
                            "context_config": {
                                "sentences_before": 3,
                                "sentences_after": 3,
                            },
                            "corpus_key": [
                                {
                                    "customer_id": self._vectara_customer_id,
                                    "corpus_id": self._vectara_corpus_id,
                                    "metadataFilter": filter,
                                    "lexical_interpolation_config": {"lambda": alpha},
                                }
                            ],
                        }
                    ]
                }
            ),
            timeout=10,
        )

        if response.status_code != 200:
            logging.error(
                "Query failed %s",
                f"(code {response.status_code}, reason {response.reason}, details "
                f"{response.text})",
            )
            return []

        result = response.json()
        responses = result["responseSet"][0]["response"]
        vectara_default_metadata = ["lang", "len", "offset"]
        docs = [
            (
                Document(
                    page_content=x["text"],
                    metadata={
                        m["name"]: m["value"]
                        for m in x["metadata"]
                        if m["name"] not in vectara_default_metadata
                    },
                ),
                x["score"],
            )
            for x in responses
        ]
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.025,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return Vectara documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 5.
            filter: Dictionary of argument(s) to filter on metadata. For example a
                filter can be "doc.rating > 3.0 and part.lang = 'deu'"} see
                https://docs.vectara.com/docs/search-apis/sql/filter-overview for more
                details.

        Returns:
            List of Documents most similar to the query
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, alpha=alpha, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls: Type[Vectara],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Vectara:
        """Construct Vectara wrapper from raw documents.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python

                from langchain import Vectara
                vectara = Vectara.from_texts(texts, customer_id, corpus_id, api_key)
        """
        # Note: Vectara generates its own embeddings, so we ignore the provided
        # embeddings (required by interface).
        customer_id = kwargs["customer_id"]
        corpus_id = kwargs["corpus_id"]
        api_key = kwargs["api_key"]

        vectara = cls(customer_id, corpus_id, api_key)
        vectara.add_texts(texts, metadatas)
        return vectara

    @classmethod
    def from_documents(
        cls: Type[Vectara],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Vectara:
        """Create a Vectara vectorstore from a list of documents.

        Args:
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
        Returns:
            Vectara: Vectara vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts, embedding=embedding, metadatas=metadatas, **kwargs
        )


class VectaraRetriever(BaseRetriever):
    def __init__(
        self,
        store: Vectara,
        alpha: float = 0.025,
        k: int = 5,
        filter: Optional[str] = None,
    ):
        """Instantiate Vectara-backed Retriever.

        Args:
            store: Vectara vector store.
            k: Number of Documents to return. Defaults to 5.
            alpha: parameter for hybrid search (called "lambda" in Vectara
                documentation).
            filter: Dictionary of argument(s) to filter on metadata. For example a
                filter can be "doc.rating > 3.0 and part.lang = 'deu'"} see
                https://docs.vectara.com/docs/search-apis/sql/filter-overview
                for more details.
        """
        self.store = store
        self.alpha = alpha
        self.k = k
        self.filter = filter

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> None:
        """Add text to the Vectara vectorstore

        Args:
            texts (List[str]): The text
            metadatas (List[dict]): Metadata dicts, must line up with existing store
        """
        self.store.add_texts(texts, metadatas)

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.store.similarity_search(
            query, k=self.k, alpha=self.alpha, filter=self.filter
        )
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
