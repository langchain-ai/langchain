"""Wrapper around Vectara vector database."""
from __future__ import annotations

from typing import Any, List, Optional, Iterable, Tuple
import json
import logging
from hashlib import md5
import requests
from colorama import Fore, Style

from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore


def _error_msg(response):
    """Returns an error message constructed from the passed http response."""
    return f"(code {response.status_code}, reason {response.reason}, details {response.text})"


"""Implementation of Vector Store using Vectara (https://vectara.com)"""
class Vectara(VectorStore):
    """Wrapper around Vectara

    Example:
        .. code-block:: python

            from langchain.vectorstores import Vectara

            # The environment should be the one specified next to the API key
            vectorstore = Vectara(index, "text")
    """

    def __init__(
        self,
        vectara_customer_id: int,
        vectara_corpus_id: int,
        vectara_api_key: str
    ):
        """Initialize with Vectara API."""
        self._vectara_customer_id = vectara_customer_id
        self._vectara_corpus_id = vectara_corpus_id
        self._vectara_api_key = vectara_api_key
        self._session = requests.Session()  # to resuse connections
        logging.debug("Using corpus id %s", self._vectara_corpus_id)

    def _get_post_headers(self):
        """Returns headers that should be attached to each post request."""
        return {
            "x-api-key": self._vectara_api_key,
            "customer-id": str(self._vectara_customer_id),
            "Content-Type": "application/json",
        }

    def _delete_doc(self, doc_id: str):
        """
        Delete a document from the Vectara corpus.

        Args:
            url (str): URL of the page to delete.
            doc_id (str): ID of the document to delete.

        Returns:
            bool: True if the delete was successful, False otherwise.
        """
        body = {'customer_id': self._vectara_customer_id, 'corpus_id': self._vectara_corpus_id, 'document_id': doc_id}
        post_headers = { 'x-api-key': self._vectara_api_key, 'customer-id': str(self._vectara_customer_id) }
        response = requests.post(
            f"https://api.vectara.io/v1/delete-doc", data=json.dumps(body),
            verify=True, headers=post_headers)        
        if response.status_code != 200:
            logging.error(f"Delete request failed for doc_id = {doc_id} with status code {response.status_code}, reason {response.reason}, text {response.text}")
            return False
        return True

    def _index_doc(self, doc_id, text, metadata):
        request = {}
        request["customer_id"] = self._vectara_customer_id
        request["corpus_id"] = self._vectara_corpus_id
        request["document"] = {"document_id": doc_id, "metadta": metadata, "parts": [{"text": text}]}

        response = self._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/core/index",
            data=json.dumps(request),
            timeout=30,
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
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return Vectara documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 5.
            filter: Dictionary of argument(s) to filter on metadata. For example a filter can be "doc.rating > 3.0 and part.lang = 'deu'"}
                see https://docs.vectara.com/docs/search-apis/sql/filter-overview for more details
        Returns:
            List of Documents most similar to the query and score for each
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
                            "corpus_key": [
                                {
                                    "customer_id": self._vectara_customer_id,
                                    "corpus_id": self._vectara_corpus_id,
                                    "metadataFilter": filter,
                                }
                            ],
                        }
                    ]
                }
            ),
            timeout=10,
        )

        if response.status_code != 200:
            logging.error("Query failed %s", _error_msg(response))
            return []

        result = response.json()
        return [(Document(page_content=x["text"], metadata=x["metadata"]), x["score"]) for x in result["responseSet"][0]["response"]]

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return Vectara documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 5.
            filter: Dictionary of argument(s) to filter on metadata. For example a filter can be "doc.rating > 3.0 and part.lang = 'deu'"}
                see https://docs.vectara.com/docs/search-apis/sql/filter-overview for more details
        Returns:
            List of Documents most similar to the query
        """
        docs_and_scores = self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        customer_id: int,
        corpus_id: int,
        api_key: str,
        **kwargs: Any,
    ) -> Vectara:
        """Construct Vectara wrapper from raw documents.
        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import Vectara
                vectara = Vectara.from_texts(
                    texts,
                )
        """

        vectara = cls(customer_id, corpus_id, api_key)
        vectara.add_texts(texts)
        return vectara
