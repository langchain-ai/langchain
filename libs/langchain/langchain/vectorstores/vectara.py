from __future__ import annotations

import json
import logging
import os
from hashlib import md5
from typing import Any, Iterable, List, Optional, Tuple, Type

import requests

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import Field
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever

logger = logging.getLogger(__name__)


class Vectara(VectorStore):
    """`Vectara API` vector store.

     See (https://vectara.com).

    Example:
        .. code-block:: python

            from langchain.vectorstores import Vectara

            vectorstore = Vectara(
                vectara_customer_id=vectara_customer_id,
                vectara_corpus_id=vectara_corpus_id,
                vectara_api_key=vectara_api_key
            )
    """

    def __init__(
        self,
        vectara_customer_id: Optional[str] = None,
        vectara_corpus_id: Optional[str] = None,
        vectara_api_key: Optional[str] = None,
        vectara_api_timeout: int = 60,
    ):
        """Initialize with Vectara API."""
        self._vectara_customer_id = vectara_customer_id or os.environ.get(
            "VECTARA_CUSTOMER_ID"
        )
        self._vectara_corpus_id = vectara_corpus_id or os.environ.get(
            "VECTARA_CORPUS_ID"
        )
        self._vectara_api_key = vectara_api_key or os.environ.get("VECTARA_API_KEY")
        if (
            self._vectara_customer_id is None
            or self._vectara_corpus_id is None
            or self._vectara_api_key is None
        ):
            logger.warning(
                "Can't find Vectara credentials, customer_id or corpus_id in "
                "environment."
            )
        else:
            logger.debug(f"Using corpus id {self._vectara_corpus_id}")
        self._session = requests.Session()  # to reuse connections
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        self._session.mount("http://", adapter)
        self.vectara_api_timeout = vectara_api_timeout

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return None

    def _get_post_headers(self) -> dict:
        """Returns headers that should be attached to each post request."""
        return {
            "x-api-key": self._vectara_api_key,
            "customer-id": self._vectara_customer_id,
            "Content-Type": "application/json",
        }

    def _delete_doc(self, doc_id: str) -> bool:
        """
        Delete a document from the Vectara corpus.

        Args:
            url (str): URL of the page to delete.
            doc_id (str): ID of the document to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        body = {
            "customer_id": self._vectara_customer_id,
            "corpus_id": self._vectara_corpus_id,
            "document_id": doc_id,
        }
        response = self._session.post(
            "https://api.vectara.io/v1/delete-doc",
            data=json.dumps(body),
            verify=True,
            headers=self._get_post_headers(),
            timeout=self.vectara_api_timeout,
        )
        if response.status_code != 200:
            logger.error(
                f"Delete request failed for doc_id = {doc_id} with status code "
                f"{response.status_code}, reason {response.reason}, text "
                f"{response.text}"
            )
            return False
        return True

    def _index_doc(self, doc: dict) -> str:
        request: dict[str, Any] = {}
        request["customer_id"] = self._vectara_customer_id
        request["corpus_id"] = self._vectara_corpus_id
        request["document"] = doc

        response = self._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/index",
            data=json.dumps(request),
            timeout=self.vectara_api_timeout,
            verify=True,
        )

        status_code = response.status_code

        result = response.json()
        status_str = result["status"]["code"] if "status" in result else None
        if status_code == 409 or status_str and (status_str == "ALREADY_EXISTS"):
            return "E_ALREADY_EXISTS"
        elif status_str and (status_str == "FORBIDDEN"):
            return "E_NO_PERMISSIONS"
        else:
            return "E_SUCCEEDED"

    def add_files(
        self,
        files_list: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Vectara provides a way to add documents directly via our API where
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
        for inx, file in enumerate(files_list):
            if not os.path.exists(file):
                logger.error(f"File {file} does not exist, skipping")
                continue
            md = metadatas[inx] if metadatas else {}
            files: dict = {
                "file": (file, open(file, "rb")),
                "doc_metadata": json.dumps(md),
            }
            headers = self._get_post_headers()
            headers.pop("Content-Type")
            response = self._session.post(
                f"https://api.vectara.io/upload?c={self._vectara_customer_id}&o={self._vectara_corpus_id}&d=True",
                files=files,
                verify=True,
                headers=headers,
                timeout=self.vectara_api_timeout,
            )

            if response.status_code == 409:
                doc_id = response.json()["document"]["documentId"]
                logger.info(
                    f"File {file} already exists on Vectara (doc_id={doc_id}), skipping"
                )
            elif response.status_code == 200:
                doc_id = response.json()["document"]["documentId"]
                doc_ids.append(doc_id)
            else:
                logger.info(f"Error indexing file {file}: {response.json()}")

        return doc_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        doc_metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            doc_metadata: optional metadata for the document

        This function indexes all the input text strings in the Vectara corpus as a
        single Vectara document, where each input text is considered a "section" and the
        metadata are associated with each section.
        if 'doc_metadata' is provided, it is associated with the Vectara document.

        Returns:
            document ID of the document added

        """
        doc_hash = md5()
        for t in texts:
            doc_hash.update(t.encode())
        doc_id = doc_hash.hexdigest()
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if doc_metadata:
            doc_metadata["source"] = "langchain"
        else:
            doc_metadata = {"source": "langchain"}
        doc = {
            "document_id": doc_id,
            "metadataJson": json.dumps(doc_metadata),
            "section": [
                {"text": text, "metadataJson": json.dumps(md)}
                for text, md in zip(texts, metadatas)
            ],
        }

        success_str = self._index_doc(doc)
        if success_str == "E_ALREADY_EXISTS":
            self._delete_doc(doc_id)
            self._index_doc(doc)
        elif success_str == "E_NO_PERMISSIONS":
            print(
                """No permissions to add document to Vectara. 
                Check your corpus ID, customer ID and API key"""
            )
        return [doc_id]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        lambda_val: float = 0.025,
        filter: Optional[str] = None,
        n_sentence_context: int = 2,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return Vectara documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 5.
            lambda_val: lexical match parameter for hybrid search.
            filter: Dictionary of argument(s) to filter on metadata. For example a
                filter can be "doc.rating > 3.0 and part.lang = 'deu'"} see
                https://docs.vectara.com/docs/search-apis/sql/filter-overview
                for more details.
            n_sentence_context: number of sentences before/after the matching segment
                to add, defaults to 2

        Returns:
            List of Documents most similar to the query and score for each.
        """
        data = json.dumps(
            {
                "query": [
                    {
                        "query": query,
                        "start": 0,
                        "num_results": k,
                        "context_config": {
                            "sentences_before": n_sentence_context,
                            "sentences_after": n_sentence_context,
                        },
                        "corpus_key": [
                            {
                                "customer_id": self._vectara_customer_id,
                                "corpus_id": self._vectara_corpus_id,
                                "metadataFilter": filter,
                                "lexical_interpolation_config": {"lambda": lambda_val},
                            }
                        ],
                    }
                ]
            }
        )

        response = self._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/query",
            data=data,
            timeout=self.vectara_api_timeout,
        )

        if response.status_code != 200:
            logger.error(
                "Query failed %s",
                f"(code {response.status_code}, reason {response.reason}, details "
                f"{response.text})",
            )
            return []

        result = response.json()

        responses = result["responseSet"][0]["response"]
        documents = result["responseSet"][0]["document"]

        metadatas = []
        for x in responses:
            md = {m["name"]: m["value"] for m in x["metadata"]}
            doc_num = x["documentIndex"]
            doc_md = {m["name"]: m["value"] for m in documents[doc_num]["metadata"]}
            md.update(doc_md)
            metadatas.append(md)

        docs = [
            (
                Document(
                    page_content=x["text"],
                    metadata=md,
                ),
                x["score"],
            )
            for x, md in zip(responses, metadatas)
        ]

        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        lambda_val: float = 0.025,
        filter: Optional[str] = None,
        n_sentence_context: int = 2,
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
            n_sentence_context: number of sentences before/after the matching segment
                to add, defaults to 2

        Returns:
            List of Documents most similar to the query
        """
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            lambda_val=lambda_val,
            filter=filter,
            n_sentence_context=n_sentence_context,
            **kwargs,
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
                vectara = Vectara.from_texts(
                    texts,
                    vectara_customer_id=customer_id,
                    vectara_corpus_id=corpus_id,
                    vectara_api_key=api_key,
                )
        """
        # Note: Vectara generates its own embeddings, so we ignore the provided
        # embeddings (required by interface)
        doc_metadata = kwargs.pop("doc_metadata", {})
        vectara = cls(**kwargs)
        vectara.add_texts(texts, metadatas, doc_metadata=doc_metadata, **kwargs)
        return vectara

    @classmethod
    def from_files(
        cls: Type[Vectara],
        files: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Vectara:
        """Construct Vectara wrapper from raw documents.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python

                from langchain import Vectara
                vectara = Vectara.from_files(
                    files_list,
                    vectara_customer_id=customer_id,
                    vectara_corpus_id=corpus_id,
                    vectara_api_key=api_key,
                )
        """
        # Note: Vectara generates its own embeddings, so we ignore the provided
        # embeddings (required by interface)
        vectara = cls(**kwargs)
        vectara.add_files(files, metadatas)
        return vectara

    def as_retriever(self, **kwargs: Any) -> VectaraRetriever:
        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return VectaraRetriever(vectorstore=self, **kwargs, tags=tags)


class VectaraRetriever(VectorStoreRetriever):
    """Retriever class for `Vectara`."""

    vectorstore: Vectara
    """Vectara vectorstore."""
    search_kwargs: dict = Field(
        default_factory=lambda: {
            "lambda_val": 0.025,
            "k": 5,
            "filter": "",
            "n_sentence_context": "2",
        }
    )
    """Search params.
        k: Number of Documents to return. Defaults to 5.
        lambda_val: lexical match parameter for hybrid search.
        filter: Dictionary of argument(s) to filter on metadata. For example a
            filter can be "doc.rating > 3.0 and part.lang = 'deu'"} see
            https://docs.vectara.com/docs/search-apis/sql/filter-overview
            for more details.
        n_sentence_context: number of sentences before/after the matching segment to add
    """

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        doc_metadata: Optional[dict] = {},
    ) -> None:
        """Add text to the Vectara vectorstore.

        Args:
            texts (List[str]): The text
            metadatas (List[dict]): Metadata dicts, must line up with existing store
        """
        self.vectorstore.add_texts(texts, metadatas, doc_metadata)
