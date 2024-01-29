"""Wrapper around LLMRails vector database."""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Iterable, List, Optional, Tuple

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever


class LLMRails(VectorStore):
    """Implementation of Vector Store using LLMRails.

     See https://llmrails.com/

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import LLMRails

            vectorstore = LLMRails(
                api_key=llm_rails_api_key,
                datastore_id=datastore_id
            )
    """

    def __init__(
        self,
        datastore_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize with LLMRails API."""
        self._datastore_id = datastore_id or os.environ.get("LLM_RAILS_DATASTORE_ID")
        self._api_key = api_key or os.environ.get("LLM_RAILS_API_KEY")
        if self._api_key is None:
            logging.warning("Can't find Rails credentials in environment.")

        self._session = requests.Session()  # to reuse connections
        self.datastore_id = datastore_id
        self.base_url = "https://api.llmrails.com/v1"

    def _get_post_headers(self) -> dict:
        """Returns headers that should be attached to each post request."""
        return {"X-API-KEY": self._api_key}

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        names: List[str] = []
        for text in texts:
            doc_name = str(uuid.uuid4())
            response = self._session.post(
                f"{self.base_url}/datastores/{self._datastore_id}/text",
                json={"name": doc_name, "text": text},
                verify=True,
                headers=self._get_post_headers(),
            )

            if response.status_code != 200:
                logging.error(
                    f"Create request failed for doc_name = {doc_name} with status code "
                    f"{response.status_code}, reason {response.reason}, text "
                    f"{response.text}"
                )

                return names

            names.append(doc_name)

        return names

    def add_files(
        self,
        files_list: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> bool:
        """
        LLMRails provides a way to add documents directly via our API where
        pre-processing and chunking occurs internally in an optimal way
        This method provides a way to use that API in LangChain

        Args:
            files_list: Iterable of strings, each representing a local file path.
                    Files could be text, HTML, PDF, markdown, doc/docx, ppt/pptx, etc.
                    see API docs for full list

        Returns:
            List of ids associated with each of the files indexed
        """
        files = []

        for file in files_list:
            if not os.path.exists(file):
                logging.error(f"File {file} does not exist, skipping")
                continue

            files.append(("file", (os.path.basename(file), open(file, "rb"))))

        response = self._session.post(
            f"{self.base_url}/datastores/{self._datastore_id}/file",
            files=files,
            verify=True,
            headers=self._get_post_headers(),
        )

        if response.status_code != 200:
            logging.error(
                f"Create request failed for datastore = {self._datastore_id} "
                f"with status code {response.status_code}, reason {response.reason}, "
                f"text {response.text}"
            )

            return False

        return True

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Return LLMRails documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 5 Max 10.
            alpha: parameter for hybrid search .

        Returns:
            List of Documents most similar to the query and score for each.
        """
        response = self._session.post(
            headers=self._get_post_headers(),
            url=f"{self.base_url}/datastores/{self._datastore_id}/search",
            data=json.dumps({"k": k, "text": query}),
            timeout=10,
        )

        if response.status_code != 200:
            logging.error(
                "Query failed %s",
                f"(code {response.status_code}, reason {response.reason}, details "
                f"{response.text})",
            )
            return []

        results = response.json()["results"]
        docs = [
            (
                Document(
                    page_content=x["text"],
                    metadata={
                        key: value
                        for key, value in x["metadata"].items()
                        if key != "score"
                    },
                ),
                x["metadata"]["score"],
            )
            for x in results
        ]

        return docs

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return LLMRails documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 5.

        Returns:
            List of Documents most similar to the query
        """
        docs_and_scores = self.similarity_search_with_score(query, k=k)

        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> LLMRails:
        """Construct LLMRails wrapper from raw documents.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python

                from langchain_community.vectorstores import LLMRails
                llm_rails = LLMRails.from_texts(
                    texts,
                    datastore_id=datastore_id,
                    api_key=llm_rails_api_key
                )
        """
        # Note: LLMRails generates its own embeddings, so we ignore the provided
        # embeddings (required by interface)
        llm_rails = cls(**kwargs)
        llm_rails.add_texts(texts)
        return llm_rails

    def as_retriever(self, **kwargs: Any) -> LLMRailsRetriever:
        return LLMRailsRetriever(vectorstore=self, **kwargs)


class LLMRailsRetriever(VectorStoreRetriever):
    """Retriever for LLMRails."""

    vectorstore: LLMRails
    search_kwargs: dict = Field(default_factory=lambda: {"k": 5})
    """Search params.
        k: Number of Documents to return. Defaults to 5.
        alpha: parameter for hybrid search .
    """

    def add_texts(self, texts: List[str]) -> None:
        """Add text to the datastore.

        Args:
            texts (List[str]): The text
        """
        self.vectorstore.add_texts(texts)
