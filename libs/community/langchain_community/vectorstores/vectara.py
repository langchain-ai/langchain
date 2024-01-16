from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from hashlib import md5
from typing import Any, Iterable, List, Optional, Tuple, Type

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig:
    """
    is_enabled: True if summary is enabled, False otherwise
    max_results: maximum number of results to summarize
    response_lang: requested language for the summary
    """

    is_enabled: bool = False
    max_results: int = 7
    response_lang: str = "eng"


@dataclass
class MMRConfig:
    """
    is_enabled: True if MMR is enabled, False otherwise
    mmr_k: number of results to fetch for MMR, defaults to 50
    diversity_bias: number between 0 and 1 that determines the degree
        of diversity among the results with 0 corresponding
        to minimum diversity and 1 to maximum diversity.
        Defaults to 0.3.
        Note: diversity_bias is equivalent 1-lambda_mult
        where lambda_mult is the value often used in max_marginal_relevance_search()
        We chose to use that since we believe it's more intuitive to the user.
    """

    is_enabled: bool = False
    mmr_k: int = 50
    diversity_bias: float = 0.3


@dataclass
class VectaraQueryConfig:
    """
    k: Number of Documents to return. Defaults to 10.
    lambda_val: lexical match parameter for hybrid search.
    filter Dictionary of argument(s) to filter on metadata. For example a
        filter can be "doc.rating > 3.0 and part.lang = 'deu'"} see
        https://docs.vectara.com/docs/search-apis/sql/filter-overview
        for more details.
    score_threshold: minimal score threshold for the result.
        If defined, results with score less than this value will be
        filtered out.
    n_sentence_context: number of sentences before/after the matching segment
        to add, defaults to 2
    mmr_config: MMRConfig configuration dataclass
    summary_config: SummaryConfig configuration dataclass
    """

    k: int = 10
    lambda_val: float = 0.0
    filter: str = ""
    score_threshold: Optional[float] = None
    n_sentence_context: int = 2
    mmr_config: MMRConfig = field(default_factory=MMRConfig)
    summary_config: SummaryConfig = field(default_factory=SummaryConfig)


class Vectara(VectorStore):
    """`Vectara API` vector store.

     See (https://vectara.com).

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import Vectara

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
        vectara_api_timeout: int = 120,
        source: str = "langchain",
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
        self._source = source

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
            "X-Source": self._source,
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

    def _index_doc(self, doc: dict, use_core_api: bool = False) -> str:
        request: dict[str, Any] = {}
        request["customer_id"] = self._vectara_customer_id
        request["corpus_id"] = self._vectara_corpus_id
        request["document"] = doc

        api_endpoint = (
            "https://api.vectara.io/v1/core/index"
            if use_core_api
            else "https://api.vectara.io/v1/index"
        )
        response = self._session.post(
            headers=self._get_post_headers(),
            url=api_endpoint,
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

        use_core_api = kwargs.get("use_core_api", False)
        section_key = "parts" if use_core_api else "section"
        doc = {
            "document_id": doc_id,
            "metadataJson": json.dumps(doc_metadata),
            section_key: [
                {"text": text, "metadataJson": json.dumps(md)}
                for text, md in zip(texts, metadatas)
            ],
        }

        success_str = self._index_doc(doc, use_core_api=use_core_api)

        if success_str == "E_ALREADY_EXISTS":
            self._delete_doc(doc_id)
            self._index_doc(doc)
        elif success_str == "E_NO_PERMISSIONS":
            print(
                """No permissions to add document to Vectara. 
                Check your corpus ID, customer ID and API key"""
            )
        return [doc_id]

    def vectara_query(
        self,
        query: str,
        config: VectaraQueryConfig,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run a Vectara query

        Args:
            query: Text to look up documents similar to.
            config: VectaraQueryConfig object
        Returns:
            A list of k Documents matching the given query
            If summary is enabled, last document is the summary text with 'summary'=True
        """
        if isinstance(config.mmr_config, dict):
            config.mmr_config = MMRConfig(**config.mmr_config)
        if isinstance(config.summary_config, dict):
            config.summary_config = SummaryConfig(**config.summary_config)

        data = {
            "query": [
                {
                    "query": query,
                    "start": 0,
                    "numResults": config.mmr_config.mmr_k
                    if config.mmr_config.is_enabled
                    else config.k,
                    "contextConfig": {
                        "sentencesBefore": config.n_sentence_context,
                        "sentencesAfter": config.n_sentence_context,
                    },
                    "corpusKey": [
                        {
                            "customerId": self._vectara_customer_id,
                            "corpusId": self._vectara_corpus_id,
                            "metadataFilter": config.filter,
                            "lexicalInterpolationConfig": {"lambda": config.lambda_val},
                        }
                    ],
                }
            ]
        }
        if config.mmr_config.is_enabled:
            data["query"][0]["rerankingConfig"] = {
                "rerankerId": 272725718,
                "mmrConfig": {"diversityBias": config.mmr_config.diversity_bias},
            }
        if config.summary_config.is_enabled:
            data["query"][0]["summary"] = [
                {
                    "maxSummarizedResults": config.summary_config.max_results,
                    "responseLang": config.summary_config.response_lang,
                }
            ]

        response = self._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/query",
            data=json.dumps(data),
            timeout=self.vectara_api_timeout,
        )

        if response.status_code != 200:
            logger.error(
                "Query failed %s",
                f"(code {response.status_code}, reason {response.reason}, details "
                f"{response.text})",
            )
            return [], ""

        result = response.json()

        if config.score_threshold:
            responses = [
                r
                for r in result["responseSet"][0]["response"]
                if r["score"] > config.score_threshold
            ]
        else:
            responses = result["responseSet"][0]["response"]
        documents = result["responseSet"][0]["document"]

        metadatas = []
        for x in responses:
            md = {m["name"]: m["value"] for m in x["metadata"]}
            doc_num = x["documentIndex"]
            doc_md = {m["name"]: m["value"] for m in documents[doc_num]["metadata"]}
            if "source" not in doc_md:
                doc_md["source"] = "vectara"
            md.update(doc_md)
            metadatas.append(md)

        res = [
            (
                Document(
                    page_content=x["text"],
                    metadata=md,
                ),
                x["score"],
            )
            for x, md in zip(responses, metadatas)
        ]

        if config.mmr_config.is_enabled:
            res = res[: config.k]
        if config.summary_config.is_enabled:
            summary = result["responseSet"][0]["summary"][0]["text"]
            res.append(
                (Document(page_content=summary, metadata={"summary": True}), 0.0)
            )

        return res

    def similarity_search_with_score(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return Vectara documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 10.
            any other querying variable in VectaraQueryConfig like:
            - lambda_val: lexical match parameter for hybrid search.
            - filter: filter string
            - score_threshold: minimal score threshold for the result.
            - n_sentence_context: number of sentences before/after the matching segment
            - mmr_config: optional configuration for MMR (see MMRConfig dataclass)
            - summary_config: optional configuration for summary
              (see SummaryConfig dataclass)
        Returns:
            List of Documents most similar to the query and score for each.
        """
        config = VectaraQueryConfig(**kwargs)
        docs = self.vectara_query(query, config)
        return docs

    def similarity_search(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Document]:
        """Return Vectara documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            any other querying variable in VectaraQueryConfig

        Returns:
            List of Documents most similar to the query
        """
        docs_and_scores = self.similarity_search_with_score(
            query,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        fetch_k: int = 50,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 5.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 50
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            kwargs: any other querying variable in VectaraQueryConfig
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        kwargs["mmr_config"] = MMRConfig(
            is_enabled=True, mmr_k=fetch_k, diversity_bias=1 - lambda_mult
        )
        return self.similarity_search(query, **kwargs)

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

                from langchain_community.vectorstores import Vectara
                vectara = Vectara.from_texts(
                    texts,
                    vectara_customer_id=customer_id,
                    vectara_corpus_id=corpus_id,
                    vectara_api_key=api_key,
                )
        """
        # Notes:
        # * Vectara generates its own embeddings, so we ignore the provided
        #   embeddings (required by interface)
        # * when metadatas[] are provided they are associated with each "part"
        #   in Vectara. doc_metadata can be used to provide additional metadata
        #   for the document itself (applies to all "texts" in this call)
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

                from langchain_community.vectorstores import Vectara
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


class VectaraRetriever(VectorStoreRetriever):
    """Retriever class for `Vectara`."""

    vectorstore: Vectara
    """Vectara vectorstore."""
    search_kwargs: dict = Field(
        default_factory=lambda: {
            "lambda_val": 0.0,
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
        doc_metadata: Optional[dict] = None,
    ) -> None:
        """Add text to the Vectara vectorstore.

        Args:
            texts (List[str]): The text
            metadatas (List[dict]): Metadata dicts, must line up with existing store
        """
        self.vectorstore.add_texts(texts, metadatas, doc_metadata or {})
