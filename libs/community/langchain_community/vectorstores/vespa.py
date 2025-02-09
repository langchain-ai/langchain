from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever


class VespaStore(VectorStore):
    """
    `Vespa` vector store.

    To use, you should have the python client library ``pyvespa`` installed.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import VespaStore
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            from vespa.application import Vespa

            # Create a vespa client dependent upon your application,
            # e.g. either connecting to Vespa Cloud or a local deployment
            # such as Docker. Please refer to the PyVespa documentation on
            # how to initialize the client.

            vespa_app = Vespa(url="...", port=..., application_package=...)

            # You need to instruct LangChain on which fields to use for embeddings
            vespa_config = dict(
                page_content_field="text",
                embedding_field="embedding",
                input_field="query_embedding",
                metadata_fields=["date", "rating", "author"]
            )

            embedding_function = OpenAIEmbeddings()
            vectorstore = VespaStore(vespa_app, embedding_function, **vespa_config)

    """

    def __init__(
        self,
        app: Any,
        embedding_function: Optional[Embeddings] = None,
        page_content_field: Optional[str] = None,
        embedding_field: Optional[str] = None,
        input_field: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize with a PyVespa client.
        """
        try:
            from vespa.application import Vespa
        except ImportError:
            raise ImportError(
                "Could not import Vespa python package. "
                "Please install it with `pip install pyvespa`."
            )
        if not isinstance(app, Vespa):
            raise ValueError(
                f"app should be an instance of vespa.application.Vespa, got {type(app)}"
            )

        self._vespa_app = app
        self._embedding_function = embedding_function
        self._page_content_field = page_content_field
        self._embedding_field = embedding_field
        self._input_field = input_field
        self._metadata_fields = metadata_fields

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        embeddings = None
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(list(texts))

        if ids is None:
            ids = [str(f"{i + 1}") for i, _ in enumerate(texts)]

        batch = []
        for i, text in enumerate(texts):
            fields: Dict[str, Union[str, List[float]]] = {}
            if self._page_content_field is not None:
                fields[self._page_content_field] = text
            if self._embedding_field is not None and embeddings is not None:
                fields[self._embedding_field] = embeddings[i]
            if metadatas is not None and self._metadata_fields is not None:
                for metadata_field in self._metadata_fields:
                    if metadata_field in metadatas[i]:
                        fields[metadata_field] = metadatas[i][metadata_field]
            batch.append({"id": ids[i], "fields": fields})

        results = self._vespa_app.feed_batch(batch)
        for result in results:
            if not (str(result.status_code).startswith("2")):
                raise RuntimeError(
                    f"Could not add document to Vespa. "
                    f"Error code: {result.status_code}. "
                    f"Message: {result.json['message']}"
                )
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            return False
        batch = [{"id": id} for id in ids]
        result = self._vespa_app.delete_batch(batch)
        return sum([0 if r.status_code == 200 else 1 for r in result]) == 0

    def _create_query(
        self, query_embedding: List[float], k: int = 4, **kwargs: Any
    ) -> Dict:
        hits = k
        doc_embedding_field = self._embedding_field
        input_embedding_field = self._input_field
        ranking_function = kwargs["ranking"] if "ranking" in kwargs else "default"
        filter = kwargs["filter"] if "filter" in kwargs else None

        approximate = kwargs["approximate"] if "approximate" in kwargs else False
        approximate = "true" if approximate else "false"

        yql = "select * from sources * where "
        yql += f"{{targetHits: {hits}, approximate: {approximate}}}"
        yql += f"nearestNeighbor({doc_embedding_field}, {input_embedding_field})"
        if filter is not None:
            yql += f" and {filter}"

        query = {
            "yql": yql,
            f"input.query({input_embedding_field})": query_embedding,
            "ranking": ranking_function,
            "hits": hits,
        }
        return query

    def similarity_search_by_vector_with_score(
        self, query_embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Performs similarity search from a embeddings vector.

        Args:
            query_embedding: Embeddings vector to search for.
            k: Number of results to return.
            custom_query: Use this custom query instead default query (kwargs)
            kwargs: other vector store specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if "custom_query" in kwargs:
            query = kwargs["custom_query"]
        else:
            query = self._create_query(query_embedding, k, **kwargs)

        try:
            response = self._vespa_app.query(body=query)
        except Exception as e:
            raise RuntimeError(
                f"Could not retrieve data from Vespa: "
                f"{e.args[0][0]['summary']}. "
                f"Error: {e.args[0][0]['message']}"
            )
        if not str(response.status_code).startswith("2"):
            raise RuntimeError(
                f"Could not retrieve data from Vespa. "
                f"Error code: {response.status_code}. "
                f"Message: {response.json['message']}"
            )

        root = response.json["root"]
        if "errors" in root:
            import json

            raise RuntimeError(json.dumps(root["errors"]))

        if response is None or response.hits is None:
            return []

        docs = []
        for child in response.hits:
            page_content = child["fields"][self._page_content_field]
            score = child["relevance"]
            metadata = {"id": child["id"]}
            if self._metadata_fields is not None:
                for field in self._metadata_fields:
                    metadata[field] = child["fields"].get(field)
            doc = Document(page_content=page_content, metadata=metadata)
            docs.append((doc, score))
        return docs

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        results = self.similarity_search_by_vector_with_score(embedding, k, **kwargs)
        return [r[0] for r in results]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        query_emb = []
        if self._embedding_function is not None:
            query_emb = self._embedding_function.embed_query(query)
        return self.similarity_search_by_vector_with_score(query_emb, k, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        results = self.similarity_search_with_score(query, k, **kwargs)
        return [r[0] for r in results]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError("MMR search not implemented")

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError("MMR search by vector not implemented")

    @classmethod
    def from_texts(
        cls: Type[VespaStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> VespaStore:
        vespa = cls(embedding_function=embedding, **kwargs)
        vespa.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vespa

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        return super().as_retriever(**kwargs)
