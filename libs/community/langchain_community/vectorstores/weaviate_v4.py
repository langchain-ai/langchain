from __future__ import annotations

import datetime
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from langchain.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from weaviate import classes as wvc

if TYPE_CHECKING:
    import weaviate


def _default_schema(index_name: str) -> Dict:
    return {
        "class": index_name,
        "properties": [wvc.Property(name="text", data_type=wvc.DataType.TEXT)],
    }


def _create_weaviate_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> weaviate.client.WeaviateClient:
    try:
        import weaviate
    except ImportError:
        raise ImportError(
            "Could not import weaviate python package."
            "Please install it with `pip install weaviate-client`"
        )
    url = url or os.environ.get("WEAVIATE_URL")
    api_key = api_key or os.environ.get("WEAVIATE_API_KEY")

    if api_key:
        return weaviate.connect_to_wcs(cluster_url=url, auth_credentials=api_key)

    return weaviate.connect_to_local(host=url)


def _default_score_normalizer(val: float) -> float:
    return 1 - 1 / (1 + np.exp(val))


def _json_serializable(value: Any) -> Any:
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    return value


class Weaviate(VectorStore):
    """`Weaviate` vector store.

    To use, you should have the ``weaviate-client`` python package installed.

    Example:
        .. code-block:: python

            import weaviate
            from langchain.vectorstores import Weaviate

            client = weaviate.Client(url=os.environ["WEAVIATE_URL"], ...)
            weaviate = Weaviate(client, index_name, text_key)

    """

    def __init__(
        self,
        client: Any,
        index_name: str,
        text_key: str,
        embedding: Optional[Embeddings] = None,
        attributes: Optional[List[str]] = None,
        relevance_score_fn: Optional[
            Callable[[float], float]
        ] = _default_score_normalizer,
        by_text: bool = True,
    ):
        """
        Initialize with Weaviate client.

        Args:
            client: Initialized Weaviate client
            index_name: Name of the Weaviate collection to use
            text_key: Name of the variable containing the text in Weaviate collection
            embedding: Text embedding model
            attributes: Extra variables to be returned from Weaviate collection when\
                searching
            relevance_score_fn: Function for converting whatever distance function the
                vector store uses to a relevance score, which is a normalized similarity
                score (0 means dissimilar, 1 means similar).
            by_text: Whether to search by text similarity or vector
        """
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "Could not import weaviate python package. "
                "Please install it with `pip install weaviate-client`."
            )
        if not isinstance(client, weaviate.client.WeaviateClient):
            raise ValueError(
                f"Expected weaviate.client.WeaviateClient, got {type(client)}"
            )
        self._client = client
        self._index_name = index_name
        self._collection = self._client.collections.get(self._index_name)
        self._embedding = embedding
        self._text_key = text_key
        self._query_attrs = [self._text_key]
        self.relevance_score_fn = relevance_score_fn
        self._by_text = by_text
        if attributes:
            self._query_attrs.extend(attributes)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return (
            self.relevance_score_fn
            if self.relevance_score_fn
            else _default_score_normalizer
        )

    def _objects_to_documents(self, objects: List[dict]) -> List[Document]:
        docs = []
        for obj in objects:
            contents = []
            for attr in self._query_attrs:
                if attr in obj.properties:
                    contents.append(f"{attr}={obj.properties.pop(attr)}")
            docs.append(
                Document(page_content=" ".join(contents), metadata=obj.properties)
            )

        return docs

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upload texts with metadata (properties) to Weaviate."""
        from weaviate.util import get_valid_uuid

        embeddings: Optional[List[List[float]]] = None
        if self._embedding:
            if not isinstance(texts, list):
                texts = list(texts)
            embeddings = self._embedding.embed_documents(texts)

        batch_data = []
        for i, text in enumerate(texts):
            data_properties = {self._text_key: text}
            if metadatas is not None:
                for key, val in metadatas[i].items():
                    data_properties[key] = _json_serializable(val)

            # Allow for ids (consistent w/ other methods)
            # # Or uuids (backwards compatible w/ existing arg)
            # If the UUID of one of the objects already exists
            # then the existing object will be replaced by the new object.
            _id = get_valid_uuid(uuid4())
            if "uuids" in kwargs:
                _id = kwargs["uuids"][i]
            elif "ids" in kwargs:
                _id = kwargs["ids"][i]

            batch_data.append(
                wvc.DataObject(
                    uuid=_id,
                    properties=data_properties,
                    vector=embeddings[i] if embeddings else None,
                )
            )

            collection = self._collection
            if kwargs.get("tenant"):
                collection = self._collection.with_tenant(kwargs.pop("tenant"))

            result = collection.data.insert_many(batch_data)

        return list(result.uuids.values)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        if self._by_text:
            return self.similarity_search_by_text(query, k, **kwargs)
        else:
            if self._embedding is None:
                raise ValueError(
                    "_embedding cannot be None for similarity_search when "
                    "_by_text=False"
                )
            embedding = self._embedding.embed_query(query)
            return self.similarity_search_by_vector(embedding, k, **kwargs)

    def similarity_search_by_text(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        collection = self._collection
        if kwargs.get("tenant"):
            collection = self._collection.with_tenant(kwargs.pop("tenant"))

        result = collection.query.near_text(query=query, limit=k, **kwargs)

        return self._objects_to_documents(result.objects)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Look up similar documents by embedding vector in Weaviate.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the embedding.
        """
        collection = self._collection
        if kwargs.get("tenant"):
            collection = self._collection.with_tenant(kwargs.pop("tenant"))

        result = collection.query.near_vector(near_vector=embedding, limit=k, **kwargs)

        return self._objects_to_documents(result.objects)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if self._embedding is not None:
            embedding = self._embedding.embed_query(query)
        else:
            raise ValueError(
                "max_marginal_relevance_search requires a suitable Embeddings object"
            )

        return self.max_marginal_relevance_search_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        collection = self._collection
        if kwargs.get("tenant"):
            collection = self._collection.with_tenant(kwargs.pop("tenant"))

        result = collection.query.near_vector(
            near_vector=embedding, limit=fetch_k, include_vector=True, **kwargs
        )

        embeddings = [obj.vector for obj in result.objects]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )

        return self._objects_to_documents([result.objects[idx] for idx in mmr_selected])

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Return list of documents most similar to the query
        text and cosine distance in float for each.
        Lower score represents more similarity.
        """
        if self._embedding is None:
            raise ValueError(
                "_embedding cannot be None for similarity_search_with_score"
            )

        collection = self._collection
        if kwargs.get("tenant"):
            collection = self._collection.with_tenant(kwargs.pop("tenant"))

        embedded_query = self._embedding.embed_query(query)

        if self._by_text:
            result = collection.query.near_text(
                query=query, limit=k, include_vector=True, **kwargs
            )
        else:
            result = collection.query.near_vector(
                near_vector=embedded_query, limit=k, include_vector=True, **kwargs
            )

        docs = self._objects_to_documents(result.objects)
        docs_and_scores = [
            (docs[i], np.dot(obj.vector, embedded_query))
            for i, obj in enumerate(result.objects)
        ]

        return docs_and_scores

    def bm25_search(
        self, query: str, query_properties: List[str] = None, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Keyword search, also called "BM25 (Best match 25)" or "sparse vector" search,
            returns objects that have the highest BM25F scores.

        Args:
            query: Text to look up documents similar to.
            query_properties: On which properties of the object to search.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        collection = self._collection
        if kwargs.get("tenant"):
            collection = self._collection.with_tenant(kwargs.pop("tenant"))

        result = collection.query.bm25(
            query=query, query_properties=query_properties, limit=k, **kwargs
        )

        return self._objects_to_documents(result.objects)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        client: Optional[weaviate.client.WeaviateClient] = None,
        weaviate_url: Optional[str] = None,
        weaviate_api_key: Optional[str] = None,
        batch_size: Optional[int] = None,
        index_name: Optional[str] = None,
        text_key: str = "text",
        by_text: bool = False,
        relevance_score_fn: Optional[
            Callable[[float], float]
        ] = _default_score_normalizer,
        **kwargs: Any,
    ) -> Weaviate:
        """Construct Weaviate wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in the Weaviate instance.
            3. Adds the documents to the newly created Weaviate index.

        This is intended to be a quick way to get started.

        Args:
            texts: Texts to add to vector store.
            embedding: Text embedding model to use.
            metadatas: Metadata associated with each text.
            client: weaviate.Client to use.
            weaviate_url: The Weaviate URL. If using Weaviate Cloud Services get it
                from the ``Details`` tab. Can be passed in as a named param or by
                setting the environment variable ``WEAVIATE_URL``. Should not be
                specified if client is provided.
            weaviate_api_key: The Weaviate API key. If enabled and using Weaviate Cloud
                Services, get it from ``Details`` tab. Can be passed in as a named param
                or by setting the environment variable ``WEAVIATE_API_KEY``. Should
                not be specified if client is provided.
            batch_size: Size of batch operations.
            index_name: Index name.
            text_key: Key to use for uploading/retrieving text to/from vectorstore.
            by_text: Whether to search by text or by embedding.
            relevance_score_fn: Function for converting whatever distance function the
                vector store uses to a relevance score, which is a normalized similarity
                score (0 means dissimilar, 1 means similar).
            **kwargs: Additional named parameters to pass to ``Weaviate.__init__()``.

        Example:
            .. code-block:: python

                from langchain.embeddings import OpenAIEmbeddings
                from langchain.vectorstores import Weaviate

                embeddings = OpenAIEmbeddings()
                weaviate = Weaviate.from_texts(
                    texts,
                    embeddings,
                    weaviate_url="http://localhost:8080"
                )
        """

        try:
            from weaviate.util import get_valid_uuid
        except ImportError as e:
            raise ImportError(
                "Could not import weaviate python  package. "
                "Please install it with `pip install weaviate-client`"
            ) from e

        client = client or _create_weaviate_client(
            url=weaviate_url,
            api_key=weaviate_api_key,
        )

        index_name = index_name or f"LangChain_{uuid4().hex}"
        schema = _default_schema(index_name)
        # check whether the index already exists
        if not client.collections.exists(index_name):
            collection = client.collections.create(
                name=schema["class"],
                properties=schema["properties"],
            )

        embeddings = embedding.embed_documents(texts) if embedding else None
        attributes = list(metadatas[0].keys()) if metadatas else None

        # If the UUID of one of the objects already exists
        # then the existing object will be replaced by the new object.
        if "uuids" in kwargs:
            uuids = kwargs.pop("uuids")
        else:
            uuids = [get_valid_uuid(uuid4()) for _ in range(len(texts))]

        batch_data = []
        for i, text in enumerate(texts):
            _id = uuids[i]
            data_properties = {
                text_key: text,
            }
            if metadatas is not None:
                for key in metadatas[i].keys():
                    data_properties[key] = metadatas[i][key]

            batch_data.append(
                wvc.DataObject(
                    uuid=_id,
                    properties=data_properties,
                    vector=embeddings[i] if embeddings else None,
                )
            )

        if batch_size and batch_size < len(batch_data):
            chunks = (
                batch_data[i : i + batch_size]
                for i in range(0, len(batch_data), batch_size)
            )
            for chunk in chunks:
                collection.data.insert_many(chunk)
        else:
            collection.data.insert_many(batch_data)

        return cls(
            client,
            index_name,
            text_key,
            embedding=embedding,
            attributes=attributes,
            relevance_score_fn=relevance_score_fn,
            by_text=by_text,
            **kwargs,
        )

    def delete(self, ids: Optional[List[UUID]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """

        if not ids:
            raise ValueError("No ids provided to delete.")

        collection = self._collection
        if kwargs.get("tenant"):
            collection = self._collection.with_tenant(kwargs.pop("tenant"))

        collection.data.delete_many(where=wvc.Filter("id").contains_any(ids))
