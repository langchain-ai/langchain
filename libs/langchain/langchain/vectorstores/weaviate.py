from __future__ import annotations

import datetime
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)
from uuid import uuid4

import numpy as np

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    import weaviate


def _default_schema(index_name: str) -> Dict:
    return {
        "class": index_name,
        "properties": [
            {
                "name": "text",
                "dataType": ["text"],
            }
        ],
    }


def _create_weaviate_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> weaviate.Client:
    try:
        import weaviate
    except ImportError:
        raise ImportError(
            "Could not import weaviate python  package. "
            "Please install it with `pip install weaviate-client`"
        )
    url = url or os.environ.get("WEAVIATE_URL")
    api_key = api_key or os.environ.get("WEAVIATE_API_KEY")
    auth = weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
    return weaviate.Client(url=url, auth_client_secret=auth, **kwargs)


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
        """Initialize with Weaviate client."""
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "Could not import weaviate python package. "
                "Please install it with `pip install weaviate-client`."
            )
        if not isinstance(client, weaviate.Client):
            raise ValueError(
                f"client should be an instance of weaviate.Client, got {type(client)}"
            )
        self._client = client
        self._index_name = index_name
        self._embedding = embedding
        self._text_key = text_key
        self._query_attrs = [self._text_key]
        self.relevance_score_fn = relevance_score_fn
        self._by_text = by_text
        if attributes is not None:
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

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upload texts with metadata (properties) to Weaviate."""
        from weaviate.util import get_valid_uuid

        ids = []
        embeddings: Optional[List[List[float]]] = None
        if self._embedding:
            if not isinstance(texts, list):
                texts = list(texts)
            embeddings = self._embedding.embed_documents(texts)

        with self._client.batch as batch:
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

                batch.add_data_object(
                    data_object=data_properties,
                    class_name=self._index_name,
                    uuid=_id,
                    vector=embeddings[i] if embeddings else None,
                )
                ids.append(_id)
        return ids

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
        content: Dict[str, Any] = {"concepts": [query]}
        if kwargs.get("search_distance"):
            content["certainty"] = kwargs.get("search_distance")
        query_obj = self._client.query.get(self._index_name, self._query_attrs)
        if kwargs.get("where_filter"):
            query_obj = query_obj.with_where(kwargs.get("where_filter"))
        if kwargs.get("additional"):
            query_obj = query_obj.with_additional(kwargs.get("additional"))
        result = query_obj.with_near_text(content).with_limit(k).do()
        if "errors" in result:
            raise ValueError(f"Error during query: {result['errors']}")
        docs = []
        for res in result["data"]["Get"][self._index_name]:
            text = res.pop(self._text_key)
            docs.append(Document(page_content=text, metadata=res))
        return docs

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Look up similar documents by embedding vector in Weaviate."""
        vector = {"vector": embedding}
        query_obj = self._client.query.get(self._index_name, self._query_attrs)
        if kwargs.get("where_filter"):
            query_obj = query_obj.with_where(kwargs.get("where_filter"))
        if kwargs.get("additional"):
            query_obj = query_obj.with_additional(kwargs.get("additional"))
        result = query_obj.with_near_vector(vector).with_limit(k).do()
        if "errors" in result:
            raise ValueError(f"Error during query: {result['errors']}")
        docs = []
        for res in result["data"]["Get"][self._index_name]:
            text = res.pop(self._text_key)
            docs.append(Document(page_content=text, metadata=res))
        return docs

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
        vector = {"vector": embedding}
        query_obj = self._client.query.get(self._index_name, self._query_attrs)
        if kwargs.get("where_filter"):
            query_obj = query_obj.with_where(kwargs.get("where_filter"))
        results = (
            query_obj.with_additional("vector")
            .with_near_vector(vector)
            .with_limit(fetch_k)
            .do()
        )

        payload = results["data"]["Get"][self._index_name]
        embeddings = [result["_additional"]["vector"] for result in payload]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )

        docs = []
        for idx in mmr_selected:
            text = payload[idx].pop(self._text_key)
            payload[idx].pop("_additional")
            meta = payload[idx]
            docs.append(Document(page_content=text, metadata=meta))
        return docs

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
        content: Dict[str, Any] = {"concepts": [query]}
        if kwargs.get("search_distance"):
            content["certainty"] = kwargs.get("search_distance")
        query_obj = self._client.query.get(self._index_name, self._query_attrs)
        if kwargs.get("where_filter"):
            query_obj = query_obj.with_where(kwargs.get("where_filter"))

        embedded_query = self._embedding.embed_query(query)
        if not self._by_text:
            vector = {"vector": embedded_query}
            result = (
                query_obj.with_near_vector(vector)
                .with_limit(k)
                .with_additional("vector")
                .do()
            )
        else:
            result = (
                query_obj.with_near_text(content)
                .with_limit(k)
                .with_additional("vector")
                .do()
            )

        if "errors" in result:
            raise ValueError(f"Error during query: {result['errors']}")

        docs_and_scores = []
        for res in result["data"]["Get"][self._index_name]:
            text = res.pop(self._text_key)
            score = np.dot(res["_additional"]["vector"], embedded_query)
            docs_and_scores.append((Document(page_content=text, metadata=res), score))
        return docs_and_scores

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        client: Optional[weaviate.Client] = None,
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
        if batch_size:
            client.batch.configure(batch_size=batch_size)

        index_name = index_name or f"LangChain_{uuid4().hex}"
        schema = _default_schema(index_name)
        # check whether the index already exists
        if not client.schema.contains(schema):
            client.schema.create_class(schema)

        embeddings = embedding.embed_documents(texts) if embedding else None
        attributes = list(metadatas[0].keys()) if metadatas else None

        with client.batch as batch:
            for i, text in enumerate(texts):
                data_properties = {
                    text_key: text,
                }
                if metadatas is not None:
                    for key in metadatas[i].keys():
                        data_properties[key] = metadatas[i][key]

                # If the UUID of one of the objects already exists
                # then the existing objectwill be replaced by the new object.
                if "uuids" in kwargs:
                    _id = kwargs["uuids"][i]
                else:
                    _id = get_valid_uuid(uuid4())

                # if an embedding strategy is not provided, we let
                # weaviate create the embedding. Note that this will only
                # work if weaviate has been installed with a vectorizer module
                # like text2vec-contextionary for example
                params = {
                    "uuid": _id,
                    "data_object": data_properties,
                    "class_name": index_name,
                }
                if embeddings is not None:
                    params["vector"] = embeddings[i]

                batch.add_data_object(**params)

            batch.flush()

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

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """

        if ids is None:
            raise ValueError("No ids provided to delete.")

        # TODO: Check if this can be done in bulk
        for id in ids:
            self._client.data_object.delete(uuid=id)
