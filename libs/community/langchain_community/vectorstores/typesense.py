from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from typesense.client import Client
    from typesense.collection import Collection


class Typesense(VectorStore):
    """`Typesense` vector store.

    To use, you should have the ``typesense`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embedding.openai import OpenAIEmbeddings
            from langchain_community.vectorstores import Typesense
            import typesense

            node = {
                "host": "localhost",  # For Typesense Cloud use xxx.a1.typesense.net
                "port": "8108",       # For Typesense Cloud use 443
                "protocol": "http"    # For Typesense Cloud use https
            }
            typesense_client = typesense.Client(
                {
                  "nodes": [node],
                  "api_key": "<API_KEY>",
                  "connection_timeout_seconds": 2
                }
            )
            typesense_collection_name = "langchain-memory"

            embedding = OpenAIEmbeddings()
            vectorstore = Typesense(
                typesense_client=typesense_client,
                embedding=embedding,
                typesense_collection_name=typesense_collection_name,
                text_key="text",
            )
    """

    def __init__(
        self,
        typesense_client: Client,
        embedding: Embeddings,
        *,
        typesense_collection_name: Optional[str] = None,
        text_key: str = "text",
    ):
        """Initialize with Typesense client."""
        try:
            from typesense import Client
        except ImportError:
            raise ImportError(
                "Could not import typesense python package. "
                "Please install it with `pip install typesense`."
            )
        if not isinstance(typesense_client, Client):
            raise ValueError(
                f"typesense_client should be an instance of typesense.Client, "
                f"got {type(typesense_client)}"
            )
        self._typesense_client = typesense_client
        self._embedding = embedding
        self._typesense_collection_name = (
            typesense_collection_name or f"langchain-{str(uuid.uuid4())}"
        )
        self._text_key = text_key

    @property
    def _collection(self) -> Collection:
        return self._typesense_client.collections[self._typesense_collection_name]

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _prep_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]],
        ids: Optional[List[str]],
    ) -> List[dict]:
        """Embed and create the documents"""
        _ids = ids or (str(uuid.uuid4()) for _ in texts)
        _metadatas: Iterable[dict] = metadatas or ({} for _ in texts)
        embedded_texts = self._embedding.embed_documents(list(texts))
        return [
            {"id": _id, "vec": vec, f"{self._text_key}": text, "metadata": metadata}
            for _id, vec, text, metadata in zip(_ids, embedded_texts, texts, _metadatas)
        ]

    def _create_collection(self, num_dim: int) -> None:
        fields = [
            {"name": "vec", "type": "float[]", "num_dim": num_dim},
            {"name": f"{self._text_key}", "type": "string"},
            {"name": ".*", "type": "auto"},
        ]
        self._typesense_client.collections.create(
            {"name": self._typesense_collection_name, "fields": fields}
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embedding and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        from typesense.exceptions import ObjectNotFound

        docs = self._prep_texts(texts, metadatas, ids)
        try:
            self._collection.documents.import_(docs, {"action": "upsert"})
        except ObjectNotFound:
            # Create the collection if it doesn't already exist
            self._create_collection(len(docs[0]["vec"]))
            self._collection.documents.import_(docs, {"action": "upsert"})
        return [doc["id"] for doc in docs]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        filter: Optional[str] = "",
    ) -> List[Tuple[Document, float]]:
        """Return typesense documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 10.
                Minimum 10 results would be returned.
            filter: typesense filter_by expression to filter documents on

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedded_query = [str(x) for x in self._embedding.embed_query(query)]
        query_obj = {
            "q": "*",
            "vector_query": f'vec:([{",".join(embedded_query)}], k:{k})',
            "filter_by": filter,
            "collection": self._typesense_collection_name,
        }
        docs = []
        response = self._typesense_client.multi_search.perform(
            {"searches": [query_obj]}, {}
        )
        for hit in response["results"][0]["hits"]:
            document = hit["document"]
            metadata = document["metadata"]
            text = document[self._text_key]
            score = hit["vector_distance"]
            docs.append((Document(page_content=text, metadata=metadata), score))
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[str] = "",
        **kwargs: Any,
    ) -> List[Document]:
        """Return typesense documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 10.
                Minimum 10 results would be returned.
            filter: typesense filter_by expression to filter documents on

        Returns:
            List of Documents most similar to the query and score for each
        """
        docs_and_score = self.similarity_search_with_score(query, k=k, filter=filter)
        return [doc for doc, _ in docs_and_score]

    @classmethod
    def from_client_params(
        cls,
        embedding: Embeddings,
        *,
        host: str = "localhost",
        port: Union[str, int] = "8108",
        protocol: str = "http",
        typesense_api_key: Optional[str] = None,
        connection_timeout_seconds: int = 2,
        **kwargs: Any,
    ) -> Typesense:
        """Initialize Typesense directly from client parameters.

        Example:
            .. code-block:: python

                from langchain_community.embedding.openai import OpenAIEmbeddings
                from langchain_community.vectorstores import Typesense

                # Pass in typesense_api_key as kwarg or set env var "TYPESENSE_API_KEY".
                vectorstore = Typesense(
                    OpenAIEmbeddings(),
                    host="localhost",
                    port="8108",
                    protocol="http",
                    typesense_collection_name="langchain-memory",
                )
        """
        try:
            from typesense import Client
        except ImportError:
            raise ImportError(
                "Could not import typesense python package. "
                "Please install it with `pip install typesense`."
            )

        node = {
            "host": host,
            "port": str(port),
            "protocol": protocol,
        }
        typesense_api_key = typesense_api_key or get_from_env(
            "typesense_api_key", "TYPESENSE_API_KEY"
        )
        client_config = {
            "nodes": [node],
            "api_key": typesense_api_key,
            "connection_timeout_seconds": connection_timeout_seconds,
        }
        return cls(Client(client_config), embedding, **kwargs)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        typesense_client: Optional[Client] = None,
        typesense_client_params: Optional[dict] = None,
        typesense_collection_name: Optional[str] = None,
        text_key: str = "text",
        **kwargs: Any,
    ) -> Typesense:
        """Construct Typesense wrapper from raw text."""
        if typesense_client:
            vectorstore = cls(typesense_client, embedding, **kwargs)
        elif typesense_client_params:
            vectorstore = cls.from_client_params(
                embedding, **typesense_client_params, **kwargs
            )
        else:
            raise ValueError(
                "Must specify one of typesense_client or typesense_client_params."
            )
        vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)
        return vectorstore
