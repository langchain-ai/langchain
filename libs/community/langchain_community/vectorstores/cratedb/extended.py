import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

import sqlalchemy
from langchain.schema.embeddings import Embeddings

from langchain_community.vectorstores.cratedb.base import (
    DEFAULT_DISTANCE_STRATEGY,
    CrateDBVectorStore,
    DistanceStrategy,
)
from langchain_community.vectorstores.pgvector import _LANGCHAIN_DEFAULT_COLLECTION_NAME


class CrateDBVectorStoreMultiCollection(CrateDBVectorStore):
    """
    Provide functionality for searching multiple collections.
    It can not be used for indexing documents.

    To use it, you should have the ``sqlalchemy-cratedb`` Python package installed.

    Synopsis::

        from langchain_community.vectorstores.cratedb import CrateDBVectorStoreMultiCollection

        multisearch = CrateDBVectorStoreMultiCollection(
            collection_names=["collection_foo", "collection_bar"],
            embedding_function=embeddings,
            connection_string=CONNECTION_STRING,
        )
        docs_with_score = multisearch.similarity_search_with_score(query)
    """  # noqa: E501

    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        collection_names: List[str] = [_LANGCHAIN_DEFAULT_COLLECTION_NAME],
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,  # type: ignore[arg-type]
        logger: Optional[logging.Logger] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        *,
        connection: Optional[sqlalchemy.engine.Connection] = None,
        engine_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.collection_names = collection_names
        self._distance_strategy = distance_strategy  # type: ignore[assignment]
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self.engine_args = engine_args or {}

        # Create a connection if not provided, otherwise use the provided connection
        self._bind = connection if connection else self._create_engine()

        self.__post_init__()

    @classmethod
    def _from(cls, *args: List, **kwargs: Dict):  # type: ignore[no-untyped-def,override]
        raise NotImplementedError("This adapter can not be used for indexing documents")

    def get_collections(self, session: sqlalchemy.orm.Session) -> Any:
        if self.CollectionStore is None:
            raise RuntimeError(
                "Collection can't be accessed without specifying "
                "dimension size of embedding vectors"
            )
        return self.CollectionStore.get_by_names(session, self.collection_names)

    def _query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query multiple collections."""
        self._init_models(embedding)
        with self.Session() as session:
            collections = self.get_collections(session)
            if not collections:
                raise ValueError("No collections found")
            return self._query_collection_multi(
                collections=collections, embedding=embedding, k=k, filter=filter
            )
