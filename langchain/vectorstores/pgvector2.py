from __future__ import annotations

import enum
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    import sqlalchemy
    from sqlalchemy.dialects.postgresql import UUID
    from sqlalchemy.orm import Mapped, Session, declarative_base, relationship
except ImportError:
    raise ValueError(
        "Could not import sqlalchemy python package. "
        "Please install it with `pip install sqlalchemy`."
    )

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore

Base = declarative_base()  # type: Any

logging.basicConfig(level=logging.DEBUG)


##
def debug() -> None:
    import debugpy

    debugpy.listen(5680)
    print("⏳ Waiting for debugger attach (5680)...")
    debugpy.wait_for_client()


##

# debug()

ADA_TOKEN_COUNT = 1536
_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


# class DistanceStrategy(str, enum.Enum):
#     EUCLIDEAN = EmbeddingStore.embedding.l2_distance
#     COSINE = EmbeddingStore.embedding.cosine_distance
#     MAX_INNER_PRODUCT = EmbeddingStore.embedding.max_inner_product


class DistanceStrategy(str, enum.Enum):
    EUCLIDEAN = "EUCLIDEAN"
    COSINE = "COSINE"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.EUCLIDEAN


@dataclass
class PGVector(VectorStore):
    """
    VectorStore implementation using Postgres and pgvector.
    - `connection_string` is a postgres connection string.
    - `embedding_function` any embedding function implementing
        `langchain.embeddings.base.Embeddings` interface.
    - `collection_name` is the name of the collection to use. (default: langchain)
        - NOTE: This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
    - `distance_strategy` is the distance strategy to use. (default: EUCLIDEAN)
        - `EUCLIDEAN` is the euclidean distance.
        - `COSINE` is the cosine distance.
    - `pre_delete_collection` if True, will delete the collection if it exists.
        (default: False)
        - Useful for testing.
    """

    connection_string: str
    embedding_function: Embeddings
    collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME
    distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY
    pre_delete_collection: bool = False
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def __post_init__(
        self,
    ) -> None:
        """
        Initialize the store.
        """
        current_test = os.environ.get("PYTEST_CURRENT_TEST", None)
        print("__post_init__ start", current_test)

        try:
            from pgvector.sqlalchemy import Vector

            self.logger.debug(f"imported pgvector as {Vector}")
        except ImportError:
            raise ValueError(
                "Could not import pgvector python package. "
                "Please install it with `pip install pgvector`."
            )
        self.create_embedding_store_class_imperatively()
        self._conn = self.connect()
        # self.create_vector_extension()
        self.create_tables_if_not_exists()
        self.create_collection()
        print("__post_init__ done", current_test)

    def create_embedding_store_class_imperatively(self) -> None:
        """
        Create the EmbeddingStore class imperatively.
        """
        self.logger.info("AA")

        table_name = "langchain_pg_embedding"
        # self.logger.debug(Base.metadata.tables.keys())
        # if table_name in Base.metadata.tables:
        #     self.logger.debug("Imperatively mapping EmbeddingStore...")
        #     from sqlalchemy.orm import registry

        #     class EmbeddingStore:
        #         pass

        #     mapper_registry = registry()
        #     embedding_store_table = Base.metadata.tables[table_name]
        #     mapper_registry.map_imperatively(EmbeddingStore, embedding_store_table)
        #     self.EmbeddingStore = EmbeddingStore
        # else:
        #     self.logger.debug("Imperatively creating EmbeddingStore...")

        class BaseModel(Base):
            __abstract__ = True
            __table_args__ = {"extend_existing": True}
            uuid = sqlalchemy.Column(
                UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
            )

        class CollectionStore(BaseModel):
            __tablename__ = "langchain_pg_collection"

            name = sqlalchemy.Column(sqlalchemy.String)
            cmetadata = sqlalchemy.Column(sqlalchemy.JSON)

            embeddings = relationship(
                "EmbeddingStore",
                back_populates="collection",
                passive_deletes=True,
            )

            @classmethod
            def get_by_name(
                cls, session: Session, name: str
            ) -> Optional[CollectionStore]:
                return session.query(cls).filter(cls.name == name).first()

            @classmethod
            def get_or_create(
                cls,
                session: Session,
                name: str,
                cmetadata: Optional[dict] = None,
            ) -> Tuple["CollectionStore", bool]:
                """
                Get or create a collection.
                Returns [Collection, bool] where the bool is True if the collection was
                created.
                """
                created = False
                collection = cls.get_by_name(session, name)
                if collection:
                    return collection, created

                collection = cls(name=name, metadata=cmetadata)
                session.add(collection)
                session.commit()
                created = True
                return collection, created

        class EmbeddingStore(BaseModel):
            __tablename__ = table_name

            try:
                from pgvector.sqlalchemy import Vector
            except ImportError:
                raise ValueError(
                    "Could not import pgvector python package. "
                    "Please install it with `pip install pgvector`."
                )

            collection_id: Mapped[UUID] = sqlalchemy.Column(
                UUID(as_uuid=True),
                sqlalchemy.ForeignKey(
                    f"{CollectionStore.__tablename__}.uuid",
                    ondelete="CASCADE",
                ),
            )
            collection = relationship(CollectionStore, back_populates="embeddings")

            embedding: Vector = sqlalchemy.Column(Vector(ADA_TOKEN_COUNT))
            document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
            cmetadata = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)

            # custom_id : any user defined id
            custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)

        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore

    def connect(self) -> sqlalchemy.engine.Connection:
        engine = sqlalchemy.create_engine(self.connection_string)
        conn = engine.connect()
        return conn

    def create_vector_extension(self) -> None:
        try:
            with Session(self._conn) as session:
                statement = sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
                session.execute(statement)
                session.commit()
        except Exception as e:
            print(e)

    def create_tables_if_not_exists(self) -> None:
        Base.metadata.create_all(self._conn)

    def drop_tables(self) -> None:
        Base.metadata.drop_all(self._conn)

    def create_collection(self) -> None:
        # if self.pre_delete_collection:
        #     self.delete_collection()
        with Session(self._conn) as session:
            self.CollectionStore.get_or_create(session, self.collection_name)

    def delete_collection(self) -> None:
        print("Trying to delete collection")
        with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                print("Collection not found")
                return
            session.delete(collection)
            session.commit()
        print("Collection deleted")

    def get_collection(self, session: Session) -> Optional["PGVector.CollectionStore"]:
        return self.CollectionStore.get_by_name(session, self.collection_name)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        embeddings = self.embedding_function.embed_documents(list(texts))

        if not metadatas:
            metadatas = [{} for _ in texts]

        with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            print(f"Collections: {collection}")
            for text, metadata, embedding, id in zip(texts, metadatas, embeddings, ids):
                embedding_store = self.EmbeddingStore(
                    embedding=embedding,
                    document=text,
                    cmetadata=metadata,
                    custom_id=id,
                )
                print(f"EmbeddingStore: {embedding_store}")
                collection.embeddings.append(embedding_store)
                session.add(embedding_store)
            session.commit()
            print("Added texts")

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with PGVector with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
        )

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(embedding, k)
        return docs

    @property
    def distance_function(self) -> Callable:
        lut = {
            DistanceStrategy.EUCLIDEAN: self.EmbeddingStore.embedding.l2_distance,
            DistanceStrategy.COSINE: self.EmbeddingStore.embedding.cosine_distance,
        }

        return lut.get(
            self.distance_strategy, self.EmbeddingStore.embedding.l2_distance
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            results: list = (
                session.query(
                    self.EmbeddingStore,
                    self.distance_function(embedding).label("distance"),  # type: ignore
                )
                .filter(self.EmbeddingStore.collection_id == collection.uuid)
                .order_by(sqlalchemy.asc("distance"))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
                .limit(k)
                .all()
            )
            docs = [
                (
                    Document(
                        page_content=result.EmbeddingStore.document,
                        metadata=result.EmbeddingStore.cmetadata,
                    ),
                    result.distance if self.embedding_function is not None else None,
                )
                for result in results
            ]
            return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> "PGVector":
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.
        """

        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
        )

        store.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        return store

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="PGVECTOR_CONNECTION_STRING",
        )

        if not connection_string:
            raise ValueError(
                "Postgres connection string is required"
                "Either pass it as a parameter"
                "or set the PGVECTOR_CONNECTION_STRING environment variable."
            )

        return connection_string

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> "PGVector":
        """
        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.
        """

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        connection_string = cls.get_connection_string(kwargs)

        kwargs["connection_string"] = connection_string

        return cls.from_texts(
            texts=texts,
            pre_delete_collection=pre_delete_collection,
            embedding=embedding,
            distance_strategy=distance_strategy,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            **kwargs,
        )

    @classmethod
    def connection_string_from_db_params(
        cls,
        driver: str,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        """Return connection string from database parameters."""
        return f"postgresql+{driver}://{user}:{password}@{host}:{port}/{database}"
