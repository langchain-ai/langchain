"""VectorStore wrapper around a Postgres database."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import sqlalchemy
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Session, declarative_base, relationship

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore

Base = declarative_base()  # type: Any


ADA_TOKEN_COUNT = 1536
_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


class BaseModel(Base):
    __abstract__ = True
    uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class CollectionStore(BaseModel):
    __tablename__ = "langchain_pg_collection"

    name = sqlalchemy.Column(sqlalchemy.String)
    cmetadata = sqlalchemy.Column(JSON)

    embeddings = relationship(
        "EmbeddingStore",
        back_populates="collection",
        passive_deletes=True,
    )

    @classmethod
    def get_by_name(cls, session: Session, name: str) -> Optional["CollectionStore"]:
        return session.query(cls).filter(cls.name == name).first()  # type: ignore

    @classmethod
    def get_or_create(
        cls,
        session: Session,
        name: str,
        cmetadata: Optional[dict] = None,
    ) -> Tuple["CollectionStore", bool]:
        """
        Get or create a collection.
        Returns [Collection, bool] where the bool is True if the collection was created.
        """
        created = False
        collection = cls.get_by_name(session, name)
        if collection:
            return collection, created

        collection = cls(name=name, cmetadata=cmetadata)
        session.add(collection)
        session.commit()
        created = True
        return collection, created


class EmbeddingStore(BaseModel):
    __tablename__ = "langchain_pg_embedding"

    collection_id = sqlalchemy.Column(
        UUID(as_uuid=True),
        sqlalchemy.ForeignKey(
            f"{CollectionStore.__tablename__}.uuid",
            ondelete="CASCADE",
        ),
    )
    collection = relationship(CollectionStore, back_populates="embeddings")

    embedding = sqlalchemy.Column(sqlalchemy.ARRAY(sqlalchemy.REAL))  # type: ignore
    document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    cmetadata = sqlalchemy.Column(JSON, nullable=True)

    # custom_id : any user defined id
    custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)


class QueryResult:
    EmbeddingStore: EmbeddingStore
    distance: float


class PGEmbedding(VectorStore):
    """
    VectorStore implementation using Postgres and the pg_embedding extension.
    pg_embedding uses sequential scan by default. but you can create a HNSW index
    using the create_hnsw_index method.
    - `connection_string` is a postgres connection string.
    - `embedding_function` any embedding function implementing
        `langchain.embeddings.base.Embeddings` interface.
    - `collection_name` is the name of the collection to use. (default: langchain)
        - NOTE: This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
    - `distance_strategy` is the distance strategy to use. (default: EUCLIDEAN)
        - `EUCLIDEAN` is the euclidean distance.
    - `pre_delete_collection` if True, will delete the collection if it exists.
        (default: False)
        - Useful for testing.
    """

    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[dict] = None,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.__post_init__()

    def __post_init__(
        self,
    ) -> None:
        self._conn = self.connect()
        self.create_hnsw_extension()
        self.create_tables_if_not_exists()
        self.create_collection()

    def connect(self) -> sqlalchemy.engine.Connection:
        engine = sqlalchemy.create_engine(self.connection_string)
        conn = engine.connect()
        return conn

    def create_hnsw_extension(self) -> None:
        try:
            with Session(self._conn) as session:
                statement = sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS embedding")
                session.execute(statement)
                session.commit()
        except Exception as e:
            self.logger.exception(e)

    def create_tables_if_not_exists(self) -> None:
        with self._conn.begin():
            Base.metadata.create_all(self._conn)

    def drop_tables(self) -> None:
        with self._conn.begin():
            Base.metadata.drop_all(self._conn)

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        with Session(self._conn) as session:
            CollectionStore.get_or_create(
                session, self.collection_name, cmetadata=self.collection_metadata
            )

    def create_hnsw_index(
        self,
        max_elements: int = 10000,
        dims: int = ADA_TOKEN_COUNT,
        m: int = 8,
        ef_construction: int = 16,
        ef_search: int = 16,
    ) -> None:
        create_index_query = sqlalchemy.text(
            "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_idx "
            "ON langchain_pg_embedding USING hnsw (embedding) "
            "WITH ("
            "maxelements = {}, "
            "dims = {}, "
            "m = {}, "
            "efconstruction = {}, "
            "efsearch = {}"
            ");".format(max_elements, dims, m, ef_construction, ef_search)
        )

        # Execute the queries
        try:
            with Session(self._conn) as session:
                # Create the HNSW index
                session.execute(create_index_query)
                session.commit()
            print("HNSW extension and index created successfully.")
        except Exception as e:
            print(f"Failed to create HNSW extension or index: {e}")

    def delete_collection(self) -> None:
        self.logger.debug("Trying to delete collection")
        with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                self.logger.warning("Collection not found")
                return
            session.delete(collection)
            session.commit()

    def get_collection(self, session: Session) -> Optional["CollectionStore"]:
        return CollectionStore.get_by_name(session, self.collection_name)

    @classmethod
    def _initialize_from_embeddings(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGEmbedding:
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            pre_delete_collection=pre_delete_collection,
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    def add_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: List[str],
        **kwargs: Any,
    ) -> None:
        with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            for text, metadata, embedding, id in zip(texts, metadatas, embeddings, ids):
                embedding_store = EmbeddingStore(
                    embedding=embedding,
                    document=text,
                    cmetadata=metadata,
                    custom_id=id,
                )
                collection.embeddings.append(embedding_store)
                session.add(embedding_store)
            session.commit()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        embeddings = self.embedding_function.embed_documents(list(texts))

        if not metadatas:
            metadatas = [{} for _ in texts]

        with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            for text, metadata, embedding, id in zip(texts, metadatas, embeddings, ids):
                embedding_store = EmbeddingStore(
                    embedding=embedding,
                    document=text,
                    cmetadata=metadata,
                    custom_id=id,
                )
                collection.embeddings.append(embedding_store)
                session.add(embedding_store)
            session.commit()

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        with Session(self._conn) as session:
            collection = self.get_collection(session)
            set_enable_seqscan_stmt = sqlalchemy.text("SET enable_seqscan = off")
            session.execute(set_enable_seqscan_stmt)
            if not collection:
                raise ValueError("Collection not found")

            filter_by = EmbeddingStore.collection_id == collection.uuid

            if filter is not None:
                filter_clauses = []
                for key, value in filter.items():
                    IN = "in"
                    if isinstance(value, dict) and IN in map(str.lower, value):
                        value_case_insensitive = {
                            k.lower(): v for k, v in value.items()
                        }
                        filter_by_metadata = EmbeddingStore.cmetadata[key].astext.in_(
                            value_case_insensitive[IN]
                        )
                        filter_clauses.append(filter_by_metadata)
                    else:
                        filter_by_metadata = EmbeddingStore.cmetadata[
                            key
                        ].astext == str(value)
                        filter_clauses.append(filter_by_metadata)

                filter_by = sqlalchemy.and_(filter_by, *filter_clauses)

            results: List[QueryResult] = (
                session.query(
                    EmbeddingStore,
                    func.abs(EmbeddingStore.embedding.op("<->")(embedding)).label(
                        "distance"
                    ),
                )  # Specify the columns you need here, e.g., EmbeddingStore.embedding
                .filter(filter_by)
                .order_by(
                    func.abs(EmbeddingStore.embedding.op("<->")(embedding)).asc()
                )  # Using PostgreSQL specific operator with the correct column name
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
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls: Type[PGEmbedding],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGEmbedding:
        embeddings = embedding.embed_documents(list(texts))

        return cls._initialize_from_embeddings(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGEmbedding:
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls._initialize_from_embeddings(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls: Type[PGEmbedding],
        embedding: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGEmbedding:
        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            pre_delete_collection=pre_delete_collection,
        )

        return store

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="POSTGRES_CONNECTION_STRING",
        )

        if not connection_string:
            raise ValueError(
                "Postgres connection string is required"
                "Either pass it as a parameter"
                "or set the POSTGRES_CONNECTION_STRING environment variable."
            )

        return connection_string

    @classmethod
    def from_documents(
        cls: Type[PGEmbedding],
        documents: List[Document],
        embedding: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGEmbedding:
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        connection_string = cls.get_connection_string(kwargs)

        kwargs["connection_string"] = connection_string

        return cls.from_texts(
            texts=texts,
            pre_delete_collection=pre_delete_collection,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            **kwargs,
        )
