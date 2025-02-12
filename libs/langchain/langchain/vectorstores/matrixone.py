"""Wrapper around Matrixone vector database."""
import uuid
from operator import itemgetter
from typing import Any, Callable, Iterable, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance
from sqlalchemy.orm import sessionmaker
from typing import List
from sqlalchemy import String, TEXT, create_engine, Table, Column, text
import sqlalchemy.types as types
from sqlalchemy.orm import registry
import sqlalchemy
import json
import uuid
import numpy as np
from mo_vector.client import MoVectorClient,QueryResult
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from retry import retry

key_metadata = 'metadata'
key_page_content = 'page_content'
key_id = 'id'
key_payload = 'payload'
key_doc_embedding_vector = 'doc_embedding_vector'


embed_model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L12-cos-v5", trust_remote_code=True)
embed_model_dims = embed_model.get_sentence_embedding_dimension()

class MODoubleVector(types.UserDefinedType):
    impl = types.TEXT

    def __init__(self, precision: int = None):
        if precision == None:
            raise ValueError(
                "precision is None. "
                "Please input precision."
            )
        self.precision = precision

    def get_col_spec(self, **kw):
        return "vecf64(%s)" % self.precision

    def bind_processor(self, dialect):
        def process(value):
            if value == None:
                return None
            return json.dumps(value, separators=(',', ':'))
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value == None:
                return None
            return json.loads(value)
        return process


class MODocEmbedding:
    def __init__(self, id=None, payload=None, doc_embedding_vector=None):
        if id == None:
            id = uuid.uuid4().hex
        self.id = id
        self.doc_embedding_vector = doc_embedding_vector
        self.payload = payload

    @classmethod
    def _to_vector_str(cls, vector: List[float]) -> str:
        return json.dumps(vector, separators=(',', ':'))


class MODocEmbeddingWithScore(MODocEmbedding):
    def __init__(self, id=None, payload=None, doc_embedding_vector=None):
        super.__init__(id, payload, doc_embedding_vector)
        self.score = 0


class Matrixone(VectorStore):
    """
    Wrapper around Matrixone vector database.
    Example:
        .. code-block:: python
            from langchain.vectorstores.matrixone import Matrixone
            MO = Matrixone(host="127.0.0.1", port=6001, user="user", passwd="pwd", db="database_name", table_name=table_name, embedding=embedding)
    """

    def __init__(self,
                 table_name: str,
                 embedding: Embeddings,
                 host: str,
                 user: str,
                 password: str,
                 dbname: str,
                 port: str = 6001):
        """Initialize with necessary components."""

        self.table_name = table_name
        self.embedding = embedding
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.dbname = dbname
        connectionSQL = "mysql+pymysql://%s:%s@%s:%d" % (
            user, password, host, port)
        self.engine = create_engine(connectionSQL, echo=True,
                                    pool_recycle=3600, pool_pre_ping=True)
        self.vector_store = MoVectorClient(
            table_name=self.table_name,
            connection_string=connectionSQL,
            vector_dimension=embed_model_dims,
            drop_existing_table=True,
        )

        with self.engine.connect() as conn:
            conn.execute(
                text("create database if not exists {database};use {database};".format(database=dbname)))
            conn.commit()
        connectionSQL = "mysql+pymysql://%s:%s@%s:%d/%s" % (
            user, password, host, port, dbname)
        self.engine = create_engine(connectionSQL, echo=True,
                                    pool_recycle=3600, pool_pre_ping=True)

    def _new_mo_doc_embedding_table_and_registry(self, dimensions):
        self.mapper_registry = registry()
        table = Table(
            self.table_name,
            self.mapper_registry.metadata,
            Column("id", String(256), primary_key=True),
            Column("payload", TEXT, nullable=False),
            Column("doc_embedding_vector", MODoubleVector(
                dimensions), nullable=False)
        )
        self.mapper_registry.metadata.create_all(bind=self.engine)

        if sqlalchemy.inspection.inspect(subject=MODocEmbedding, raiseerr=False) != None:
            return
        self.mapper_registry.map_imperatively(MODocEmbedding, table, properties={
            'id': table.c.id,
            'payload': table.c.payload,
            'doc_embedding_vector': table.c.doc_embedding_vector,
        })

    def _get_session(self):
        Session = sessionmaker(bind=self.engine)

        return Session()

    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        payloads = self._build_payloads(texts=texts, metadatas=metadatas)

        vectors = self.embedding.embed_documents(texts=texts)

        if len(vectors) <= 0:
            return []
        dimensions = len(vectors[0])

        self._new_mo_doc_embedding_table_and_registry(dimensions)

        self.pingdb()

        session = self._get_session()

        docs = []
        ids = []
        for i in range(len(texts)):
            id = uuid.uuid4().hex
            docs.append(MODocEmbedding(id=id, payload=json.dumps(
                payloads[i]), doc_embedding_vector=vectors[i]))
            self.vector_store.insert(embeddings=self.text_to_embedding(texts[i]), 
                                     ids=[id],
                                     metadatas=[metadatas[i]] if metadatas is not None else None,
                                     texts=[texts[i]])
            ids.append(id)

        session.add_all(docs)

        session.commit()

        session.close()

        return ids

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return docs most similar to query.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query.
        """
        results = self.similarity_search_with_score(query, k)
        return list(map(itemgetter(0), results))

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.
        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query vector.
        """
        results = self.similarity_search_by_vector_with_score(
            embedding=embedding, k=k)
        return list(map(itemgetter(0), results))

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
        return self.similarity_search_by_vector_with_score(embedding=self.embedding.embed_query(query), k=k)

    @retry(tries=3)
    def similarity_search_by_vector_with_score(
        self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:

        self.pingdb()

        session = self._get_session()

        sql = text("SELECT *,cosine_similarity(doc_embedding_vector, :embedding_str) as score FROM %s ORDER BY score DESC LIMIT :limit_count ;" % (self.table_name))
        results = session.execute(
            sql, {'embedding_str': self._to_vector_str(embedding), 'limit_count': k})

        session.commit()
        session.close()

        return [
            (
                self._document_from_payload(
                    result.payload, result.id, result.doc_embedding_vector),
                result.score,
            )
            for result in results.mappings().all()
        ]

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
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        query_embedding = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            query_embedding, k, fetch_k, lambda_mult, **kwargs
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
        results = self.max_marginal_relevance_search_with_score_by_vector(
            embedding=embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )
        return list(map(itemgetter(0), results))

    @retry(tries=3)
    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance and distance for
            each.
        """

        self.pingdb()

        session = self._get_session()

        sql = text("SELECT *,cosine_similarity(doc_embedding_vector, :embedding_str) as score FROM %s ORDER BY score DESC LIMIT :limit_count ;" % (self.table_name))
        results = session.execute(
            sql, {'embedding_str': self._to_vector_str(embedding), 'limit_count': fetch_k})

        session.commit()
        session.close()

        results_maps = results.mappings().all()
        embeddings = [
            self._str_to_vector(result.doc_embedding_vector) for result in results_maps
        ]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )

        return [
            (
                self._document_from_payload(
                    results_maps[i].payload, results_maps[i].id, results_maps[i].doc_embedding_vector),
                results_maps[i].score,
            )
            for i in mmr_selected
        ]

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embeddings

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.
        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.
        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        self.pingdb()

        session = self._get_session()

        docs = session.query(MODocEmbedding).filter(MODocEmbedding.id.in_(ids))
        for doc in docs:
            session.delete(doc)

        session.commit()

        session.close()

        return True

    def pingdb(self) -> None:
        # ping db to keep connection alive.
        try:
            session = sessionmaker(bind=self.engine)()
            session.execute("SELECT 1;")
            session.commit()
        except:
            session.rollback()
        finally:
            session.close()

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._cosine_relevance_score_fn

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        user: str,
        password: str,
        dbname: str,
        metadatas: List[dict] = None,
        host: str = '127.0.0.1',
        port: int = 6001,
        table_name: str = 'mo_doc_vector',
        **kwargs: Any,
    ) -> "Matrixone":
        """Construct Matrixone wrapper from raw documents.
        This is a user friendly interface that:
            1. Embeds documents.
            2. Initializes the Matrixone database
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python
                from langchain import Matrixone
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                mo = Matrixone.from_texts(texts=texts,embedding=embedding,user=user,password=password,dbname=dbname)
        """
        mo = Matrixone(table_name=table_name, embedding=embedding, host=host,
                       port=port, user=user, password=password, dbname=dbname)
        mo._delete_table_()
        mo.add_texts(texts=texts, metadatas=metadatas)
        return mo

    def _delete_table_(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(
                text("drop table if exists {database}.{table_name};".format(database=self.dbname, table_name=self.table_name)))
            conn.commit()

    @classmethod
    def _build_payloads(
        cls, texts: Iterable[str], metadatas: Optional[List[dict]]
    ) -> List[dict]:
        return [
            {
                key_page_content: text,
                key_metadata: metadatas[i] if metadatas is not None else None,
            }
            for i, text in enumerate(texts)
        ]

    @classmethod
    def _document_from_payload(cls, payload: str, id: str, embedding: List[float]) -> Document:
        payload_map = json.loads(payload)
        if not isinstance(payload_map, dict):
            return None
        metadata_map = payload_map.get(key_metadata, None)
        if metadata_map == None:
            payload_map[key_metadata] = {}
        payload_map[key_metadata][key_id] = id
        payload_map[key_metadata][key_doc_embedding_vector] = embedding

        return Document(
            page_content=payload_map[key_page_content],
            metadata=payload_map[key_metadata],
        )

    @classmethod
    def _to_vector_str(cls, vector: List[float]) -> str:
        return json.dumps(vector, separators=(',', ':'))

    @classmethod
    def _str_to_vector(cls, vector_str: str) -> List[float]:
        return json.loads(vector_str)

    @classmethod
    def text_to_embedding(text):
       embedding = embed_model.encode(text)
       return embedding.tolist()

    @classmethod
    def create_full_text_index(self):
        self.vector_store.create_full_text_index()

    @classmethod
    def mix_query(
        self,
        query_vector: List[float],
        key_words: List[str] = None,
        rerank_option: Optional[dict] = None,
        k: int = 5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[QueryResult]:
        return self.vector_store.mix_query(
            query_vector=query_vector,
            key_words=key_words,
            rerank_option=rerank_option,
            k=k,
            filter=filter,
        )