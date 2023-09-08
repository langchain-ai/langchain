from __future__ import annotations

import json
import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)


class SQLiteVSS(VectorStore):
    """Wrapper around SQLite with vss extension as a vector database.
    To use, you should have the ``sqlite-vss`` python package installed.
    Example:
        .. code-block:: python
            from langchain.vectorstores import SQLiteVSS
            from langchain.embeddings.openai import OpenAIEmbeddings
            ...
    """

    def __init__(
        self,
        table: str,
        connection: Optional[sqlite3.Connection],
        embedding: Embeddings,
        db_file: str = "vss.db",
    ):
        """Initialize with sqlite client with vss extension."""
        try:
            import sqlite_vss  # noqa  # pylint: disable=unused-import
        except ImportError:
            raise ImportError(
                "Could not import sqlite-vss python package. "
                "Please install it with `pip install sqlite-vss`."
            )

        if not connection:
            connection = self.create_connection(db_file)

        if not isinstance(embedding, Embeddings):
            warnings.warn("embeddings input must be Embeddings object.")

        self._connection = connection
        self._table = table
        self._embedding = embedding

        self.create_table_if_not_exists()

    def create_table_if_not_exists(self) -> None:
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table}
            (
              rowid INTEGER PRIMARY KEY AUTOINCREMENT,
              text TEXT,
              metadata BLOB,
              text_embedding BLOB
            )
            ;
            """
        )
        self._connection.execute(
            f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vss_{self._table} USING vss0(
                  text_embedding({self.get_dimensionality()})
                );
            """
        )
        self._connection.execute(
            f"""
                CREATE TRIGGER IF NOT EXISTS embed_text 
                AFTER INSERT ON {self._table}
                BEGIN
                    INSERT INTO vss_{self._table}(rowid, text_embedding)
                    VALUES (new.rowid, new.text_embedding) 
                    ;
                END;
            """
        )
        self._connection.commit()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore index.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        max_id = self._connection.execute(
            f"SELECT max(rowid) as rowid FROM {self._table}"
        ).fetchone()["rowid"]
        if max_id is None:  # no text added yet
            max_id = 0

        embeds = self._embedding.embed_documents(list(texts))
        if not metadatas:
            metadatas = [{} for _ in texts]
        data_input = [
            (text, json.dumps(metadata), json.dumps(embed))
            for text, metadata, embed in zip(texts, metadatas, embeds)
        ]
        self._connection.executemany(
            f"INSERT INTO {self._table}(text, metadata, text_embedding) "
            f"VALUES (?,?,?)",
            data_input,
        )
        self._connection.commit()
        # pulling every ids we just inserted
        results = self._connection.execute(
            f"SELECT rowid FROM {self._table} WHERE rowid > {max_id}"
        )
        return [row["rowid"] for row in results]

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        sql_query = f"""
            SELECT 
                text,
                metadata,
                distance
            FROM {self._table} e
            INNER JOIN vss_{self._table} v on v.rowid = e.rowid  
            WHERE vss_search(
              v.text_embedding,
              vss_search_params('{json.dumps(embedding)}', {k})
            )
        """
        cursor = self._connection.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()

        documents = []
        for row in results:
            metadata = json.loads(row["metadata"]) or {}
            doc = Document(page_content=row["text"], metadata=metadata)
            documents.append((doc, row["distance"]))

        return documents

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k
        )
        return [doc for doc, _ in documents]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query."""
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k
        )
        return documents

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k
        )
        return [doc for doc, _ in documents]

    @classmethod
    def from_texts(
        cls: Type[SQLiteVSS],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        table: str = "langchain",
        db_file: str = "vss.db",
        **kwargs: Any,
    ) -> SQLiteVSS:
        """Return VectorStore initialized from texts and embeddings."""
        connection = cls.create_connection(db_file)
        vss = cls(
            table=table, connection=connection, db_file=db_file, embedding=embedding
        )
        vss.add_texts(texts=texts, metadatas=metadatas)
        return vss

    @staticmethod
    def create_connection(db_file: str) -> sqlite3.Connection:
        import sqlite3

        import sqlite_vss

        connection = sqlite3.connect(db_file)
        connection.row_factory = sqlite3.Row
        connection.enable_load_extension(True)
        sqlite_vss.load(connection)
        connection.enable_load_extension(False)
        return connection

    def get_dimensionality(self) -> int:
        """
        Function that does a dummy embedding to figure out how many dimensions
        this embedding function returns. Needed for the virtual table DDL.
        """
        dummy_text = "This is a dummy text"
        dummy_embedding = self._embedding.embed_query(dummy_text)
        return len(dummy_embedding)
