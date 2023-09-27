from __future__ import annotations

import json
import logging
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.pydantic_v1 import BaseSettings
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore

logger = logging.getLogger()
DEBUG = False


def has_mul_sub_str(s: str, *args: Any) -> bool:
    """
    Check if a string has multiple substrings.
    Args:
        s: The string to check
        *args: The substrings to check for in the string

    Returns:
        bool: True if all substrings are present in the string, False otherwise
    """
    for a in args:
        if a not in s:
            return False
    return True


def debug_output(s: Any) -> None:
    """
    Print a debug message if DEBUG is True.
    Args:
        s: The message to print
    """
    if DEBUG:
        print(s)


def get_named_result(connection: Any, query: str) -> List[dict[str, Any]]:
    """
    Get a named result from a query.
    Args:
        connection: The connection to the database
        query: The query to execute

    Returns:
        List[dict[str, Any]]: The result of the query
    """
    cursor = connection.cursor()
    cursor.execute(query)
    columns = cursor.description
    result = []
    for value in cursor.fetchall():
        r = {}
        for idx, datum in enumerate(value):
            k = columns[idx][0]
            r[k] = datum
        result.append(r)
    debug_output(result)
    cursor.close()
    return result


class StarRocksSettings(BaseSettings):
    """StarRocks client configuration.

    Attribute:
        StarRocks_host (str) : An URL to connect to MyScale backend.
                             Defaults to 'localhost'.
        StarRocks_port (int) : URL port to connect with HTTP. Defaults to 8443.
        username (str) : Username to login. Defaults to None.
        password (str) : Password to login. Defaults to None.
        database (str) : Database name to find the table. Defaults to 'default'.
        table (str) : Table name to operate on.
                      Defaults to 'vector_table'.

        column_map (Dict) : Column type map to project column name onto langchain
                            semantics. Must have keys: `text`, `id`, `vector`,
                            must be same size to number of columns. For example:
                            .. code-block:: python

                                {
                                    'id': 'text_id',
                                    'embedding': 'text_embedding',
                                    'document': 'text_plain',
                                    'metadata': 'metadata_dictionary_in_json',
                                }

                            Defaults to identity map.
    """

    host: str = "localhost"
    port: int = 9030
    username: str = "root"
    password: str = ""

    column_map: Dict[str, str] = {
        "id": "id",
        "document": "document",
        "embedding": "embedding",
        "metadata": "metadata",
    }

    database: str = "default"
    table: str = "langchain"

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    class Config:
        env_file = ".env"
        env_prefix = "starrocks_"
        env_file_encoding = "utf-8"


class StarRocks(VectorStore):
    """`StarRocks` vector store.

    You need a `pymysql` python package, and a valid account
    to connect to StarRocks.

    Right now StarRocks has only implemented `cosine_similarity` function to
    compute distance between two vectors. And there is no vector inside right now,
    so we have to iterate all vectors and compute spatial distance.

    For more information, please visit
        [StarRocks official site](https://www.starrocks.io/)
        [StarRocks github](https://github.com/StarRocks/starrocks)
    """

    def __init__(
        self,
        embedding: Embeddings,
        config: Optional[StarRocksSettings] = None,
        **kwargs: Any,
    ) -> None:
        """StarRocks Wrapper to LangChain

        embedding_function (Embeddings):
        config (StarRocksSettings): Configuration to StarRocks Client
        """
        try:
            import pymysql  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "Could not import pymysql python package. "
                "Please install it with `pip install pymysql`."
            )
        try:
            from tqdm import tqdm

            self.pgbar = tqdm
        except ImportError:
            # Just in case if tqdm is not installed
            self.pgbar = lambda x, **kwargs: x
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = StarRocksSettings()
        assert self.config
        assert self.config.host and self.config.port
        assert self.config.column_map and self.config.database and self.config.table
        for k in ["id", "embedding", "document", "metadata"]:
            assert k in self.config.column_map

        # initialize the schema
        dim = len(embedding.embed_query("test"))

        self.schema = f"""\
CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(    
    {self.config.column_map['id']} string,
    {self.config.column_map['document']} string,
    {self.config.column_map['embedding']} array<float>,
    {self.config.column_map['metadata']} string
) ENGINE = OLAP PRIMARY KEY(id) DISTRIBUTED BY HASH(id) \
  PROPERTIES ("replication_num" = "1")\
"""
        self.dim = dim
        self.BS = "\\"
        self.must_escape = ("\\", "'")
        self.embedding_function = embedding
        self.dist_order = "DESC"
        debug_output(self.config)

        # Create a connection to StarRocks
        self.connection = pymysql.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.username,
            password=self.config.password,
            database=self.config.database,
            **kwargs,
        )

        debug_output(self.schema)
        get_named_result(self.connection, self.schema)

    def escape_str(self, value: str) -> str:
        return "".join(f"{self.BS}{c}" if c in self.must_escape else c for c in value)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def _build_insert_sql(self, transac: Iterable, column_names: Iterable[str]) -> str:
        ks = ",".join(column_names)
        embed_tuple_index = tuple(column_names).index(
            self.config.column_map["embedding"]
        )
        _data = []
        for n in transac:
            n = ",".join(
                [
                    f"'{self.escape_str(str(_n))}'"
                    if idx != embed_tuple_index
                    else f"array<float>{str(_n)}"
                    for (idx, _n) in enumerate(n)
                ]
            )
            _data.append(f"({n})")
        i_str = f"""
                INSERT INTO
                    {self.config.database}.{self.config.table}({ks})
                VALUES
                {','.join(_data)}
                """
        return i_str

    def _insert(self, transac: Iterable, column_names: Iterable[str]) -> None:
        _insert_query = self._build_insert_sql(transac, column_names)
        debug_output(_insert_query)
        get_named_result(self.connection, _insert_query)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 32,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert more texts through the embeddings and add to the VectorStore.

        Args:
            texts: Iterable of strings to add to the VectorStore.
            ids: Optional list of ids to associate with the texts.
            batch_size: Batch size of insertion
            metadata: Optional column data to be inserted

        Returns:
            List of ids from adding the texts into the VectorStore.

        """
        # Embed and create the documents
        ids = ids or [sha1(t.encode("utf-8")).hexdigest() for t in texts]
        colmap_ = self.config.column_map
        transac = []
        column_names = {
            colmap_["id"]: ids,
            colmap_["document"]: texts,
            colmap_["embedding"]: self.embedding_function.embed_documents(list(texts)),
        }
        metadatas = metadatas or [{} for _ in texts]
        column_names[colmap_["metadata"]] = map(json.dumps, metadatas)
        assert len(set(colmap_) - set(column_names)) >= 0
        keys, values = zip(*column_names.items())
        try:
            t = None
            for v in self.pgbar(
                zip(*values), desc="Inserting data...", total=len(metadatas)
            ):
                assert (
                    len(v[keys.index(self.config.column_map["embedding"])]) == self.dim
                )
                transac.append(v)
                if len(transac) == batch_size:
                    if t:
                        t.join()
                    t = Thread(target=self._insert, args=[transac, keys])
                    t.start()
                    transac = []
            if len(transac) > 0:
                if t:
                    t.join()
                self._insert(transac, keys)
            return [i for i in ids]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        config: Optional[StarRocksSettings] = None,
        text_ids: Optional[Iterable[str]] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> StarRocks:
        """Create StarRocks wrapper with existing texts

        Args:
            embedding_function (Embeddings): Function to extract text embedding
            texts (Iterable[str]): List or tuple of strings to be added
            config (StarRocksSettings, Optional): StarRocks configuration
            text_ids (Optional[Iterable], optional): IDs for the texts.
                                                     Defaults to None.
            batch_size (int, optional): Batchsize when transmitting data to StarRocks.
                                        Defaults to 32.
            metadata (List[dict], optional): metadata to texts. Defaults to None.
        Returns:
            StarRocks Index
        """
        ctx = cls(embedding, config, **kwargs)
        ctx.add_texts(texts, ids=text_ids, batch_size=batch_size, metadatas=metadatas)
        return ctx

    def __repr__(self) -> str:
        """Text representation for StarRocks Vector Store, prints backends, username
            and schemas. Easy to use with `str(StarRocks())`

        Returns:
            repr: string to show connection info and data schema
        """
        _repr = f"\033[92m\033[1m{self.config.database}.{self.config.table} @ "
        _repr += f"{self.config.host}:{self.config.port}\033[0m\n\n"
        _repr += f"\033[1musername: {self.config.username}\033[0m\n\nTable Schema:\n"
        width = 25
        fields = 3
        _repr += "-" * (width * fields + 1) + "\n"
        columns = ["name", "type", "key"]
        _repr += f"|\033[94m{columns[0]:24s}\033[0m|\033[96m{columns[1]:24s}"
        _repr += f"\033[0m|\033[96m{columns[2]:24s}\033[0m|\n"
        _repr += "-" * (width * fields + 1) + "\n"
        q_str = f"DESC {self.config.database}.{self.config.table}"
        debug_output(q_str)
        rs = get_named_result(self.connection, q_str)
        for r in rs:
            _repr += f"|\033[94m{r['Field']:24s}\033[0m|\033[96m{r['Type']:24s}"
            _repr += f"\033[0m|\033[96m{r['Key']:24s}\033[0m|\n"
        _repr += "-" * (width * fields + 1) + "\n"
        return _repr

    def _build_query_sql(
        self, q_emb: List[float], topk: int, where_str: Optional[str] = None
    ) -> str:
        q_emb_str = ",".join(map(str, q_emb))
        if where_str:
            where_str = f"WHERE {where_str}"
        else:
            where_str = ""

        q_str = f"""
            SELECT {self.config.column_map['document']}, 
                {self.config.column_map['metadata']}, 
                cosine_similarity_norm(array<float>[{q_emb_str}],
                  {self.config.column_map['embedding']}) as dist
            FROM {self.config.database}.{self.config.table}
            {where_str}
            ORDER BY dist {self.dist_order}
            LIMIT {topk}
            """

        debug_output(q_str)
        return q_str

    def similarity_search(
        self, query: str, k: int = 4, where_str: Optional[str] = None, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search with StarRocks

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): where condition string.
                                                 Defaults to None.

            NOTE: Please do not let end-user to fill this and always be aware
                  of SQL injection. When dealing with metadatas, remember to
                  use `{self.metadata_column}.attribute` instead of `attribute`
                  alone. The default name for it is `metadata`.

        Returns:
            List[Document]: List of Documents
        """
        return self.similarity_search_by_vector(
            self.embedding_function.embed_query(query), k, where_str, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search with StarRocks by vectors

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): where condition string.
                                                 Defaults to None.

            NOTE: Please do not let end-user to fill this and always be aware
                  of SQL injection. When dealing with metadatas, remember to
                  use `{self.metadata_column}.attribute` instead of `attribute`
                  alone. The default name for it is `metadata`.

        Returns:
            List[Document]: List of (Document, similarity)
        """
        q_str = self._build_query_sql(embedding, k, where_str)
        try:
            return [
                Document(
                    page_content=r[self.config.column_map["document"]],
                    metadata=json.loads(r[self.config.column_map["metadata"]]),
                )
                for r in get_named_result(self.connection, q_str)
            ]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, where_str: Optional[str] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with StarRocks

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): where condition string.
                                                 Defaults to None.

            NOTE: Please do not let end-user to fill this and always be aware
                  of SQL injection. When dealing with metadatas, remember to
                  use `{self.metadata_column}.attribute` instead of `attribute`
                  alone. The default name for it is `metadata`.

        Returns:
            List[Document]: List of documents
        """
        q_str = self._build_query_sql(
            self.embedding_function.embed_query(query), k, where_str
        )
        try:
            return [
                (
                    Document(
                        page_content=r[self.config.column_map["document"]],
                        metadata=json.loads(r[self.config.column_map["metadata"]]),
                    ),
                    r["dist"],
                )
                for r in get_named_result(self.connection, q_str)
            ]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

    def drop(self) -> None:
        """
        Helper function: Drop data
        """
        get_named_result(
            self.connection,
            f"DROP TABLE IF EXISTS {self.config.database}.{self.config.table}",
        )

    @property
    def metadata_column(self) -> str:
        return self.config.column_map["metadata"]
