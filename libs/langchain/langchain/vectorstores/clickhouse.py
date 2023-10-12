from __future__ import annotations

import json
import logging
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from langchain.docstore.document import Document
from langchain.pydantic_v1 import BaseSettings
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore

logger = logging.getLogger()


def has_mul_sub_str(s: str, *args: Any) -> bool:
    """
    Check if a string contains multiple substrings.
    Args:
        s: string to check.
        *args: substrings to check.

    Returns:
        True if all substrings are in the string, False otherwise.
    """
    for a in args:
        if a not in s:
            return False
    return True


class ClickhouseSettings(BaseSettings):
    """`ClickHouse` client configuration.

    Attribute:
        host (str) : An URL to connect to MyScale backend.
                             Defaults to 'localhost'.
        port (int) : URL port to connect with HTTP. Defaults to 8443.
        username (str) : Username to login. Defaults to None.
        password (str) : Password to login. Defaults to None.
        index_type (str): index type string.
        index_param (list): index build parameter.
        index_query_params(dict): index query parameters.
        database (str) : Database name to find the table. Defaults to 'default'.
        table (str) : Table name to operate on.
                      Defaults to 'vector_table'.
        metric (str) : Metric to compute distance,
                       supported are ('angular', 'euclidean', 'manhattan', 'hamming',
                       'dot'). Defaults to 'angular'.
                       https://github.com/spotify/annoy/blob/main/src/annoymodule.cc#L149-L169

        column_map (Dict) : Column type map to project column name onto langchain
                            semantics. Must have keys: `text`, `id`, `vector`,
                            must be same size to number of columns. For example:
                            .. code-block:: python

                                {
                                    'id': 'text_id',
                                    'uuid': 'global_unique_id'
                                    'embedding': 'text_embedding',
                                    'document': 'text_plain',
                                    'metadata': 'metadata_dictionary_in_json',
                                }

                            Defaults to identity map.
    """

    host: str = "localhost"
    port: int = 8123

    username: Optional[str] = None
    password: Optional[str] = None

    index_type: str = "annoy"
    # Annoy supports L2Distance and cosineDistance.
    index_param: Optional[Union[List, Dict]] = ["'L2Distance'", 100]
    index_query_params: Dict[str, str] = {}

    column_map: Dict[str, str] = {
        "id": "id",
        "uuid": "uuid",
        "document": "document",
        "embedding": "embedding",
        "metadata": "metadata",
    }

    database: str = "default"
    table: str = "langchain"
    metric: str = "angular"

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    class Config:
        env_file = ".env"
        env_prefix = "clickhouse_"
        env_file_encoding = "utf-8"


class Clickhouse(VectorStore):
    """`ClickHouse VectorSearch` vector store.

    You need a `clickhouse-connect` python package, and a valid account
    to connect to ClickHouse.

    ClickHouse can not only search with simple vector indexes,
    it also supports complex query with multiple conditions,
    constraints and even sub-queries.

    For more information, please visit
        [ClickHouse official site](https://clickhouse.com/clickhouse)
    """

    def __init__(
        self,
        embedding: Embeddings,
        config: Optional[ClickhouseSettings] = None,
        **kwargs: Any,
    ) -> None:
        """ClickHouse Wrapper to LangChain

        embedding_function (Embeddings):
        config (ClickHouseSettings): Configuration to ClickHouse Client
        Other keyword arguments will pass into
            [clickhouse-connect](https://docs.clickhouse.com/)
        """
        try:
            from clickhouse_connect import get_client
        except ImportError:
            raise ImportError(
                "Could not import clickhouse connect python package. "
                "Please install it with `pip install clickhouse-connect`."
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
            self.config = ClickhouseSettings()
        assert self.config
        assert self.config.host and self.config.port
        assert (
            self.config.column_map
            and self.config.database
            and self.config.table
            and self.config.metric
        )
        for k in ["id", "embedding", "document", "metadata", "uuid"]:
            assert k in self.config.column_map
        assert self.config.metric in [
            "angular",
            "euclidean",
            "manhattan",
            "hamming",
            "dot",
        ]

        # initialize the schema
        dim = len(embedding.embed_query("test"))

        index_params = (
            (
                ",".join([f"'{k}={v}'" for k, v in self.config.index_param.items()])
                if self.config.index_param
                else ""
            )
            if isinstance(self.config.index_param, Dict)
            else ",".join([str(p) for p in self.config.index_param])
            if isinstance(self.config.index_param, List)
            else self.config.index_param
        )

        self.schema = f"""\
CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(
    {self.config.column_map['id']} Nullable(String),
    {self.config.column_map['document']} Nullable(String),
    {self.config.column_map['embedding']} Array(Float32),
    {self.config.column_map['metadata']} JSON,
    {self.config.column_map['uuid']} UUID DEFAULT generateUUIDv4(),
    CONSTRAINT cons_vec_len CHECK length({self.config.column_map['embedding']}) = {dim},
    INDEX vec_idx {self.config.column_map['embedding']} TYPE \
{self.config.index_type}({index_params}) GRANULARITY 1000
) ENGINE = MergeTree ORDER BY uuid SETTINGS index_granularity = 8192\
"""
        self.dim = dim
        self.BS = "\\"
        self.must_escape = ("\\", "'")
        self.embedding_function = embedding
        self.dist_order = "ASC"  # Only support ConsingDistance and L2Distance

        # Create a connection to clickhouse
        self.client = get_client(
            host=self.config.host,
            port=self.config.port,
            username=self.config.username,
            password=self.config.password,
            **kwargs,
        )
        # Enable JSON type
        self.client.command("SET allow_experimental_object_type=1")
        # Enable Annoy index
        self.client.command("SET allow_experimental_annoy_index=1")
        self.client.command(self.schema)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def escape_str(self, value: str) -> str:
        return "".join(f"{self.BS}{c}" if c in self.must_escape else c for c in value)

    def _build_insert_sql(self, transac: Iterable, column_names: Iterable[str]) -> str:
        ks = ",".join(column_names)
        _data = []
        for n in transac:
            n = ",".join([f"'{self.escape_str(str(_n))}'" for _n in n])
            _data.append(f"({n})")
        i_str = f"""
                INSERT INTO TABLE 
                    {self.config.database}.{self.config.table}({ks})
                VALUES
                {','.join(_data)}
                """
        return i_str

    def _insert(self, transac: Iterable, column_names: Iterable[str]) -> None:
        _insert_query = self._build_insert_sql(transac, column_names)
        self.client.command(_insert_query)

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
        config: Optional[ClickhouseSettings] = None,
        text_ids: Optional[Iterable[str]] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Clickhouse:
        """Create ClickHouse wrapper with existing texts

        Args:
            embedding_function (Embeddings): Function to extract text embedding
            texts (Iterable[str]): List or tuple of strings to be added
            config (ClickHouseSettings, Optional): ClickHouse configuration
            text_ids (Optional[Iterable], optional): IDs for the texts.
                                                     Defaults to None.
            batch_size (int, optional): Batchsize when transmitting data to ClickHouse.
                                        Defaults to 32.
            metadata (List[dict], optional): metadata to texts. Defaults to None.
            Other keyword arguments will pass into
                [clickhouse-connect](https://clickhouse.com/docs/en/integrations/python#clickhouse-connect-driver-api)
        Returns:
            ClickHouse Index
        """
        ctx = cls(embedding, config, **kwargs)
        ctx.add_texts(texts, ids=text_ids, batch_size=batch_size, metadatas=metadatas)
        return ctx

    def __repr__(self) -> str:
        """Text representation for ClickHouse Vector Store, prints backends, username
            and schemas. Easy to use with `str(ClickHouse())`

        Returns:
            repr: string to show connection info and data schema
        """
        _repr = f"\033[92m\033[1m{self.config.database}.{self.config.table} @ "
        _repr += f"{self.config.host}:{self.config.port}\033[0m\n\n"
        _repr += f"\033[1musername: {self.config.username}\033[0m\n\nTable Schema:\n"
        _repr += "-" * 51 + "\n"
        for r in self.client.query(
            f"DESC {self.config.database}.{self.config.table}"
        ).named_results():
            _repr += (
                f"|\033[94m{r['name']:24s}\033[0m|\033[96m{r['type']:24s}\033[0m|\n"
            )
        _repr += "-" * 51 + "\n"
        return _repr

    def _build_query_sql(
        self, q_emb: List[float], topk: int, where_str: Optional[str] = None
    ) -> str:
        q_emb_str = ",".join(map(str, q_emb))
        if where_str:
            where_str = f"PREWHERE {where_str}"
        else:
            where_str = ""

        settings_strs = []
        if self.config.index_query_params:
            for k in self.config.index_query_params:
                settings_strs.append(f"SETTING {k}={self.config.index_query_params[k]}")
        q_str = f"""
            SELECT {self.config.column_map['document']}, 
                {self.config.column_map['metadata']}, dist
            FROM {self.config.database}.{self.config.table}
            {where_str}
            ORDER BY L2Distance({self.config.column_map['embedding']}, [{q_emb_str}]) 
                AS dist {self.dist_order}
            LIMIT {topk} {' '.join(settings_strs)}
            """
        return q_str

    def similarity_search(
        self, query: str, k: int = 4, where_str: Optional[str] = None, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search with ClickHouse

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
        """Perform a similarity search with ClickHouse by vectors

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
                    metadata=r[self.config.column_map["metadata"]],
                )
                for r in self.client.query(q_str).named_results()
            ]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, where_str: Optional[str] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with ClickHouse

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
                        metadata=r[self.config.column_map["metadata"]],
                    ),
                    r["dist"],
                )
                for r in self.client.query(q_str).named_results()
            ]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

    def drop(self) -> None:
        """
        Helper function: Drop data
        """
        self.client.command(
            f"DROP TABLE IF EXISTS {self.config.database}.{self.config.table}"
        )

    @property
    def metadata_column(self) -> str:
        return self.config.column_map["metadata"]
