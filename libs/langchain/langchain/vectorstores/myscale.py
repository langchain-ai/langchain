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


class MyScaleSettings(BaseSettings):
    """MyScale client configuration.

    Attribute:
        myscale_host (str) : An URL to connect to MyScale backend.
                             Defaults to 'localhost'.
        myscale_port (int) : URL port to connect with HTTP. Defaults to 8443.
        username (str) : Username to login. Defaults to None.
        password (str) : Password to login. Defaults to None.
        index_type (str): index type string.
        index_param (dict): index build parameter.
        database (str) : Database name to find the table. Defaults to 'default'.
        table (str) : Table name to operate on.
                      Defaults to 'vector_table'.
        metric (str) : Metric to compute distance,
                       supported are ('L2', 'Cosine', 'IP'). Defaults to 'Cosine'.
        column_map (Dict) : Column type map to project column name onto langchain
                            semantics. Must have keys: `text`, `id`, `vector`,
                            must be same size to number of columns. For example:
                            .. code-block:: python

                                {
                                    'id': 'text_id',
                                    'vector': 'text_embedding',
                                    'text': 'text_plain',
                                    'metadata': 'metadata_dictionary_in_json',
                                }

                            Defaults to identity map.

    """

    host: str = "localhost"
    port: int = 8443

    username: Optional[str] = None
    password: Optional[str] = None

    index_type: str = "MSTG"
    index_param: Optional[Dict[str, str]] = None

    column_map: Dict[str, str] = {
        "id": "id",
        "text": "text",
        "vector": "vector",
        "metadata": "metadata",
    }

    database: str = "default"
    table: str = "langchain"
    metric: str = "Cosine"

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    class Config:
        env_file = ".env"
        env_prefix = "myscale_"
        env_file_encoding = "utf-8"


class MyScale(VectorStore):
    """`MyScale` vector store.

    You need a `clickhouse-connect` python package, and a valid account
    to connect to MyScale.

    MyScale can not only search with simple vector indexes.
    It also supports a complex query with multiple conditions,
    constraints and even sub-queries.

    For more information, please visit
        [myscale official site](https://docs.myscale.com/en/overview/)
    """

    def __init__(
        self,
        embedding: Embeddings,
        config: Optional[MyScaleSettings] = None,
        **kwargs: Any,
    ) -> None:
        """MyScale Wrapper to LangChain

        embedding (Embeddings):
        config (MyScaleSettings): Configuration to MyScale Client
        Other keyword arguments will pass into
            [clickhouse-connect](https://docs.myscale.com/)
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
            self.pgbar = lambda x: x
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = MyScaleSettings()
        assert self.config
        assert self.config.host and self.config.port
        assert (
            self.config.column_map
            and self.config.database
            and self.config.table
            and self.config.metric
        )
        for k in ["id", "vector", "text", "metadata"]:
            assert k in self.config.column_map
        assert self.config.metric.upper() in ["IP", "COSINE", "L2"]
        if self.config.metric in ["ip", "cosine", "l2"]:
            logger.warning(
                "Lower case metric types will be deprecated "
                "the future. Please use one of ('IP', 'Cosine', 'L2')"
            )

        # initialize the schema
        dim = len(embedding.embed_query("try this out"))

        index_params = (
            ", " + ",".join([f"'{k}={v}'" for k, v in self.config.index_param.items()])
            if self.config.index_param
            else ""
        )
        schema_ = f"""
            CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(
                {self.config.column_map['id']} String,
                {self.config.column_map['text']} String,
                {self.config.column_map['vector']} Array(Float32),
                {self.config.column_map['metadata']} JSON,
                CONSTRAINT cons_vec_len CHECK length(\
                    {self.config.column_map['vector']}) = {dim},
                VECTOR INDEX vidx {self.config.column_map['vector']} \
                    TYPE {self.config.index_type}(\
                        'metric_type={self.config.metric}'{index_params})
            ) ENGINE = MergeTree ORDER BY {self.config.column_map['id']}
        """
        self.dim = dim
        self.BS = "\\"
        self.must_escape = ("\\", "'")
        self._embeddings = embedding
        self.dist_order = (
            "ASC" if self.config.metric.upper() in ["COSINE", "L2"] else "DESC"
        )

        # Create a connection to myscale
        self.client = get_client(
            host=self.config.host,
            port=self.config.port,
            username=self.config.username,
            password=self.config.password,
            **kwargs,
        )
        self.client.command("SET allow_experimental_object_type=1")
        self.client.command(schema_)

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    def escape_str(self, value: str) -> str:
        return "".join(f"{self.BS}{c}" if c in self.must_escape else c for c in value)

    def _build_istr(self, transac: Iterable, column_names: Iterable[str]) -> str:
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
        _i_str = self._build_istr(transac, column_names)
        self.client.command(_i_str)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 32,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            ids: Optional list of ids to associate with the texts.
            batch_size: Batch size of insertion
            metadata: Optional column data to be inserted

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        # Embed and create the documents
        ids = ids or [sha1(t.encode("utf-8")).hexdigest() for t in texts]
        colmap_ = self.config.column_map

        transac = []
        column_names = {
            colmap_["id"]: ids,
            colmap_["text"]: texts,
            colmap_["vector"]: map(self._embeddings.embed_query, texts),
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
                assert len(v[keys.index(self.config.column_map["vector"])]) == self.dim
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
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        config: Optional[MyScaleSettings] = None,
        text_ids: Optional[Iterable[str]] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> MyScale:
        """Create Myscale wrapper with existing texts

        Args:
            texts (Iterable[str]): List or tuple of strings to be added
            embedding (Embeddings): Function to extract text embedding
            config (MyScaleSettings, Optional): Myscale configuration
            text_ids (Optional[Iterable], optional): IDs for the texts.
                                                     Defaults to None.
            batch_size (int, optional): Batchsize when transmitting data to MyScale.
                                        Defaults to 32.
            metadata (List[dict], optional): metadata to texts. Defaults to None.
            Other keyword arguments will pass into
                [clickhouse-connect](https://clickhouse.com/docs/en/integrations/python#clickhouse-connect-driver-api)
        Returns:
            MyScale Index
        """
        ctx = cls(embedding, config, **kwargs)
        ctx.add_texts(texts, ids=text_ids, batch_size=batch_size, metadatas=metadatas)
        return ctx

    def __repr__(self) -> str:
        """Text representation for myscale, prints backends, username and schemas.
            Easy to use with `str(Myscale())`

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

    def _build_qstr(
        self, q_emb: List[float], topk: int, where_str: Optional[str] = None
    ) -> str:
        q_emb_str = ",".join(map(str, q_emb))
        if where_str:
            where_str = f"PREWHERE {where_str}"
        else:
            where_str = ""

        q_str = f"""
            SELECT {self.config.column_map['text']}, 
                {self.config.column_map['metadata']}, dist
            FROM {self.config.database}.{self.config.table}
            {where_str}
            ORDER BY distance({self.config.column_map['vector']}, [{q_emb_str}]) 
                AS dist {self.dist_order}
            LIMIT {topk}
            """
        return q_str

    def similarity_search(
        self, query: str, k: int = 4, where_str: Optional[str] = None, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search with MyScale

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
            self._embeddings.embed_query(query), k, where_str, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search with MyScale by vectors

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
        q_str = self._build_qstr(embedding, k, where_str)
        try:
            return [
                Document(
                    page_content=r[self.config.column_map["text"]],
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
        """Perform a similarity search with MyScale

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
            List[Document]: List of documents most similar to the query text
            and cosine distance in float for each.
            Lower score represents more similarity.
        """
        q_str = self._build_qstr(self._embeddings.embed_query(query), k, where_str)
        try:
            return [
                (
                    Document(
                        page_content=r[self.config.column_map["text"]],
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

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        assert not (
            ids is None and where_str is None
        ), "You need to specify where to be deleted! Either with `ids` or `where_str`"
        conds = []
        if ids:
            conds.extend([f"{self.config.column_map['id']} = '{id}'" for id in ids])
        if where_str:
            conds.append(where_str)
        assert len(conds) > 0
        where_str_final = " AND ".join(conds)
        qstr = (
            f"DELETE FROM {self.config.database}.{self.config.table} "
            f"WHERE {where_str_final}"
        )
        try:
            self.client.command(qstr)
            return True
        except Exception as e:
            logger.error(str(e))
            return False

    @property
    def metadata_column(self) -> str:
        return self.config.column_map["metadata"]


class MyScaleWithoutJSON(MyScale):
    """MyScale vector store without metadata column

    This is super handy if you are working to a SQL-native table
    """

    def __init__(
        self,
        embedding: Embeddings,
        config: Optional[MyScaleSettings] = None,
        must_have_cols: List[str] = [],
        **kwargs: Any,
    ) -> None:
        """Building a myscale vector store without metadata column

        embedding (Embeddings): embedding model
        config (MyScaleSettings): Configuration to MyScale Client
        must_have_cols (List[str]): column names to be included in query
        Other keyword arguments will pass into
            [clickhouse-connect](https://docs.myscale.com/)
        """
        super().__init__(embedding, config, **kwargs)
        self.must_have_cols: List[str] = must_have_cols

    def _build_qstr(
        self, q_emb: List[float], topk: int, where_str: Optional[str] = None
    ) -> str:
        q_emb_str = ",".join(map(str, q_emb))
        if where_str:
            where_str = f"PREWHERE {where_str}"
        else:
            where_str = ""

        q_str = f"""
            SELECT {self.config.column_map['text']}, dist, 
                {','.join(self.must_have_cols)}
            FROM {self.config.database}.{self.config.table}
            {where_str}
            ORDER BY distance({self.config.column_map['vector']}, [{q_emb_str}]) 
                AS dist {self.dist_order}
            LIMIT {topk}
            """
        return q_str

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search with MyScale by vectors

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
        q_str = self._build_qstr(embedding, k, where_str)
        try:
            return [
                Document(
                    page_content=r[self.config.column_map["text"]],
                    metadata={k: r[k] for k in self.must_have_cols},
                )
                for r in self.client.query(q_str).named_results()
            ]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, where_str: Optional[str] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with MyScale

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
            List[Document]: List of documents most similar to the query text
            and cosine distance in float for each.
            Lower score represents more similarity.
        """
        q_str = self._build_qstr(self._embeddings.embed_query(query), k, where_str)
        try:
            return [
                (
                    Document(
                        page_content=r[self.config.column_map["text"]],
                        metadata={k: r[k] for k in self.must_have_cols},
                    ),
                    r["dist"],
                )
                for r in self.client.query(q_str).named_results()
            ]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

    @property
    def metadata_column(self) -> str:
        return ""
