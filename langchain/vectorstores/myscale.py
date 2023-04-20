"""Wrapper around MyScale vector database."""
from __future__ import annotations

import json
import logging 
from os import getenv
from pydantic import BaseSettings
from hashlib import sha1
from typing import List, Any, Optional, Iterable, Dict, Tuple
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
import sqlparse
import tqdm
from threading import Thread

logger = logging.getLogger()

def clean_parsed(parsed):
    return [p for p in parsed
            if p.is_keyword or
            type(p) in [sqlparse.sql.Parenthesis, sqlparse.sql.Identifier]]


def has_mul_sub_str(s, *args):
    for a in args:
        if a not in s:
            return False
    return True


class MyScaleSettings(BaseSettings):
    """MyScale Client Configuration

    Attribute:
        myscale_host (str) : An URL to connect to MyScale backend. Defaults to 'localhost'.
        myscale_port (int) : URL port to connect with HTTP. Defaults to 8123.
        username (str) : Usernamed to login. Defaults to None.
        password (str) : Password to login. Defaults to None.
        index_type (str): index type string
        index_param (dict): index build parameter 
        database (str) : Database name to find the table. Defaults to 'default'.
        table (str) : Table name to operate on. Defaults to 'vector_table'.
        metric (str) : Metric to compute distance, supported are ('l2', 'cosine', 'ip'). Defaults to 'cosine'.
        column_map (Dict) : Column type map to given schema. Must set if schema is not None.
                            Must have keys: `text`, `id`, `vector`, must be same size to number of columns.
                            For example
                            ```python
                            {
                                'id': 'text_id',
                                'vector': 'text_embedding',
                                'text': 'text_plain',
                                'metadata': 'metadata_dictionary_in_json',
                            }
                            ```
                            Defaults to identity map.

    Returns:
        _type_: _description_
    """
    host: str = "localhost"
    port: int = 8123

    username: str = None
    password: str = None

    index_type: str = 'IVFFLAT'
    index_param: Dict[str, str] = None

    column_map: Dict[str, str] = {
        'id': 'id',
        'text': 'text',
        'vector': 'vector',
        'metadata': 'metadata'
    }

    database: str = 'default'
    table: str = 'langchain'
    metric: str = 'cosine'

    def __getitem__(self, item):
        return getattr(self, item)

    class Config:
        env_file = ".env"
        env_prefix = 'myscale_'
        env_file_encoding = "utf-8"


class MyScale(VectorStore):
    """Wrapper around MyScale vector database

    You need a `clickhouse-connect` python package, and a valid account to connect to MyScale.

    MyScale can not only search with simple vector indexes, it also supports complex query
    with multiple conditions, constraints and even sub-queries.
    For more information, please visit [myscale official site](https://docs.myscale.com/en/overview/)
    """

    def __init__(self,
                 embedding_function: Embeddings,
                 config: MyScaleSettings = None,
                 **kwargs) -> None:
        """MyScale Wrapper to LangChain

            embedding_function (Embeddings): 
            config (MytScaleSettings): Configuration to MyScale Client
            overwrite_table (bool, optional): If the table exists, will overwrite your existing table
            Other keyword arguments will pass into [clickhouse-connect](https://clickhouse.com/docs/en/integrations/python#clickhouse-connect-driver-api)
        """
        try:
            from clickhouse_connect import get_client
        except ImportError:
            raise ValueError(
                "Could not import clickhouse connect python package. "
                "Please install it with `pip install clickhouse-connect`."
            )
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = MyScaleSettings()
        assert self.config
        assert self.config.host and self.config.port
        assert self.config.column_map and self.config.database and self.config.table and self.config.metric
        for k in ['id', 'vector', 'text', 'metadata']:
            assert k in self.config.column_map
        assert self.config.metric in ['ip', 'cosine', 'l2']

        # initialize the schema
        dim = len(embedding_function('try this out'))

        index_params = ', ' + ','.join([f"'{k}={v}'" for k, v in self.config.index_param.items()]) if self.config.index_param else ''
        schema_ = f"""
            CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(
                {self.config.column_map['id']} String,
                {self.config.column_map['text']} String,
                {self.config.column_map['vector']} Array(Float32),
                {self.config.column_map['metadata']} JSON,
                CONSTRAINT cons_vec_len CHECK length({self.config.column_map['vector']}) = {dim},
                VECTOR INDEX vidx {self.config.column_map['vector']} TYPE {self.config.index_type}('metric_type={self.config.metric}'{index_params})
            ) ENGINE = MergeTree ORDER BY {self.config.column_map['id']}
        """
        self.dim = dim
        self.embedding_function = embedding_function

        # Create a connection to myscale
        self.client = get_client(host=self.config.host,
                                 port=self.config.port,
                                 username=self.config.username,
                                 password=self.config.password,
                                 **kwargs)
        self.client.command('SET allow_experimental_object_type=1')
        self.client.command(schema_)
    
    def _build_istr(self, transac, column_names):
        ks = ','.join(column_names)
        _data = []
        for n in transac:
            n = ','.join([f"'{str(_n)}'" for _n in n])
            _data.append(f"({n})")
        i_str = f'''
                INSERT INTO TABLE 
                    {self.config.database}.{self.config.table}({ks})
                VALUES
                {','.join(_data)}
                '''
        return i_str

    def _insert(self, transac, column_names):
        _i_str = self._build_istr(transac, column_names)
        self.client.command(_i_str)

    def add_texts(
        self,
        texts: Iterable[str],
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        metadatas: List[dict] = None,
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
        ids = ids or [sha1(t.encode('utf-8')).hexdigest() for t in texts]
        colmap_ = self.config.column_map

        transac = []
        column_names = {colmap_['id']: ids,
                        colmap_['text']: texts,
                        colmap_['vector']: map(self.embedding_function, texts)}
        metadatas = metadatas or map(lambda x: {}, range(len(texts)))
        column_names[colmap_['metadata']] = map(json.dumps, metadatas)
        assert len(set(colmap_)-set(column_names)) >= 0
        keys, values = zip(*column_names.items())
        try:
            t = None
            for v in tqdm.tqdm(zip(*values), desc='Inserting data...', total=len(texts)):
                assert len(v[keys.index(self.config.column_map['vector'])]) == self.dim
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
            return ids
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            raise e

    @classmethod
    def from_texts(
        cls: MyScale,
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: List[dict] = None,
        config: MyScaleSettings = None,
        text_ids: Optional[Iterable] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> MyScale:
        """Create Myscale wrapper with existing texts

        Args:
            embedding_function (Embeddings): Function to extract text embedding
            texts (Iterable[str]): List or tuple of strings to be added
            config (MyScaleSettings, Optional): Myscale configuration
            text_ids (Optional[Iterable], optional): IDs for the texts. Defaults to None.
            batch_size (int, optional): Batchsize when transmitting data to MyScale. Defaults to 32.
            metadata (List[dict], optional): metadata to texts. Defaults to None.
            Other keyword arguments will pass into [clickhouse-connect](https://clickhouse.com/docs/en/integrations/python#clickhouse-connect-driver-api)
        Returns:
            MyScale: _description_
        """
        ctx = cls(embedding.embed_query, config, **kwargs)
        ctx.add_texts(texts, ids=text_ids, batch_size=batch_size, metadatas=metadatas)
        return ctx

    def __repr__(self):
        """Text representation for myscale. easy to use with `str(Myscale())`

        Returns:
            _type_: _description_
        """
        _repr = f'\033[92m\033[1m{self.config.database}.{self.config.table} @ {self.config.host}:{self.config.port}\033[0m\n\n'
        _repr += f'\033[1musername: {self.config.username}\033[0m\n\nTable Schema:\n'
        _repr += '-' * 51 + '\n'
        for r in self.client.query(f'DESC {self.config.database}.{self.config.table}').named_results():
            _repr += f"|\033[94m{r['name']:24s}\033[0m|\033[96m{r['type']:24s}\033[0m|\n"
        _repr += '-' * 51 + '\n'
        return _repr

    def _build_qstr(self, q_emb: List[float], topk: int, where_str: Optional[str] = None) -> str:
        q_emb = ','.join(map(str, q_emb))
        if where_str:
            where_str = f"WHERE {where_str}"
        else:
            where_str = ''

        q_str = f"""
            SELECT {self.config.column_map['text']}, {self.config.column_map['metadata']}, dist
            FROM {self.config.database}.{self.config.table}
            {where_str}
            ORDER BY distance({self.config.column_map['vector']}, [{q_emb}]) AS dist
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
            where_str (Optional[str], optional): where condition string. Defaults to None.

            NOTE: Please do not let end-user to fill this out and always be aware of SQL injection.
                  When dealing with metadatas, remeber to use `{metadata-name-you-set}.attribute` 
                  instead of `attribute` alone. The default name for metadat column is `metadata`.

        Returns:
            List[Document]: List of Documents
        """
        return self.similarity_search_by_vector(self.embedding_function(query), k, where_str, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, where_str: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search with MyScale by vectors

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): where condition string. Defaults to None.

            NOTE: Please do not let end-user to fill this out and always be aware of SQL injection.
                  When dealing with metadatas, remeber to use `{metadata-name-you-set}.attribute` 
                  instead of `attribute` alone. The default name for metadat column is `metadata`.

        Returns:
            List[Document]: List of (Document, similarity)
        """
        q_str = self._build_qstr(embedding, k, where_str)
        try:
            return [Document(page_content=r[self.config.column_map['text']],
                            metadata=r[self.config.column_map['metadata']])
                    for r in self.client.query(q_str).named_results()]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            raise e
        

    def similarity_search_with_relevance_scores(self,
                                                query: str,
                                                k: int = 4,
                                                where_str: Optional[str] = None,
                                                **kwargs: Any) -> List[Tuple[Document, float]]:
        """Perform a similarity search with MyScale

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): where condition string. Defaults to None.

            NOTE: Please do not let end-user to fill this out and always be aware of SQL injection.
                  When dealing with metadatas, remeber to use `{metadata-name-you-set}.attribute` 
                  instead of `attribute` alone. The default name for metadat column is `metadata`.

        Returns:
            List[Document]: List of documents
        """
        q_str = self._build_qstr(self.embedding_function(query), k, where_str)
        try:
            return [(Document(page_content=r[self.config.column_map['text']],
                            metadata=r[self.config.column_map['metadata']]),
                    r['dist'])
                    for r in self.client.query(q_str).named_results()]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            raise e

    
    def drop(self):
        """ 
        Helper function: Drop data in the index table
        """
        self.client.command(f"DROP TABLE IF EXISTS {self.config.database}.{self.config.table}")
    
    @property
    def metadata_column(self):
        return self.config.column_map['metadata']