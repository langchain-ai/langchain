"""Wrapper around MyScale vector database."""
from __future__ import annotations

import json
from pydantic import BaseSettings
from hashlib import sha1
from typing import List, Any, Optional, Iterable, Dict, Tuple
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
import sqlparse
import tqdm
from threading import Thread


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
        database (str) : Database name to find the table. Defaults to 'default'.
        table (str) : Table name to operate on. Defaults to 'vector_table'.
        metric (str) : Metric to compute distance, supported are ('l2', 'cosine', 'ip'). Defaults to 'cosine'.
        column_map (Dict) : Column type map to given schema. Must set if schema is not None.
                            Must have keys: `text`, `id`, `vector`, must be same size to number of columns.
                            Other key type could be arbitary. For example
                            ```python
                            {
                                'id': 'text_id',
                                'vector': 'text_embedding',
                                'text': 'text_plain',
                                'metadata': 'metadata_dictionary_in_json',
                                'this-can-be-any-type': 'a-column-of-meow-cats'
                                ...
                            }
                            ```
                            Defaults to identity map.

    Returns:
        _type_: _description_
    """
    myscale_host: str = "localhost"
    myscale_port: int = 8123

    username: str = None
    password: str = None

    index_type: str = 'IVFFLAT'

    column_map: Dict[str, str] = {
        'id': 'id',
        'text': 'text',
        'vector': 'vector',
        'metadata': 'metadata'
    }

    database: str = 'default'
    table: str = 'vector_table'
    metric: str = 'cosine'

    def __getitem__(self, item):
        return getattr(self, item)

    class Config:
        env_file = ".env"
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
                 config: MyScaleSettings = MyScaleSettings(),
                 create_table: bool = True) -> None:
        """MyScale Wrapper to LangChain

            embedding_function (Embeddings): 
            config (MytScaleSettings): Configuration to MyScale Client
            create_table (bool, optional): Create table if you want. Defaults to True.

        """
        try:
            from clickhouse_connect import get_client
        except ImportError:
            raise ValueError(
                "Could not import clickhouse connect python package. "
                "Please install it with `pip install clickhouse-connect`."
            )
        super().__init__()
        self.config = config
        assert self.config.myscale_host and self.config.myscale_port
        assert self.config.column_map and self.config.database and self.config.table and self.config.metric
        for k in ['id', 'vector', 'text', 'metadata']:
            assert k in self.config.column_map
        # FIXME @ fangruil: this should be myscale syntax
        assert self.config.metric in ['l2Distance', 'cosineDistance']

        # initialize the schema
        dim = len(embedding_function('try this out'))

        # FIXME @ fangruil: this should be myscale syntax
        # VECTOR INDEX vec_idx {self.config.column_map['vector']} TYPE {self.config.index_type}('metric_type={self.config.metric}') GRANULARITY 1,
        schema_ = f"""
            CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(
                {self.config.column_map['id']} String,
                {self.config.column_map['text']} String,
                {self.config.column_map['vector']} Array(Float32),
                {self.config.column_map['metadata']} String,
                CONSTRAINT cons_vec_len CHECK length({self.config.column_map['vector']}) = {dim},
                INDEX vidx {self.config.column_map['vector']} TYPE annoy(10, 'cosineDistance') GRANULARITY 1,
            ) ENGINE = MergeTree ORDER BY {self.config.column_map['id']}
        """
        self.config = config
        self.dim = dim
        self.embedding_function = embedding_function

        # Create a connection to myscale
        self.client = get_client(host=self.config.myscale_host,
                                 port=self.config.myscale_port,
                                 username=self.config.username,
                                 password=self.config.password)
        if create_table:
            # FIXME @ fangruil: this should be our ann benchmark switch
            self.client.command('SET allow_experimental_annoy_index=1')
            self.client.command(schema_)

    def add_texts(
        self,
        texts: Iterable[str],
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        metadata: List[dict] = None,
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
                        colmap_['vector']: map(self.embedding_function, texts), }
        if metadata:
            column_names[colmap_['metadata']] = map(json.dumps, metadata)
        assert len(set(colmap_)-set(column_names)) >= 0
        keys, values = zip(*column_names.items())
        t = None
        for v in tqdm.tqdm(zip(*values), total=len(texts)):
            transac.append(v)
            if len(transac) == batch_size:
                if t:
                    t.join()
                t = Thread(target=self.client.insert,
                           args=[self.config.table, transac],
                           kwargs={'column_names': keys,
                                   'database': self.config.database})
                t.start()
                transac = []
        if len(transac) > 0:
            if t:
                t.join()
            self.client.insert(self.config.table, transac,
                               column_names=keys, database=self.config.database)
        return ids

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        embedding_function: Embeddings,
        config: MyScaleSettings = MyScaleSettings(),
        text_ids: Optional[Iterable] = None,
        batch_size: int = 32,
        metadata: List[dict] = None,
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

        Returns:
            MyScale: _description_
        """
        ctx = cls(embedding_function,
                  config)
        ctx.add_texts(texts, ids=text_ids, batch_size=batch_size, metadata=metadata)
        return ctx

    def __repr__(self):
        """Text representation for myscale. easy to use with `str(Myscale())`

        Returns:
            _type_: _description_
        """
        _repr = f'\033[92m\033[1m{self.config.database}.{self.config.table} @ {self.config.myscale_host}:{self.config.myscale_port}\033[0m\n\n'
        _repr += f'\033[1musername: {self.config.username}\033[0m\n\n'
        _repr += '-' * 51 + '\n'
        for r in i.client.query(f'DESC {i.config.database}.{i.config.table}').named_results():
            _repr += f"|\033[94m{r['name']:24s}\033[0m|\033[96m{r['type']:24s}\033[0m|\n"
        _repr += '-' * 51 + '\n'
        return _repr

    def _build_qstr(self, q_emb: List[float], topk: int, where_condition: str) -> str:
        q_emb = ','.join(map(str, q_emb))

        # FIXME @ fangruil: this should be myscale distance function
        q_str = f"""
            SELECT {self.config.column_map['text']}, {self.config.column_map['metadata']}, dist
            FROM {self.config.database}.{self.config.table}
            {where_condition}
            ORDER BY cosineDistance({self.config.column_map['vector']}, [{q_emb}]) AS dist
            LIMIT {topk}
            """
        return q_str

    def similarity_search(
        self, query: str, k: int = 4, where_condition: Optional[str] = None, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search with MyScale

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_condition (Optional[str], optional): where condition string. Defaults to None.

            NOTE: `where_condition` is a hacked string. Please do not let end-user to fill this out.
                   Please be aware of SQL injection.

        Returns:
            List[Document]: List of Documents
        """
        return self.similarity_search_by_vector(self.embedding_function(query), k, where_condition, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, where_condition: Optional[str] = None, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search with MyScale by vectors

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_condition (Optional[str], optional): where condition string. Defaults to None.

            NOTE: `where_condition` is a hacked string. Please do not let end-user to fill this out.
                   Please be aware of SQL injection.

        Returns:
            List[Document]: List of (Document, similarity)
        """
        if not where_condition:
            where_condition = ""
        q_str = self._build_qstr(embedding, k, where_condition)
        return [Document(page_content=r[self.config.column_map['text']],
                         metadata=json.loads(r[self.config.column_map['metadata']])
                         if type(r[self.config.column_map['metadata']]) is str and
                         len((r[self.config.column_map['metadata']])) > 0 else {})
                for r in self.client.query(q_str).named_results()]

    def similarity_search_with_relevance_scores(self,
                                                query: str,
                                                k: int = 4,
                                                where_condition: Optional[str] = None,
                                                **kwargs: Any) -> List[Tuple[Document, float]]:
        """Perform a similarity search with MyScale

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_condition (Optional[str], optional): where condition string. Defaults to None.

            NOTE: `where_condition` is a hacked string. Please do not let end-user to fill this out.
                   Please be aware of SQL injection.

        Returns:
            List[Document]: List of documents
        """
        if not where_condition:
            where_condition = ""
        q_str = self._build_qstr(self.embedding_function(query), k, where_condition)
        return [(Document(page_content=r[self.config.column_map['text']],
                          metadata=json.loads(r[self.config.column_map['metadata']])
                          if type(r[self.config.column_map['metadata']]) is str and
                          len((r[self.config.column_map['metadata']])) > 0 else {}),
                 r['dist'])
                for r in self.client.query(q_str).named_results()]


if __name__ == '__main__':
    import string
    import random

    config = MyScaleSettings()
    config.metric = 'cosineDistance'
    config.index_type = 'annoy'
    config.column_map = {
        'id': 'text_id',
        'text': 'text_plain',
        'vector': 'text_feature',
        'metadata': 'text_metadata'
    }
    config.myscale_port = 8124
    i = MyScale.from_texts(lambda x: list([random.random() for _ in range(512)]), config=config,
                           texts=[''.join(random.choices(
                               string.ascii_uppercase + string.digits, k=1000)) for _ in range(1000)],
                           #    metadata=[{'which': 'who'} for _ in range(1000)]
                           )
    i.similarity_search('where is your daddy?')
    i.similarity_search_by_vector(list([random.random() for _ in range(512)]))
    i.similarity_search_with_relevance_scores('where is your daddy?')
    print(str(i))
    i.client.command(f'DROP TABLE IF EXISTS {i.config.database}.{i.config.table}')
