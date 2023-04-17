"""Wrapper around MyScale vector database."""
from __future__ import annotations

from hashlib import sha1
from typing import List, Any, Optional, Iterable, Dict
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
import sqlparse
import tqdm
from threading import Thread

def clean_parsed(parsed):
    return [p for p in parsed 
            if p.is_keyword or \
                type(p) in [sqlparse.sql.Parenthesis, sqlparse.sql.Identifier]]    
    
    
def has_mul_sub_str(s, *args):
    for a in args:
        if a not in s:
            return False
    return True


class MyScale(VectorStore):
    """Wrapper around MyScale vector database
    
    You need a `clickhouse-connect` python package, and a valid account to connect to MyScale.
    
    MyScale can not only search with simple vector indexes, it also supports complex query
    with multiple conditions, constraints and even sub-queries.
    For more information, please visit [myscale official site](https://docs.myscale.com/en/overview/)
    
    This wrapper provides both the high level schema and a customizable one,
    Just in case if you have a custom table structure to search.
     
    If you want to customize it, just set both `schema` and `vector_args`
    If `schema` is None, then the vector index will fallback to simple high level schema
    
    *** ATTENTION ***: Please be careful here, only use this if you know what you are doing!
    - On advanced use: customized schema
        should be the `CREATE TABLE` clause you would like to use.
        NOTE: Assuming your schema have **constraints** on vectors
    """
    def __init__(self,
                 embedding_function: Embeddings,
                 host: str = 'localhost',
                 port: int = 8123,
                 username: str = None,
                 password: str = None,
                 # simple arguments
                 database: str = 'default',
                 table: str = 'vector_table',
                 metric: str = 'cosine',
                 # advanced use to myscale
                 schema: str = None,
                 schema_colmap: dict = None) -> None:
        """Initialize MyScale with wrapped functions
        
        It accepts structured params as well as a customized sql schema.
        Feel free to plugin any existing data to play with langchain

        Args:
            embedding_function (Embeddings): Function used to embed the text
            host (str, optional): An URL to connect to MyScale backend. Defaults to 'localhost'.
            port (int, optional): URL port to connect with HTTP. Defaults to 8123.
            username (str, optional): Usernamed to login. Defaults to None.
            password (str, optional): Password to login. Defaults to None.
            database (str, optional): Database name to find the table. Defaults to 'default'.
            table (str, optional): Table name to operate on. Defaults to 'vector_table'.
            metric (str, optional): Metric to compute distance, supported are ('l2', 'cosine', 'ip'). Defaults to 'cosine'.
            schema (str, optional): Customized schema to the table. 
                                    If set, parameters ('database', 'table', 'metric') will not be effective at all.
                                    Defaults to None.
            schema_colmap (dict, optional): Column type map to given schema. Must set if schema is not None.
                                    Must have keys: `text`, `id`, `vector`, must be same size to number of columns.
                                    Other key type could be arbitary. For example
                                    ```python
                                    {
                                        'id': 'text_id',
                                        'vector': 'text_embedding',
                                        'text': 'text_plain',
                                        'this-can-be-any-type': 'a-column-of-meow-cats'
                                        ...
                                    }
                                    ```
                                    Defaults to None.

        Raises:
            ValueError: 
                - When customized schema is used, if the format of `database.table` is not valid, then raise
            AssertionError: 
                - When customized schema is used, if params does not contain valid schema
                - When customized schema is used, if params does not contain any constraint, then raise
                - When customized schema is used, if model dim mismatches schema dim, then raise
        """
        try:
            from clickhouse_connect import get_client
        except ImportError:
            raise ValueError(
                "Could not import clickhouse connect python package. "
                "Please install it with `pip install clickhouse-connect`."
            )
        try:
            import sqlparse
        except ImportError:
            raise ValueError(
                "Could not import sqlparse python package. "
                "Please install it with `pip install sqlparse`."
            )
        super().__init__()
        assert table and database and host and port
        # initialize the schema
        dim = len(embedding_function('try this out'))
        
        if schema is not None:
            assert schema_colmap
            assert 'id' in schema_colmap and 'text' in schema_colmap and 'vector' in schema_colmap
            # if customized schema is given, parse it
            parsed = [i for i in  sqlparse.parse(schema) if i.get_type().upper() == 'CREATE']
            assert len(parsed) >= 1
            parsed = parsed[0]
            schema_ = str(parsed)
            parsed = clean_parsed(parsed)
            tid = str([p for p in parsed if type(p) is sqlparse.sql.Identifier][0]).split('.')
            if len(tid) == 1:
                database, table = 'default', tid[0]
            elif len(tid) == 2:
                database, table = tid
            else:
                raise ValueError(f'Invalid table identifier {tid}')
            params = None
            for p in parsed:
                if type(p) is sqlparse.sql.Parenthesis:
                    params = str(p).split(',')
                    # SAN CHECK
                    # If you have constraints inside
                    if sum([1 if has_mul_sub_str(str(_p).upper(), 'CONSTRAINT', 'CHECK', 'LENGTH') \
                        else 0 for _p in params]) < 1:
                        raise AssertionError('We can\'t find any constraints at all. For your own safety, add at least one constraint on vector length.')
                    # If you have correct dimension constraints
                    _cons_dim = int([_p for _p in params \
                        if has_mul_sub_str(str(_p).upper(), 'CONSTRAINT', 'CHECK', 'LENGTH(')][0].split('=')[-1])
                    if dim != _cons_dim:
                        raise AssertionError(f'Constraint dimension mismatch to the embedding model you provided, which are {_cons_dim} and {dim}')
                    vec_def = str(p)[str(p).find('INDEX'):].split(')')[0]
                    # TODO @ fangruil fix it to myscale grammar
                    vector_column_ = vec_def.split(' ')[2]
                    metric_ = vec_def.split(',')[-1].translate({ord('\''):'', ord(' '):''})
            if not params:
                raise AssertionError('We can\'t find a parameter for your table. Please check your SQL.')
            colmap_ = schema_colmap
            assert vector_column_ in [v for _, v in colmap_.items()]
        else:
            assert metric in ['cosine', 'l2', 'ip']
            vector_column_ = 'vector'
            metric_ = metric
            schema_ = f"""
                CREATE TABLE IF NOT EXISTS {database}.{table}(
                    id String,
                    text String,
                    vector Array(Float32),
                    CONSTRAINT cons_vec_len CHECK length(vector) = {dim},
                    VECTOR INDEX vec_idx vector TYPE IVFFLAT('metric_type={metric_}', 'ncentroids=1000')
                )
            """
            colmap_ = {
                'id': 'id',
                'text': 'text',
                'vector': 'vector'
            }
            
        self.metric = metric_
        self.database = database
        self.table = table
        self.vector_column = vector_column_
        self.colmap = colmap_
        self.dim = dim
        self.embedding_function = embedding_function
        # Create a connection to myscale
        self.client = get_client(host=host, 
                                 port=port, 
                                 username=username,
                                 password=password)
        self.client.command('SET allow_experimental_annoy_index=1')
        self.client.command(schema_)
        
        
    def add_texts(
        self,
        texts: Iterable[str],
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        colmap_override: dict = None,
        extra_columns: Dict[str, Iterable[Any]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            ids: Optional list of ids to associate with the texts.
            batch_size: Batch size of insertion
            extra_columns: Optional column data to be inserted
            colmap_override: Override the class column map

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        # Embed and create the documents
        ids = ids or [sha1(t.encode('utf-8')).hexdigest() for t in texts]
        if colmap_override:
            colmap_ = colmap_override
        else:
            colmap_ = self.colmap
        
        transac = []
        column_names = {colmap_['id']: ids, 
                        colmap_['text']: texts, 
                        colmap_['vector']: map(self.embedding_function, texts)}
        if extra_columns:
            for k, v in extra_columns.items():
                assert len(list(v)) == len(texts)
                column_names[k] = v
        assert len(set(colmap_)-set(column_names)) >= 0
        keys, values = zip(*column_names.items())
        t = None
        for v in tqdm.tqdm(zip(*values), total=len(texts)):
            transac.append(v)
            if len(transac) == batch_size:
                if t:
                    t.join()
                t = Thread(target=self.client.insert, 
                           args=[self.table, transac], 
                           kwargs={'column_names':keys,
                                   'database':self.database})
                t.start()
                transac = []
        if len(transac) > 0:
            if t:
                t.join()
            self.client.insert(self.table, transac, 
                               column_names=keys, database=self.database)
            
    @classmethod
    def from_texts(
        cls,
        embedding_function: Embeddings,
        texts: Iterable[str],
        host: str = 'localhost',
        port: int = 8123,
        username: str = None,
        password: str = None,
        database: str = 'default',
        table: str = 'vector_table',
        metric: str = 'cosine',
        schema: str = None,
        schema_colmap: dict = None,
        text_ids: Optional[Iterable] = None,
        batch_size: int = 32,
        colmap_override: dict = None,
        extra_columns: Dict[str, Iterable[Any]] = None,
        **kwargs: Any,
    ) -> MyScale:
        ctx = cls(embedding_function, 
                  host=host, 
                  port=port,
                  username=username,
                  password=password,
                  database=database,
                  table=table,
                  metric=metric,
                  schema=schema,
                  schema_colmap=schema_colmap)
        ctx.add_texts(texts, ids=text_ids, batch_size=batch_size,
                      colmap_override=colmap_override, 
                      extra_columns=extra_columns)
        return ctx
    
    
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        return super().similarity_search(query, k, **kwargs)
        
        

if __name__ == '__main__':
    import string
    import random
    i = MyScale.from_texts(
        lambda x: list([random.random() for _ in range(512)]), port=8124,
        schema="""
        CREATE TABLE IF NOT EXISTS default.vtable(
                text_id String,
                text_plain String,
                text_feature Array(Float32),
                text_author String,
                CONSTRAINT cons_vec_len CHECK length(text_feature) = 512,
                INDEX vidx text_feature TYPE annoy(10, 'cosineDistance') GRANULARITY 1,
            ) ENGINE = MergeTree ORDER BY text_id
        """,
        schema_colmap={
            'id': 'text_id',
            'text': 'text_plain',
            'vector': 'text_feature',
            'other': 'text_author',
        }, texts=[''.join(random.choices(string.ascii_uppercase + string.digits, k=1000)) for _ in range(1000)]
        )
    # i.add_texts([''.join(random.choices(string.ascii_uppercase + string.digits, k=1000)) for _ in range(1000)])
    i.client.command('DROP TABLE IF EXISTS default.vtable')