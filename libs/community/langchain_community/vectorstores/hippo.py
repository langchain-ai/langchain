from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from transwarp_hippo_api.hippo_client import HippoClient

# Default connection
DEFAULT_HIPPO_CONNECTION = {
    "host": "localhost",
    "port": "7788",
    "username": "admin",
    "password": "admin",
}

logger = logging.getLogger(__name__)


class Hippo(VectorStore):
    """`Hippo` vector store.

    You need to install `hippo-api` and run Hippo.

    Please visit our official website for how to run a Hippo instance:
    https://www.transwarp.cn/starwarp

    Args:
        embedding_function (Embeddings): Function used to embed the text.
        table_name (str): Which Hippo table to use. Defaults to
            "test".
        database_name (str): Which Hippo database to use. Defaults to
            "default".
        number_of_shards (int): The number of shards for the Hippo table.Defaults to
            1.
        number_of_replicas (int): The number of replicas for the Hippo table.Defaults to
            1.
        connection_args (Optional[dict[str, any]]): The connection args used for
            this class comes in the form of a dict.
        index_params (Optional[dict]): Which index params to use. Defaults to
            IVF_FLAT.
        drop_old (Optional[bool]): Whether to drop the current collection. Defaults
            to False.
        primary_field (str): Name of the primary key field. Defaults to "pk".
        text_field (str): Name of the text field. Defaults to "text".
        vector_field (str): Name of the vector field. Defaults to "vector".

    The connection args used for this class comes in the form of a dict,
    here are a few of the options:
        host (str): The host of Hippo instance. Default at "localhost".
        port (str/int): The port of Hippo instance. Default at 7788.
        user (str): Use which user to connect to Hippo instance. If user and
            password are provided, we will add related header in every RPC call.
        password (str): Required when user is provided. The password
            corresponding to the user.

    Example:
        .. code-block:: python

        from langchain_community.vectorstores import Hippo
        from langchain_community.embeddings import OpenAIEmbeddings

        embedding = OpenAIEmbeddings()
        # Connect to a hippo instance on localhost
        vector_store = Hippo.from_documents(
            docs,
            embedding=embeddings,
            table_name="langchain_test",
            connection_args=HIPPO_CONNECTION
        )

    Raises:
        ValueError: If the hippo-api python package is not installed.
    """

    def __init__(
        self,
        embedding_function: Embeddings,
        table_name: str = "test",
        database_name: str = "default",
        number_of_shards: int = 1,
        number_of_replicas: int = 1,
        connection_args: Optional[Dict[str, Any]] = None,
        index_params: Optional[dict] = None,
        drop_old: Optional[bool] = False,
    ):
        self.number_of_shards = number_of_shards
        self.number_of_replicas = number_of_replicas
        self.embedding_func = embedding_function
        self.table_name = table_name
        self.database_name = database_name
        self.index_params = index_params

        # In order for a collection to be compatible,
        # 'pk' should be an auto-increment primary key and string
        self._primary_field = "pk"
        # In order for compatibility, the text field will need to be called "text"
        self._text_field = "text"
        # In order for compatibility, the vector field needs to be called "vector"
        self._vector_field = "vector"
        self.fields: List[str] = []
        # Create the connection to the server
        if connection_args is None:
            connection_args = DEFAULT_HIPPO_CONNECTION
        self.hc = self._create_connection_alias(connection_args)
        self.col: Any = None

        # If the collection exists, delete it
        try:
            if (
                self.hc.check_table_exists(self.table_name, self.database_name)
                and drop_old
            ):
                self.hc.delete_table(self.table_name, self.database_name)
        except Exception as e:
            logging.error(
                f"An error occurred while deleting the table {self.table_name}: {e}"
            )
            raise

        try:
            if self.hc.check_table_exists(self.table_name, self.database_name):
                self.col = self.hc.get_table(self.table_name, self.database_name)
        except Exception as e:
            logging.error(
                f"An error occurred while getting the table {self.table_name}: {e}"
            )
            raise

        # Initialize the vector database
        self._get_env()

    def _create_connection_alias(self, connection_args: dict) -> HippoClient:
        """Create the connection to the Hippo server."""
        # Grab the connection arguments that are used for checking existing connection
        try:
            from transwarp_hippo_api.hippo_client import HippoClient
        except ImportError as e:
            raise ImportError(
                "Unable to import transwarp_hipp_api, please install with "
                "`pip install hippo-api`."
            ) from e

        host: str = connection_args.get("host", None)
        port: int = connection_args.get("port", None)
        username: str = connection_args.get("username", "shiva")
        password: str = connection_args.get("password", "shiva")

        # Order of use is host/port, uri, address
        if host is not None and port is not None:
            if "," in host:
                hosts = host.split(",")
                given_address = ",".join([f"{h}:{port}" for h in hosts])
            else:
                given_address = str(host) + ":" + str(port)
        else:
            raise ValueError("Missing standard address type for reuse attempt")

        try:
            logger.info(f"create HippoClient[{given_address}]")
            return HippoClient([given_address], username=username, pwd=password)
        except Exception as e:
            logger.error("Failed to create new connection")
            raise e

    def _get_env(
        self, embeddings: Optional[list] = None, metadatas: Optional[List[dict]] = None
    ) -> None:
        logger.info("init ...")
        if embeddings is not None:
            logger.info("create collection")
            self._create_collection(embeddings, metadatas)
        self._extract_fields()
        self._create_index()

    def _create_collection(
        self, embeddings: list, metadatas: Optional[List[dict]] = None
    ) -> None:
        from transwarp_hippo_api.hippo_client import HippoField
        from transwarp_hippo_api.hippo_type import HippoType

        # Determine embedding dim
        dim = len(embeddings[0])
        logger.debug(f"[_create_collection] dim: {dim}")
        fields = []

        # Create the primary key field
        fields.append(HippoField(self._primary_field, True, HippoType.STRING))

        # Create the text field

        fields.append(HippoField(self._text_field, False, HippoType.STRING))

        # Create the vector field, supports binary or float vectors
        # to The binary vector type is to be developed.
        fields.append(
            HippoField(
                self._vector_field,
                False,
                HippoType.FLOAT_VECTOR,
                type_params={"dimension": dim},
            )
        )
        # to In Hippo,there is no method similar to the infer_type_data
        # types, so currently all non-vector data is converted to string type.

        if metadatas:
            #     # Create FieldSchema for each entry in metadata.
            for key, value in metadatas[0].items():
                #         # Infer the corresponding datatype of the metadata
                if isinstance(value, list):
                    value_dim = len(value)
                    fields.append(
                        HippoField(
                            key,
                            False,
                            HippoType.FLOAT_VECTOR,
                            type_params={"dimension": value_dim},
                        )
                    )
                else:
                    fields.append(HippoField(key, False, HippoType.STRING))

        logger.debug(f"[_create_collection] fields: {fields}")

        # Create the collection
        self.hc.create_table(
            name=self.table_name,
            auto_id=True,
            fields=fields,
            database_name=self.database_name,
            number_of_shards=self.number_of_shards,
            number_of_replicas=self.number_of_replicas,
        )
        self.col = self.hc.get_table(self.table_name, self.database_name)
        logger.info(
            f"[_create_collection] : "
            f"create table {self.table_name} in {self.database_name} successfully"
        )

    def _extract_fields(self) -> None:
        """Grab the existing fields from the Collection"""
        from transwarp_hippo_api.hippo_client import HippoTable

        if isinstance(self.col, HippoTable):
            schema = self.col.schema
            logger.debug(f"[_extract_fields] schema:{schema}")
            for x in schema:
                self.fields.append(x.name)
            logger.debug(f"04 [_extract_fields] fields:{self.fields}")

    # TO CAN: Translated into English, your statement would be: "Currently,
    # only the field named 'vector' (the automatically created vector field)
    # is checked for indexing. Indexes need to be created manually for other
    # vector type columns.
    def _get_index(self) -> Optional[Dict[str, Any]]:
        """Return the vector index information if it exists"""
        from transwarp_hippo_api.hippo_client import HippoTable

        if isinstance(self.col, HippoTable):
            table_info = self.hc.get_table_info(
                self.table_name, self.database_name
            ).get(self.table_name, {})
            embedding_indexes = table_info.get("embedding_indexes", None)
            if embedding_indexes is None:
                return None
            else:
                for x in self.hc.get_table_info(self.table_name, self.database_name)[
                    self.table_name
                ]["embedding_indexes"]:
                    logger.debug(f"[_get_index] embedding_indexes {embedding_indexes}")
                    if x["column"] == self._vector_field:
                        return x
        return None

    # TO Indexes can only be created for the self._vector_field field.
    def _create_index(self) -> None:
        """Create a index on the collection"""
        from transwarp_hippo_api.hippo_client import HippoTable
        from transwarp_hippo_api.hippo_type import IndexType, MetricType

        if isinstance(self.col, HippoTable) and self._get_index() is None:
            if self._get_index() is None:
                if self.index_params is None:
                    self.index_params = {
                        "index_name": "langchain_auto_create",
                        "metric_type": MetricType.L2,
                        "index_type": IndexType.IVF_FLAT,
                        "nlist": 10,
                    }

                    self.col.create_index(
                        self._vector_field,
                        self.index_params["index_name"],
                        self.index_params["index_type"],
                        self.index_params["metric_type"],
                        nlist=self.index_params["nlist"],
                    )
                    logger.debug(
                        self.col.activate_index(self.index_params["index_name"])
                    )
                    logger.info("create index successfully")
                else:
                    index_dict = {
                        "IVF_FLAT": IndexType.IVF_FLAT,
                        "FLAT": IndexType.FLAT,
                        "IVF_SQ": IndexType.IVF_SQ,
                        "IVF_PQ": IndexType.IVF_PQ,
                        "HNSW": IndexType.HNSW,
                    }

                    metric_dict = {
                        "ip": MetricType.IP,
                        "IP": MetricType.IP,
                        "l2": MetricType.L2,
                        "L2": MetricType.L2,
                    }
                    self.index_params["metric_type"] = metric_dict[
                        self.index_params["metric_type"]
                    ]

                    if self.index_params["index_type"] == "FLAT":
                        self.index_params["index_type"] = index_dict[
                            self.index_params["index_type"]
                        ]
                        self.col.create_index(
                            self._vector_field,
                            self.index_params["index_name"],
                            self.index_params["index_type"],
                            self.index_params["metric_type"],
                        )
                        logger.debug(
                            self.col.activate_index(self.index_params["index_name"])
                        )
                    elif (
                        self.index_params["index_type"] == "IVF_FLAT"
                        or self.index_params["index_type"] == "IVF_SQ"
                    ):
                        self.index_params["index_type"] = index_dict[
                            self.index_params["index_type"]
                        ]
                        self.col.create_index(
                            self._vector_field,
                            self.index_params["index_name"],
                            self.index_params["index_type"],
                            self.index_params["metric_type"],
                            nlist=self.index_params.get("nlist", 10),
                            nprobe=self.index_params.get("nprobe", 10),
                        )
                        logger.debug(
                            self.col.activate_index(self.index_params["index_name"])
                        )
                    elif self.index_params["index_type"] == "IVF_PQ":
                        self.index_params["index_type"] = index_dict[
                            self.index_params["index_type"]
                        ]
                        self.col.create_index(
                            self._vector_field,
                            self.index_params["index_name"],
                            self.index_params["index_type"],
                            self.index_params["metric_type"],
                            nlist=self.index_params.get("nlist", 10),
                            nprobe=self.index_params.get("nprobe", 10),
                            nbits=self.index_params.get("nbits", 8),
                            m=self.index_params.get("m"),
                        )
                        logger.debug(
                            self.col.activate_index(self.index_params["index_name"])
                        )
                    elif self.index_params["index_type"] == "HNSW":
                        self.index_params["index_type"] = index_dict[
                            self.index_params["index_type"]
                        ]
                        self.col.create_index(
                            self._vector_field,
                            self.index_params["index_name"],
                            self.index_params["index_type"],
                            self.index_params["metric_type"],
                            M=self.index_params.get("M"),
                            ef_construction=self.index_params.get("ef_construction"),
                            ef_search=self.index_params.get("ef_search"),
                        )
                        logger.debug(
                            self.col.activate_index(self.index_params["index_name"])
                        )
                    else:
                        raise ValueError(
                            "Index name does not match, "
                            "please enter the correct index name. "
                            "(FLAT, IVF_FLAT, IVF_PQ,IVF_SQ, HNSW)"
                        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        timeout: Optional[int] = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add text to the collection.

        Args:
            texts: An iterable that contains the text to be added.
            metadatas: An optional list of dictionaries,
            each dictionary contains the metadata associated with a text.
            timeout: Optional timeout, in seconds.
            batch_size: The number of texts inserted in each batch, defaults to 1000.
            **kwargs: Other optional parameters.

        Returns:
            A list of strings, containing the unique identifiers of the inserted texts.

        Note:
            If the collection has not yet been created,
            this method will create a new collection.
        """
        from transwarp_hippo_api.hippo_client import HippoTable

        if not texts or all(t == "" for t in texts):
            logger.debug("Nothing to insert, skipping.")
            return []
        texts = list(texts)

        logger.debug(f"[add_texts] texts: {texts}")

        try:
            embeddings = self.embedding_func.embed_documents(texts)
        except NotImplementedError:
            embeddings = [self.embedding_func.embed_query(x) for x in texts]

        if len(embeddings) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []

        logger.debug(f"[add_texts] len_embeddings:{len(embeddings)}")

        # 如果还没有创建collection则创建collection
        if not isinstance(self.col, HippoTable):
            self._get_env(embeddings, metadatas)

        # Dict to hold all insert columns
        insert_dict: Dict[str, list] = {
            self._text_field: texts,
            self._vector_field: embeddings,
        }
        logger.debug(f"[add_texts] metadatas:{metadatas}")
        logger.debug(f"[add_texts] fields:{self.fields}")
        if metadatas is not None:
            for d in metadatas:
                for key, value in d.items():
                    if key in self.fields:
                        insert_dict.setdefault(key, []).append(value)

        logger.debug(insert_dict[self._text_field])

        # Total insert count
        vectors: list = insert_dict[self._vector_field]
        total_count = len(vectors)

        if "pk" in self.fields:
            self.fields.remove("pk")

        logger.debug(f"[add_texts] total_count:{total_count}")
        for i in range(0, total_count, batch_size):
            # Grab end index
            end = min(i + batch_size, total_count)
            # Convert dict to list of lists batch for insertion
            insert_list = [insert_dict[x][i:end] for x in self.fields]
            try:
                res = self.col.insert_rows(insert_list)
                logger.info(f"05 [add_texts] insert {res}")
            except Exception as e:
                logger.error(
                    "Failed to insert batch starting at entity: %s/%s", i, total_count
                )
                raise e
        return [""]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform a similarity search on the query string.

        Args:
            query (str): The text to search for.
            k (int, optional): The number of results to return. Default is 4.
            param (dict, optional): Specifies the search parameters for the index.
            Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (int, optional): Time to wait before a timeout error.
            Defaults to None.
            kwargs: Keyword arguments for Collection.search().

        Returns:
            List[Document]: The document results of the search.
        """

        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        res = self.similarity_search_with_score(
            query=query, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in res]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Performs a search on the query string and returns results with scores.

        Args:
            query (str): The text being searched.
            k (int, optional): The number of results to return.
            Default is 4.
            param (dict): Specifies the search parameters for the index.
            Default is None.
            expr (str, optional): Filtering expression. Default is None.
            timeout (int, optional): The waiting time before a timeout error.
            Default is None.
            kwargs: Keyword arguments for Collection.search().

        Returns:
            List[float], List[Tuple[Document, any, any]]:
        """

        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        # Embed the query text.
        embedding = self.embedding_func.embed_query(query)

        ret = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return ret

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Performs a search on the query string and returns results with scores.

        Args:
            embedding (List[float]): The embedding vector being searched.
            k (int, optional): The number of results to return.
            Default is 4.
            param (dict): Specifies the search parameters for the index.
            Default is None.
            expr (str, optional): Filtering expression. Default is None.
            timeout (int, optional): The waiting time before a timeout error.
            Default is None.
            kwargs: Keyword arguments for Collection.search().

        Returns:
            List[Tuple[Document, float]]: Resulting documents and scores.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        # if param is None:
        #     param = self.search_params

        # Determine result metadata fields.
        output_fields = self.fields[:]
        output_fields.remove(self._vector_field)

        # Perform the search.
        logger.debug(f"search_field:{self._vector_field}")
        logger.debug(f"vectors:{[embedding]}")
        logger.debug(f"output_fields:{output_fields}")
        logger.debug(f"topk:{k}")
        logger.debug(f"dsl:{expr}")

        res = self.col.query(
            search_field=self._vector_field,
            vectors=[embedding],
            output_fields=output_fields,
            topk=k,
            dsl=expr,
        )
        # Organize results.
        logger.debug(f"[similarity_search_with_score_by_vector] res:{res}")
        score_col = self._text_field + "%scores"
        ret = []
        count = 0
        for items in zip(*[res[0][field] for field in output_fields]):
            meta = {field: value for field, value in zip(output_fields, items)}
            doc = Document(page_content=meta.pop(self._text_field), metadata=meta)
            logger.debug(
                f"[similarity_search_with_score_by_vector] "
                f"res[0][score_col]:{res[0][score_col]}"
            )
            score = res[0][score_col][count]
            count += 1
            ret.append((doc, score))

        return ret

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        table_name: str = "test",
        database_name: str = "default",
        connection_args: Dict[str, Any] = DEFAULT_HIPPO_CONNECTION,
        index_params: Optional[Dict[Any, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None,
        drop_old: bool = False,
        **kwargs: Any,
    ) -> "Hippo":
        """
        Creates an instance of the VST class from the given texts.

        Args:
            texts (List[str]): List of texts to be added.
            embedding (Embeddings): Embedding model for the texts.
            metadatas (List[dict], optional):
            List of metadata dictionaries for each text.Defaults to None.
            table_name (str): Name of the table. Defaults to "test".
            database_name (str): Name of the database. Defaults to "default".
            connection_args (dict[str, Any]): Connection parameters.
            Defaults to DEFAULT_HIPPO_CONNECTION.
            index_params (dict): Indexing parameters. Defaults to None.
            search_params (dict): Search parameters. Defaults to an empty dictionary.
            drop_old (bool): Whether to drop the old collection. Defaults to False.
            kwargs: Other arguments.

        Returns:
            Hippo: An instance of the VST class.
        """

        if search_params is None:
            search_params = {}
        logger.info("00 [from_texts] init the class of Hippo")
        vector_db = cls(
            embedding_function=embedding,
            table_name=table_name,
            database_name=database_name,
            connection_args=connection_args,
            index_params=index_params,
            drop_old=drop_old,
            **kwargs,
        )
        logger.debug(f"[from_texts] texts:{texts}")
        logger.debug(f"[from_texts] metadatas:{metadatas}")
        vector_db.add_texts(texts=texts, metadatas=metadatas)
        return vector_db
