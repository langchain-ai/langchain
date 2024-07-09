from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)

DEFAULT_MILVUS_CONNECTION = {
    "host": "localhost",
    "port": "19530",
    "user": "",
    "password": "",
    "secure": False,
}


@deprecated(
    since="0.2.0",
    removal="0.3.0",
    alternative_import="langchain_milvus.MilvusVectorStore",
)
class Milvus(VectorStore):
    """`Milvus` vector store.

    You need to install `pymilvus` and run Milvus.

    See the following documentation for how to run a Milvus instance:
    https://milvus.io/docs/install_standalone-docker.md

    If looking for a hosted Milvus, take a look at this documentation:
    https://zilliz.com/cloud and make use of the Zilliz vectorstore found in
    this project.

    IF USING L2/IP metric, IT IS HIGHLY SUGGESTED TO NORMALIZE YOUR DATA.

    Args:
        embedding_function (Embeddings): Function used to embed the text.
        collection_name (str): Which Milvus collection to use. Defaults to
            "LangChainCollection".
        collection_description (str): The description of the collection. Defaults to
            "".
        collection_properties (Optional[dict[str, any]]): The collection properties.
            Defaults to None.
            If set, will override collection existing properties.
            For example: {"collection.ttl.seconds": 60}.
        connection_args (Optional[dict[str, any]]): The connection args used for
            this class comes in the form of a dict.
        consistency_level (str): The consistency level to use for a collection.
            Defaults to "Session".
        index_params (Optional[dict]): Which index params to use. Defaults to
            HNSW/AUTOINDEX depending on service.
        search_params (Optional[dict]): Which search params to use. Defaults to
            default of index.
        drop_old (Optional[bool]): Whether to drop the current collection. Defaults
            to False.
        auto_id (bool): Whether to enable auto id for primary key. Defaults to False.
            If False, you needs to provide text ids (string less than 65535 bytes).
            If True, Milvus will generate unique integers as primary keys.
        primary_field (str): Name of the primary key field. Defaults to "pk".
        text_field (str): Name of the text field. Defaults to "text".
        vector_field (str): Name of the vector field. Defaults to "vector".
        metadata_field (str): Name of the metadata field. Defaults to None.
            When metadata_field is specified,
            the document's metadata will store as json.

    The connection args used for this class comes in the form of a dict,
    here are a few of the options:
        address (str): The actual address of Milvus
            instance. Example address: "localhost:19530"
        uri (str): The uri of Milvus instance. Example uri:
            "http://randomwebsite:19530",
            "tcp:foobarsite:19530",
            "https://ok.s3.south.com:19530".
        host (str): The host of Milvus instance. Default at "localhost",
            PyMilvus will fill in the default host if only port is provided.
        port (str/int): The port of Milvus instance. Default at 19530, PyMilvus
            will fill in the default port if only host is provided.
        user (str): Use which user to connect to Milvus instance. If user and
            password are provided, we will add related header in every RPC call.
        password (str): Required when user is provided. The password
            corresponding to the user.
        secure (bool): Default is false. If set to true, tls will be enabled.
        client_key_path (str): If use tls two-way authentication, need to
            write the client.key path.
        client_pem_path (str): If use tls two-way authentication, need to
            write the client.pem path.
        ca_pem_path (str): If use tls two-way authentication, need to write
            the ca.pem path.
        server_pem_path (str): If use tls one-way authentication, need to
            write the server.pem path.
        server_name (str): If use tls, need to write the common name.

    Example:
        .. code-block:: python

        from langchain_community.vectorstores import Milvus
        from langchain_community.embeddings import OpenAIEmbeddings

        embedding = OpenAIEmbeddings()
        # Connect to a milvus instance on localhost
        milvus_store = Milvus(
            embedding_function = Embeddings,
            collection_name = "LangChainCollection",
            drop_old = True,
            auto_id = True
        )

    Raises:
        ValueError: If the pymilvus python package is not installed.
    """

    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "LangChainCollection",
        collection_description: str = "",
        collection_properties: Optional[dict[str, Any]] = None,
        connection_args: Optional[dict[str, Any]] = None,
        consistency_level: str = "Session",
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        drop_old: Optional[bool] = False,
        auto_id: bool = False,
        *,
        primary_field: str = "pk",
        text_field: str = "text",
        vector_field: str = "vector",
        metadata_field: Optional[str] = None,
        partition_key_field: Optional[str] = None,
        partition_names: Optional[list] = None,
        replica_number: int = 1,
        timeout: Optional[float] = None,
        num_shards: Optional[int] = None,
    ):
        """Initialize the Milvus vector store."""
        try:
            from pymilvus import Collection, utility
        except ImportError:
            raise ImportError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            )

        # Default search params when one is not provided.
        self.default_search_params = {
            "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "HNSW": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_FLAT": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_SQ": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_PQ": {"metric_type": "L2", "params": {"ef": 10}},
            "IVF_HNSW": {"metric_type": "L2", "params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"metric_type": "L2", "params": {"search_k": 10}},
            "SCANN": {"metric_type": "L2", "params": {"search_k": 10}},
            "AUTOINDEX": {"metric_type": "L2", "params": {}},
            "GPU_CAGRA": {
                "metric_type": "L2",
                "params": {
                    "itopk_size": 128,
                    "search_width": 4,
                    "min_iterations": 0,
                    "max_iterations": 0,
                    "team_size": 0,
                },
            },
            "GPU_IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "GPU_IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
        }

        self.embedding_func = embedding_function
        self.collection_name = collection_name
        self.collection_description = collection_description
        self.collection_properties = collection_properties
        self.index_params = index_params
        self.search_params = search_params
        self.consistency_level = consistency_level
        self.auto_id = auto_id

        # In order for a collection to be compatible, pk needs to be varchar
        self._primary_field = primary_field
        # In order for compatibility, the text field will need to be called "text"
        self._text_field = text_field
        # In order for compatibility, the vector field needs to be called "vector"
        self._vector_field = vector_field
        self._metadata_field = metadata_field
        self._partition_key_field = partition_key_field
        self.fields: list[str] = []
        self.partition_names = partition_names
        self.replica_number = replica_number
        self.timeout = timeout
        self.num_shards = num_shards

        # Create the connection to the server
        if connection_args is None:
            connection_args = DEFAULT_MILVUS_CONNECTION
        self.alias = self._create_connection_alias(connection_args)
        self.col: Optional[Collection] = None

        # Grab the existing collection if it exists
        if utility.has_collection(self.collection_name, using=self.alias):
            self.col = Collection(
                self.collection_name,
                using=self.alias,
            )
            if self.collection_properties is not None:
                self.col.set_properties(self.collection_properties)
        # If need to drop old, drop it
        if drop_old and isinstance(self.col, Collection):
            self.col.drop()
            self.col = None

        # Initialize the vector store
        self._init(
            partition_names=partition_names,
            replica_number=replica_number,
            timeout=timeout,
        )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_func

    def _create_connection_alias(self, connection_args: dict) -> str:
        """Create the connection to the Milvus server."""
        from pymilvus import MilvusException, connections

        # Grab the connection arguments that are used for checking existing connection
        host: str = connection_args.get("host", None)
        port: Union[str, int] = connection_args.get("port", None)
        address: str = connection_args.get("address", None)
        uri: str = connection_args.get("uri", None)
        user = connection_args.get("user", None)

        # Order of use is host/port, uri, address
        if host is not None and port is not None:
            given_address = str(host) + ":" + str(port)
        elif uri is not None:
            if uri.startswith("https://"):
                given_address = uri.split("https://")[1]
            elif uri.startswith("http://"):
                given_address = uri.split("http://")[1]
            else:
                logger.error("Invalid Milvus URI: %s", uri)
                raise ValueError("Invalid Milvus URI: %s", uri)
        elif address is not None:
            given_address = address
        else:
            given_address = None
            logger.debug("Missing standard address type for reuse attempt")

        # User defaults to empty string when getting connection info
        if user is not None:
            tmp_user = user
        else:
            tmp_user = ""

        # If a valid address was given, then check if a connection exists
        if given_address is not None:
            for con in connections.list_connections():
                addr = connections.get_connection_addr(con[0])
                if (
                    con[1]
                    and ("address" in addr)
                    and (addr["address"] == given_address)
                    and ("user" in addr)
                    and (addr["user"] == tmp_user)
                ):
                    logger.debug("Using previous connection: %s", con[0])
                    return con[0]

        # Generate a new connection if one doesn't exist
        alias = uuid4().hex
        try:
            connections.connect(alias=alias, **connection_args)
            logger.debug("Created new connection using: %s", alias)
            return alias
        except MilvusException as e:
            logger.error("Failed to create new connection using: %s", alias)
            raise e

    def _init(
        self,
        embeddings: Optional[list] = None,
        metadatas: Optional[list[dict]] = None,
        partition_names: Optional[list] = None,
        replica_number: int = 1,
        timeout: Optional[float] = None,
    ) -> None:
        if embeddings is not None:
            self._create_collection(embeddings, metadatas)
        self._extract_fields()
        self._create_index()
        self._create_search_params()
        self._load(
            partition_names=partition_names,
            replica_number=replica_number,
            timeout=timeout,
        )

    def _create_collection(
        self, embeddings: list, metadatas: Optional[list[dict]] = None
    ) -> None:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            MilvusException,
        )
        from pymilvus.orm.types import infer_dtype_bydata

        # Determine embedding dim
        dim = len(embeddings[0])
        fields = []
        if self._metadata_field is not None:
            fields.append(FieldSchema(self._metadata_field, DataType.JSON))
        else:
            # Determine metadata schema
            if metadatas:
                # Create FieldSchema for each entry in metadata.
                for key, value in metadatas[0].items():
                    # Infer the corresponding datatype of the metadata
                    dtype = infer_dtype_bydata(value)
                    # Datatype isn't compatible
                    if dtype == DataType.UNKNOWN or dtype == DataType.NONE:
                        logger.error(
                            (
                                "Failure to create collection, "
                                "unrecognized dtype for key: %s"
                            ),
                            key,
                        )
                        raise ValueError(f"Unrecognized datatype for {key}.")
                    # Dataype is a string/varchar equivalent
                    elif dtype == DataType.VARCHAR:
                        fields.append(
                            FieldSchema(key, DataType.VARCHAR, max_length=65_535)
                        )
                    else:
                        fields.append(FieldSchema(key, dtype))

        # Create the text field
        fields.append(
            FieldSchema(self._text_field, DataType.VARCHAR, max_length=65_535)
        )
        # Create the primary key field
        if self.auto_id:
            fields.append(
                FieldSchema(
                    self._primary_field, DataType.INT64, is_primary=True, auto_id=True
                )
            )
        else:
            fields.append(
                FieldSchema(
                    self._primary_field,
                    DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=65_535,
                )
            )
        # Create the vector field, supports binary or float vectors
        fields.append(
            FieldSchema(self._vector_field, infer_dtype_bydata(embeddings[0]), dim=dim)
        )

        # Create the schema for the collection
        schema = CollectionSchema(
            fields,
            description=self.collection_description,
            partition_key_field=self._partition_key_field,
        )

        # Create the collection
        try:
            if self.num_shards is not None:
                # Issue with defaults:
                # https://github.com/milvus-io/pymilvus/blob/59bf5e811ad56e20946559317fed855330758d9c/pymilvus/client/prepare.py#L82-L85
                self.col = Collection(
                    name=self.collection_name,
                    schema=schema,
                    consistency_level=self.consistency_level,
                    using=self.alias,
                    num_shards=self.num_shards,
                )
            else:
                self.col = Collection(
                    name=self.collection_name,
                    schema=schema,
                    consistency_level=self.consistency_level,
                    using=self.alias,
                )
            # Set the collection properties if they exist
            if self.collection_properties is not None:
                self.col.set_properties(self.collection_properties)
        except MilvusException as e:
            logger.error(
                "Failed to create collection: %s error: %s", self.collection_name, e
            )
            raise e

    def _extract_fields(self) -> None:
        """Grab the existing fields from the Collection"""
        from pymilvus import Collection

        if isinstance(self.col, Collection):
            schema = self.col.schema
            for x in schema.fields:
                self.fields.append(x.name)

    def _get_index(self) -> Optional[dict[str, Any]]:
        """Return the vector index information if it exists"""
        from pymilvus import Collection

        if isinstance(self.col, Collection):
            for x in self.col.indexes:
                if x.field_name == self._vector_field:
                    return x.to_dict()
        return None

    def _create_index(self) -> None:
        """Create a index on the collection"""
        from pymilvus import Collection, MilvusException

        if isinstance(self.col, Collection) and self._get_index() is None:
            try:
                # If no index params, use a default HNSW based one
                if self.index_params is None:
                    self.index_params = {
                        "metric_type": "L2",
                        "index_type": "HNSW",
                        "params": {"M": 8, "efConstruction": 64},
                    }

                try:
                    self.col.create_index(
                        self._vector_field,
                        index_params=self.index_params,
                        using=self.alias,
                    )

                # If default did not work, most likely on Zilliz Cloud
                except MilvusException:
                    # Use AUTOINDEX based index
                    self.index_params = {
                        "metric_type": "L2",
                        "index_type": "AUTOINDEX",
                        "params": {},
                    }
                    self.col.create_index(
                        self._vector_field,
                        index_params=self.index_params,
                        using=self.alias,
                    )
                logger.debug(
                    "Successfully created an index on collection: %s",
                    self.collection_name,
                )

            except MilvusException as e:
                logger.error(
                    "Failed to create an index on collection: %s", self.collection_name
                )
                raise e

    def _create_search_params(self) -> None:
        """Generate search params based on the current index type"""
        from pymilvus import Collection

        if isinstance(self.col, Collection) and self.search_params is None:
            index = self._get_index()
            if index is not None:
                index_type: str = index["index_param"]["index_type"]
                metric_type: str = index["index_param"]["metric_type"]
                self.search_params = self.default_search_params[index_type]
                self.search_params["metric_type"] = metric_type

    def _load(
        self,
        partition_names: Optional[list] = None,
        replica_number: int = 1,
        timeout: Optional[float] = None,
    ) -> None:
        """Load the collection if available."""
        from pymilvus import Collection, utility
        from pymilvus.client.types import LoadState

        timeout = self.timeout or timeout
        if (
            isinstance(self.col, Collection)
            and self._get_index() is not None
            and utility.load_state(self.collection_name, using=self.alias)
            == LoadState.NotLoad
        ):
            self.col.load(
                partition_names=partition_names,
                replica_number=replica_number,
                timeout=timeout,
            )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        timeout: Optional[float] = None,
        batch_size: int = 1000,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert text data into Milvus.

        Inserting data when the collection has not be made yet will result
        in creating a new Collection. The data of the first entity decides
        the schema of the new collection, the dim is extracted from the first
        embedding and the columns are decided by the first metadata dict.
        Metadata keys will need to be present for all inserted values. At
        the moment there is no None equivalent in Milvus.

        Args:
            texts (Iterable[str]): The texts to embed, it is assumed
                that they all fit in memory.
            metadatas (Optional[List[dict]]): Metadata dicts attached to each of
                the texts. Defaults to None.
            should be less than 65535 bytes. Required and work when auto_id is False.
            timeout (Optional[float]): Timeout for each batch insert. Defaults
                to None.
            batch_size (int, optional): Batch size to use for insertion.
                Defaults to 1000.
            ids (Optional[List[str]]): List of text ids. The length of each item

        Raises:
            MilvusException: Failure to add texts

        Returns:
            List[str]: The resulting keys for each inserted element.
        """
        from pymilvus import Collection, MilvusException

        texts = list(texts)
        if not self.auto_id:
            assert isinstance(
                ids, list
            ), "A list of valid ids are required when auto_id is False."
            assert len(set(ids)) == len(
                texts
            ), "Different lengths of texts and unique ids are provided."
            assert all(
                len(x.encode()) <= 65_535 for x in ids
            ), "Each id should be a string less than 65535 bytes."

        try:
            embeddings = self.embedding_func.embed_documents(texts)
        except NotImplementedError:
            embeddings = [self.embedding_func.embed_query(x) for x in texts]

        if len(embeddings) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []

        # If the collection hasn't been initialized yet, perform all steps to do so
        if not isinstance(self.col, Collection):
            kwargs = {"embeddings": embeddings, "metadatas": metadatas}
            if self.partition_names:
                kwargs["partition_names"] = self.partition_names
            if self.replica_number:
                kwargs["replica_number"] = self.replica_number
            if self.timeout:
                kwargs["timeout"] = self.timeout
            self._init(**kwargs)

        # Dict to hold all insert columns
        insert_dict: dict[str, list] = {
            self._text_field: texts,
            self._vector_field: embeddings,
        }

        if not self.auto_id:
            insert_dict[self._primary_field] = ids  # type: ignore[assignment]

        if self._metadata_field is not None:
            for d in metadatas:  # type: ignore[union-attr]
                insert_dict.setdefault(self._metadata_field, []).append(d)
        else:
            # Collect the metadata into the insert dict.
            if metadatas is not None:
                for d in metadatas:
                    for key, value in d.items():
                        keys = (
                            [x for x in self.fields if x != self._primary_field]
                            if self.auto_id
                            else [x for x in self.fields]
                        )
                        if key in keys:
                            insert_dict.setdefault(key, []).append(value)

        # Total insert count
        vectors: list = insert_dict[self._vector_field]
        total_count = len(vectors)

        pks: list[str] = []

        assert isinstance(self.col, Collection)
        for i in range(0, total_count, batch_size):
            # Grab end index
            end = min(i + batch_size, total_count)
            # Convert dict to list of lists batch for insertion
            insert_list = [
                insert_dict[x][i:end] for x in self.fields if x in insert_dict
            ]
            # Insert into the collection.
            try:
                res: Collection
                timeout = self.timeout or timeout
                res = self.col.insert(insert_list, timeout=timeout, **kwargs)
                pks.extend(res.primary_keys)
            except MilvusException as e:
                logger.error(
                    "Failed to insert batch starting at entity: %s/%s", i, total_count
                )
                raise e
        return pks

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            query (str): The text to search.
            k (int, optional): How many results to return. Defaults to 4.
            param (dict, optional): The search params for the index type.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (int, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        timeout = self.timeout or timeout
        res = self.similarity_search_with_score(
            query=query, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in res]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            embedding (List[float]): The embedding vector to search.
            k (int, optional): How many results to return. Defaults to 4.
            param (dict, optional): The search params for the index type.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (int, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        timeout = self.timeout or timeout
        res = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in res]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            query (str): The text being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[float], List[Tuple[Document, any, any]]:
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        # Embed the query text.
        embedding = self.embedding_func.embed_query(query)
        timeout = self.timeout or timeout
        res = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return res

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            embedding (List[float]): The embedding vector being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        if param is None:
            param = self.search_params

        # Determine result metadata fields with PK.
        output_fields = self.fields[:]
        output_fields.remove(self._vector_field)
        timeout = self.timeout or timeout
        # Perform the search.
        res = self.col.search(
            data=[embedding],
            anns_field=self._vector_field,
            param=param,
            limit=k,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results.
        ret = []
        for result in res[0]:
            data = {x: result.entity.get(x) for x in output_fields}
            doc = self._parse_document(data)
            pair = (doc, result.score)
            ret.append(pair)

        return ret

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR.

        Args:
            query (str): The text being searched.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            param (dict, optional): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.


        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        embedding = self.embedding_func.embed_query(query)
        timeout = self.timeout or timeout
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            param=param,
            expr=expr,
            timeout=timeout,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR.

        Args:
            embedding (str): The embedding vector being searched.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            param (dict, optional): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        if param is None:
            param = self.search_params

        # Determine result metadata fields.
        output_fields = self.fields[:]
        output_fields.remove(self._vector_field)
        timeout = self.timeout or timeout
        # Perform the search.
        res = self.col.search(
            data=[embedding],
            anns_field=self._vector_field,
            param=param,
            limit=fetch_k,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results.
        ids = []
        documents = []
        scores = []
        for result in res[0]:
            data = {x: result.entity.get(x) for x in output_fields}
            doc = self._parse_document(data)
            documents.append(doc)
            scores.append(result.score)
            ids.append(result.id)

        vectors = self.col.query(
            expr=f"{self._primary_field} in {ids}",
            output_fields=[self._primary_field, self._vector_field],
            timeout=timeout,
        )
        # Reorganize the results from query to match search order.
        vectors = {x[self._primary_field]: x[self._vector_field] for x in vectors}

        ordered_result_embeddings = [vectors[x] for x in ids]

        # Get the new order of results.
        new_ordering = maximal_marginal_relevance(
            np.array(embedding), ordered_result_embeddings, k=k, lambda_mult=lambda_mult
        )

        # Reorder the values and return.
        ret = []
        for x in new_ordering:
            # Function can return -1 index
            if x == -1:
                break
            else:
                ret.append(documents[x])
        return ret

    def delete(  # type: ignore[no-untyped-def]
        self, ids: Optional[List[str]] = None, expr: Optional[str] = None, **kwargs: str
    ):
        """Delete by vector ID or boolean expression.
        Refer to [Milvus documentation](https://milvus.io/docs/delete_data.md)
        for notes and examples of expressions.

        Args:
            ids: List of ids to delete.
            expr: Boolean expression that specifies the entities to delete.
            kwargs: Other parameters in Milvus delete api.
        """
        if isinstance(ids, list) and len(ids) > 0:
            if expr is not None:
                logger.warning(
                    "Both ids and expr are provided. " "Ignore expr and delete by ids."
                )
            expr = f"{self._primary_field} in {ids}"
        else:
            assert isinstance(
                expr, str
            ), "Either ids list or expr string must be provided."
        return self.col.delete(expr=expr, **kwargs)  # type: ignore[union-attr]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = "LangChainCollection",
        connection_args: dict[str, Any] = DEFAULT_MILVUS_CONNECTION,
        consistency_level: str = "Session",
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        drop_old: bool = False,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Milvus:
        """Create a Milvus collection, indexes it with HNSW, and insert data.

        Args:
            texts (List[str]): Text data.
            embedding (Embeddings): Embedding function.
            metadatas (Optional[List[dict]]): Metadata for each text if it exists.
                Defaults to None.
            collection_name (str, optional): Collection name to use. Defaults to
                "LangChainCollection".
            connection_args (dict[str, Any], optional): Connection args to use. Defaults
                to DEFAULT_MILVUS_CONNECTION.
            consistency_level (str, optional): Which consistency level to use. Defaults
                to "Session".
            index_params (Optional[dict], optional): Which index_params to use. Defaults
                to None.
            search_params (Optional[dict], optional): Which search params to use.
                Defaults to None.
            drop_old (Optional[bool], optional): Whether to drop the collection with
                that name if it exists. Defaults to False.
            ids (Optional[List[str]]): List of text ids. Defaults to None.

        Returns:
            Milvus: Milvus Vector Store
        """
        if isinstance(ids, list) and len(ids) > 0:
            auto_id = False
        else:
            auto_id = True

        vector_db = cls(
            embedding_function=embedding,
            collection_name=collection_name,
            connection_args=connection_args,
            consistency_level=consistency_level,
            index_params=index_params,
            search_params=search_params,
            drop_old=drop_old,
            auto_id=auto_id,
            **kwargs,
        )
        vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vector_db

    def _parse_document(self, data: dict) -> Document:
        return Document(
            page_content=data.pop(self._text_field),
            metadata=data.pop(self._metadata_field) if self._metadata_field else data,
        )

    def get_pks(self, expr: str, **kwargs: Any) -> List[int] | None:
        """Get primary keys with expression

        Args:
            expr: Expression - E.g: "id in [1, 2]", or "title LIKE 'Abc%'"

        Returns:
            List[int]: List of IDs (Primary Keys)
        """

        from pymilvus import MilvusException

        if self.col is None:
            logger.debug("No existing collection to get pk.")
            return None

        try:
            query_result = self.col.query(
                expr=expr, output_fields=[self._primary_field]
            )
        except MilvusException as exc:
            logger.error("Failed to get ids: %s error: %s", self.collection_name, exc)
            raise exc
        pks = [item.get(self._primary_field) for item in query_result]
        return pks

    def upsert(  # type: ignore[override]
        self,
        ids: Optional[List[str]] = None,
        documents: List[Document] | None = None,
        **kwargs: Any,
    ) -> List[str] | None:
        """Update/Insert documents to the vectorstore.

        Args:
            ids: IDs to update - Let's call get_pks to get ids with expression \n
            documents (List[Document]): Documents to add to the vectorstore.

        Returns:
            List[str]: IDs of the added texts.
        """

        from pymilvus import MilvusException

        if documents is None or len(documents) == 0:
            logger.debug("No documents to upsert.")
            return None

        if ids is not None and len(ids):
            kwargs["ids"] = ids
            try:
                self.delete(ids=ids)
            except MilvusException:
                pass
        try:
            return self.add_documents(documents=documents, **kwargs)
        except MilvusException as exc:
            logger.error(
                "Failed to upsert entities: %s error: %s", self.collection_name, exc
            )
            raise exc
