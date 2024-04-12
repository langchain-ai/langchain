from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.config import run_in_executor

from langchain_community.vectorstores.milvus import DEFAULT_MILVUS_CONNECTION

if TYPE_CHECKING:
    from pymilvus import Collection


logger = logging.getLogger(__name__)


# default index params for dense vectors
DEFAULT_DENSE_INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64},
}

# If default index params did not work, most likely on Zilliz Cloud
# try default fallback index params for dense vectors instead
DEFAULT_FALLBACK_DENSE_INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "AUTOINDEX",
    "params": {},
}

# default index params for sparse vectors
DEFAULT_SPARSE_INDEX_PARAMS = {
    "metric_type": "IP",
    "index_type": "SPARSE_INVERTED_INDEX",
    "params": {"drop_ratio_build": 0.0},
}


# default search params
DEFAULT_DENSE_SEARCH_PARAMS = {
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

# default sparse search params
DEFAULT_SPARSE_SEARCH_PARAMS = {
    "SPARSE_INVERTED_INDEX": {
        "metric_type": "IP",
        "params": {
            "drop_ratio_search": 0.0,
        },
    },
    "SPARSE_WAND": {
        "metric_type": "IP",
        "params": {
            "drop_ratio_search": 0.0,
        },
    },
    "AUTOINDEX": {
        "metric_type": "IP",
        "params": {
            "drop_ratio_search": 0.0,
        },
    },
}


class SparseEmbeddings(ABC):
    """Interface for sparse embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> Dict[int, float]:
        """Embed query text."""

    async def aembed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        """Asynchronous Embed search docs."""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> Dict[int, float]:
        """Asynchronous Embed query text."""
        return await run_in_executor(None, self.embed_query, text)


def _create_connection_alias(connection_args: dict) -> str:
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


def _create_collection(
    embeddings: Dict[str, List[List[float]]],
    sparse_embeddings: Dict[str : List[Dict[int, float]]],
    collection_name: str,
    collection_description: str,
    alias: str,
    consistency_level: str,
    auto_id: bool,
    primary_field: str,
    text_field: str,
    metadata_field: str,
    partition_key_field: str,
    collection_properties: Optional[Dict[str, Any]] = None,
) -> Any:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        MilvusException,
    )
    from pymilvus.orm.types import infer_dtype_bydata

    fields = []

    # Create the metadata field
    fields.append(FieldSchema(metadata_field, DataType.JSON))

    # Create the text field
    fields.append(FieldSchema(text_field, DataType.VARCHAR, max_length=65_535))
    # Create the primary key field
    if auto_id:
        fields.append(
            FieldSchema(primary_field, DataType.INT64, is_primary=True, auto_id=True)
        )
    else:
        fields.append(
            FieldSchema(
                primary_field,
                DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=65_535,
            )
        )
    # Create the dense vector field, supports binary or float vectors
    for vector_field, vector_embeddings in embeddings.items():
        dim = len(vector_embeddings[0])
        fields.append(
            FieldSchema(vector_field, infer_dtype_bydata(vector_embeddings[0]), dim=dim)
        )

    # Create the sparse vector field, supports float vectors
    for vector_field in sparse_embeddings:
        fields.append(FieldSchema(vector_field, DataType.SPARSE_FLOAT_VECTOR))

    # Create the schema for the collection
    schema = CollectionSchema(
        fields,
        description=collection_description,
        partition_key_field=partition_key_field,
    )

    # Create the collection
    try:
        col = Collection(
            name=collection_name,
            schema=schema,
            consistency_level=consistency_level,
            using=alias,
        )
        # Set the collection properties if they exist
        if collection_properties is not None:
            col.set_properties(collection_properties)
        return col
    except MilvusException as e:
        logger.error("Failed to create collection: %s error: %s", collection_name, e)
        raise e


def _create_index(
    col: "Collection",
    collection_name: str,
    alias: str,
    vector_fields: List[str],
    sparse_vector_fields: List[str],
    index_params: Optional[dict] = None,
) -> None:
    """Create a index on the collection"""
    from pymilvus import MilvusException

    if not index_params:
        index_params = {}
    for vector_field in vector_fields:
        params = index_params.get(vector_field)
        if params is None:
            params = DEFAULT_DENSE_INDEX_PARAMS
        try:
            try:
                col.create_index(
                    vector_field,
                    index_params=params,
                    using=alias,
                )

            except MilvusException:
                params = DEFAULT_FALLBACK_DENSE_INDEX_PARAMS
                col.create_index(
                    vector_field,
                    index_params=params,
                    using=alias,
                )
            logger.debug(
                "Successfully created an index on "
                f"collection: {collection_name}, field_name: {vector_field}"
            )

        except MilvusException as e:
            logger.error(
                "Failed to create an index on "
                f"collection: {collection_name}, field_name: {vector_field}"
            )
            raise e

    for vector_field in sparse_vector_fields:
        params = index_params.get(vector_field)
        if params is None:
            params = DEFAULT_SPARSE_INDEX_PARAMS
        try:
            col.create_index(
                vector_field,
                index_params=params,
                using=alias,
            )
            logger.debug(
                "Successfully created an index on "
                f"collection: {collection_name}, field_name: {vector_field}"
            )
        except MilvusException as e:
            logger.error(
                "Failed to create an index on "
                f"collection: {collection_name}, field_name: {vector_field}"
            )
            raise e


def _get_indexes(col: "Collection", field_names: List[str]) -> Dict[str, Any]:
    """Return the vector index information if it exists"""
    indexes = {}
    for x in col.indexes:
        if x.field_name in field_names:
            indexes[x.field_name] = x.to_dict()
    return indexes


def _extract_fields(col) -> None:
    """Grab the existing fields from the Collection"""

    schema = col.schema
    fields = [x.name for x in schema.fields]
    return fields


def _create_search_and_rerank_params(
    col: "Collection",
    vector_fields: List[str],
    sparse_vector_fields: List[str],
    search_params: Optional[Dict[str, Dict[str, Any]]] = None,
    rerank_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate search params based on the current index type"""

    if search_params is None:
        search_params = {}

    indexes = _get_indexes(col, vector_fields)
    for vector_field, index in indexes.items():
        field_search_params = deepcopy(search_params.get(vector_field, {}))
        if "param" not in field_search_params:
            index_type: str = index["index_param"]["index_type"]
            metric_type: str = index["index_param"]["metric_type"]
            params = DEFAULT_DENSE_SEARCH_PARAMS[index_type]
            params["metric_type"] = metric_type
            field_search_params["param"] = params
        search_params[vector_field] = field_search_params

    sparse_indexes = _get_indexes(col, sparse_vector_fields)
    for sparse_vector_field, index in sparse_indexes.items():
        field_search_params = deepcopy(search_params.get(sparse_vector_field, {}))

        if "param" not in field_search_params:
            index_type: str = index["index_param"]["index_type"]
            metric_type: str = index["index_param"]["metric_type"]
            params = DEFAULT_SPARSE_SEARCH_PARAMS[index_type]
            params["metric_type"] = metric_type
            field_search_params["param"] = params
        search_params[sparse_vector_field] = field_search_params

    # TODO: make a default rerank_params
    if rerank_params is None:
        rerank_params = {"type": "RRF", "param": {"k": 60.0}}
    return search_params, rerank_params


def _get_reranker(rerank_params: Dict[str, Any], ann_search_fields: List[str]) -> Any:
    from pymilvus import RRFRanker, WeightedRanker

    rerank_type = rerank_params["type"]
    if rerank_type == "RRF":
        return RRFRanker(**rerank_params["param"])
    elif rerank_type == "Weighted":
        weights: Dict[str, float] = rerank_params["param"]["weights"]
        args = []
        for field in ann_search_fields:
            if field not in weights:
                logger.info(f"no weight set for {field}, use 0 instead")
                args.append(0.0)
            else:
                args.append(weights[field])
        return WeightedRanker(*args)
    else:
        raise ValueError(f"{rerank_type} in not allowed, only support RRF and Weighted")


def _load(
    col: "Collection",
    collection_name: str,
    alias: str,
    partition_names: Optional[list] = None,
    replica_number: int = 1,
    timeout: Optional[float] = None,
) -> None:
    """Load the collection if available."""
    from pymilvus import utility
    from pymilvus.client.types import LoadState

    if utility.load_state(collection_name, using=alias) == LoadState.NotLoad:
        col.load(
            partition_names=partition_names,
            replica_number=replica_number,
            timeout=timeout,
        )


class MilvusHybridSearchRetriever(BaseRetriever):
    """`Milvus hybrid search` retriever.

    See the documentation:
      https://milvus.io/api-reference/pymilvus/v2.4.x/ORM/Collection/hybrid_search.md
    """

    embedding_functions: Optional[Dict[str, Embeddings]] = None
    sparse_embedding_functions: Optional[Dict[str, SparseEmbeddings]] = None
    collection_name: str = "LangChainCollection"
    collection_properties: Optional[Dict[str, Any]] = None
    connection_args: Optional[Dict[str, Any]] = None
    consistency_level: str = "Session"
    collection_description: str = ""
    index_params: Optional[List[Any]] = None
    search_params: Optional[Dict[str, Dict[str, Any]]] = None
    rerank_params: Optional[Dict[str, Any]] = None
    text_field: str = "text"
    primary_field: str = "pk"
    metadata_field: str = "metadata"
    partition_key_field: Optional[str] = None
    partition_names: Optional[List[str]] = None
    auto_id: bool = False
    drop_old: bool = False
    alias: Optional[str] = None
    replica_number: int = 1
    timeout: Optional[float] = None
    col: Optional["Collection"] = None  #: :meta private:
    vector_fields: List[str]  #: :meta private:
    sparse_vector_fields: List[str]  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def create_collection(
        cls,
        values: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            import pymilvus
        except ImportError:
            raise ValueError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            )
        version = tuple(int(v) for v in pymilvus.__version__.split(".")[:2])
        if version < (2, 4):
            raise ValueError(
                "Please upgrade pymilvus python package. "
                "Please install it with `pip install pymilvus>=2.4.0`."
            )
        from pymilvus import Collection, utility

        if not values["embedding_functions"]:
            values["embedding_functions"] = {}
        values["vector_fields"] = list(values["embedding_functions"])
        if not values["sparse_embedding_functions"]:
            values["sparse_embedding_functions"] = {}
        values["sparse_vector_fields"] = list(values["sparse_embedding_functions"])

        # TODO: may should support only one sparse/dense embedding function, too
        if len(values["vector_fields"]) + len(values["vector_fields"]) < 2:
            raise ValueError("Please pass at least 2 embedding functions")

        same_vector_names = set(values["vector_fields"]) & set(
            values["sparse_vector_fields"]
        )
        if same_vector_names:
            raise ValueError(
                "Both embedding_functions and sparse_embedding_functions "
                f"contains vector_names {same_vector_names}"
            )

        if values.get("connection_args") is None:
            values["connection_args"] = DEFAULT_MILVUS_CONNECTION

        if not values.get("alias"):
            values["alias"] = _create_connection_alias(values["connection_args"])

        col = None

        # TODO: any other appoarch to do?
        collection_name = values.get(
            "collection_name", cls.__fields__["collection_name"].default
        )
        consistency_level = values.get(
            "consistency_level", cls.__fields__["consistency_level"].default
        )
        drop_old = values.get("drop_old", cls.__fields__["drop_old"].default)
        replica_number = values.get(
            "replica_number", cls.__fields__["replica_number"].default
        )

        if utility.has_collection(collection_name, using=values["alias"]):
            col = Collection(
                collection_name,
                using=values["alias"],
                consistency_level=consistency_level,
            )
            collection_properties = values.get("collection_properties")
            if collection_properties is not None:
                col.set_properties(values["collection_properties"])
        if drop_old and isinstance(col, Collection):
            col.drop()
            col = None

        if isinstance(col, Collection):
            _load(
                col,
                collection_name=collection_name,
                alias=values["alias"],
                partition_names=values.get("partition_names"),
                replica_number=replica_number,
                timeout=values.get("timeout"),
            )

            # TODO: duplicate code
            search_params, rerank_params = _create_search_and_rerank_params(
                col=col,
                vector_fields=values["vector_fields"],
                sparse_vector_fields=values["sparse_vector_fields"],
                search_params=values.get("search_params"),
                rerank_params=values.get("rerank_params"),
            )
            values["search_params"] = search_params
            values["rerank_params"] = rerank_params

        values["col"] = col

        return values

    def _init(
        self,
        embeddings: List[List[float]],
        sparse_embeddings: List[Dict[int, float]],
    ) -> None:
        self.col = _create_collection(
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
            collection_name=self.collection_name,
            collection_description=self.collection_description,
            alias=self.alias,
            consistency_level=self.consistency_level,
            auto_id=self.auto_id,
            primary_field=self.primary_field,
            text_field=self.text_field,
            metadata_field=self.metadata_field,
            partition_key_field=self.partition_key_field,
            collection_properties=self.collection_properties,
        )

        _create_index(
            col=self.col,
            collection_name=self.collection_name,
            alias=self.alias,
            vector_fields=self.vector_fields,
            sparse_vector_fields=self.sparse_vector_fields,
            index_params=self.index_params,
        )

        _load(
            col=self.col,
            collection_name=self.collection_name,
            alias=self.alias,
            replica_number=self.replica_number,
            timeout=self.timeout,
        )

        self.search_params, self.rerank_params = _create_search_and_rerank_params(
            col=self.col,
            vector_fields=self.vector_fields,
            sparse_vector_fields=self.sparse_vector_fields,
            search_params=self.search_params,
            rerank_params=self.rerank_params,
        )

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        timeout: Optional[float] = None,
        batch_size: int = 1000,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """add texts to Milvus."""
        from pymilvus import MilvusException

        total_count = len(texts)
        if not metadatas:
            metadatas = [{}] * total_count
        insert_dict: dict[str, list] = {
            self.text_field: texts,
            self.metadata_field: metadatas,
        }
        if not self.auto_id:
            insert_dict[self.primary_field] = ids

        embeddings: Dict[str, List[List[float]]] = {}
        sparse_embeddings: Dict[str, List[Dict[int, float]]] = {}

        for vector_field, func in self.embedding_functions.items():
            insert_dict[vector_field] = func.embed_documents(texts)
            embeddings[vector_field] = insert_dict[vector_field]
        for vector_field, func in self.sparse_embedding_functions.items():
            insert_dict[vector_field] = func.embed_documents(texts)
            sparse_embeddings[vector_field] = insert_dict[vector_field]

        if self.col is None:
            self._init(
                embeddings=embeddings,
                sparse_embeddings=sparse_embeddings,
            )

        pks: list[str] = []

        # TODO: fields should be self.fields?
        fields = _extract_fields(self.col)

        for i in range(0, total_count, batch_size):
            end = min(i + batch_size, total_count)

            batch_insert_list = [insert_dict[x][i:end] for x in fields]

            try:
                res: Collection
                timeout = self.timeout or timeout
                res = self.col.insert(batch_insert_list, timeout=timeout, **kwargs)
                pks.extend(res.primary_keys)
            except MilvusException as e:
                logger.error(
                    "Failed to insert batch starting at entity: %s/%s", i, total_count
                )
                raise e
        return pks

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: int = 4,
        search_params: Optional[Dict[str, Dict[str, Any]]] = None,
        include_other_fields: bool = True,
        rerank_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        from pymilvus import AnnSearchRequest
        # TODO: add docstring:
        # search_params = {"dense_vector": {"limit": 10, "expr": 'pk in [1, 2]'}}

        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        available_vector_fields = set(self.vector_fields + self.sparse_vector_fields)

        ann_search_dict = {}
        if not search_params:
            search_params = {}
        for field, search_kwargs in search_params.items():
            search_kwargs = deepcopy(search_kwargs)
            if field not in available_vector_fields:
                logger.info(f"{field} is not a available vector field, ignored")
                continue
            if "limit" not in search_kwargs:
                search_kwargs["limit"] = k
            ann_search_dict[field] = search_kwargs

        if include_other_fields:
            other_fields = available_vector_fields - set(search_params)
            for field in other_fields:
                ann_search_dict[field] = deepcopy(self.search_params[field])
                if "limit" not in ann_search_dict[field]:
                    ann_search_dict[field]["limit"] = k

        if rerank_params is None:
            rerank_params = self.rerank_params

        all_embeddings: Dict[str, Union[List[float], Dict[int, float]]] = {}
        for vector_field, func in self.embedding_functions.items():
            if vector_field in ann_search_dict:
                all_embeddings[vector_field] = func.embed_query(query)
        for sparse_vector_field, func in self.sparse_embedding_functions.items():
            if sparse_vector_field in ann_search_dict:
                all_embeddings[sparse_vector_field] = func.embed_query(query)

        reqs = []
        ann_search_fields = []
        for field, ann_search_params in ann_search_dict.items():
            reqs.append(
                AnnSearchRequest(
                    data=[all_embeddings[field]],
                    anns_field=field,
                    **ann_search_params,
                )
            )
            ann_search_fields.append(field)

        rerank = _get_reranker(rerank_params, ann_search_fields)

        logger.debug("ann_search_fields:\n{ann_search_fields}\n" "rerank: {rerank}")

        res = self.col.hybrid_search(
            reqs=reqs,
            rerank=rerank,
            limit=k,
            output_fields=[self.text_field, self.metadata_field],
            timeout=timeout,
            **kwargs,
        )

        docs = []
        for result in res[0]:
            docs.append(
                Document(
                    page_content=result.entity.get(self.text_field),
                    metadata=result.entity.get(self.metadata_field),
                )
            )
        return docs
