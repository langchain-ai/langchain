from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.milvus import Milvus

logger = logging.getLogger(__name__)


class Zilliz(Milvus):
    """Initialize wrapper around the Zilliz vector database.

    In order to use this you need to have `pymilvus` installed and a
    running Zilliz database.

    See the following documentation for how to run a Zilliz instance:
    https://docs.zilliz.com/docs/create-cluster


    IF USING L2/IP metric IT IS HIGHLY SUGGESTED TO NORMALIZE YOUR DATA.

    Args:
        embedding_function (Embeddings): Function used to embed the text.
        collection_name (str): Which Zilliz collection to use. Defaults to
            "LangChainCollection".
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

    The connection args used for this class comes in the form of a dict,
    here are a few of the options:
        address (str): The actual address of Zilliz
            instance. Example address: "localhost:19530"
        uri (str): The uri of Zilliz instance. Example uri:
            "https://in03-ba4234asae.api.gcp-us-west1.zillizcloud.com",
        host (str): The host of Zilliz instance. Default at "localhost",
            PyMilvus will fill in the default host if only port is provided.
        port (str/int): The port of Zilliz instance. Default at 19530, PyMilvus
            will fill in the default port if only host is provided.
        user (str): Use which user to connect to Zilliz instance. If user and
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

        from langchain import Zilliz
        from langchain.embeddings import OpenAIEmbeddings

        embedding = OpenAIEmbeddings()
        # Connect to a Zilliz instance
        milvus_store = Milvus(
            embedding_function = embedding,
            collection_name = "LangChainCollection",
            connection_args = {
                "uri": "https://in03-ba4234asae.api.gcp-us-west1.zillizcloud.com",
                "user": "temp",
                "password": "temp",
                "secure": True
            }
            drop_old: True,
        )

    Raises:
        ValueError: If the pymilvus python package is not installed.
    """

    def _create_index(self) -> None:
        """Create a index on the collection"""
        from pymilvus import Collection, MilvusException

        if isinstance(self.col, Collection) and self._get_index() is None:
            try:
                # If no index params, use a default AutoIndex based one
                if self.index_params is None:
                    self.index_params = {
                        "metric_type": "L2",
                        "index_type": "AUTOINDEX",
                        "params": {},
                    }

                try:
                    self.col.create_index(
                        self._vector_field,
                        index_params=self.index_params,
                        using=self.alias,
                    )

                # If default did not work, most likely Milvus self-hosted
                except MilvusException:
                    # Use HNSW based index
                    self.index_params = {
                        "metric_type": "L2",
                        "index_type": "HNSW",
                        "params": {"M": 8, "efConstruction": 64},
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

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = "LangChainCollection",
        connection_args: dict[str, Any] = {},
        consistency_level: str = "Session",
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        drop_old: bool = False,
        **kwargs: Any,
    ) -> Zilliz:
        """Create a Zilliz collection, indexes it with HNSW, and insert data.

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
            index_params (Optional[dict], optional): Which index_params to use.
                Defaults to None.
            search_params (Optional[dict], optional): Which search params to use.
                Defaults to None.
            drop_old (Optional[bool], optional): Whether to drop the collection with
                that name if it exists. Defaults to False.

        Returns:
            Zilliz: Zilliz Vector Store
        """
        vector_db = cls(
            embedding_function=embedding,
            collection_name=collection_name,
            connection_args=connection_args,
            consistency_level=consistency_level,
            index_params=index_params,
            search_params=search_params,
            drop_old=drop_old,
            **kwargs,
        )
        vector_db.add_texts(texts=texts, metadatas=metadatas)
        return vector_db
