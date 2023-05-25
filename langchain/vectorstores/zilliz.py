from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.milvus import Milvus

logger = logging.getLogger(__name__)


class Zilliz(Milvus):
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
