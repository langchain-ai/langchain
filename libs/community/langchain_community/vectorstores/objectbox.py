import asyncio
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from objectbox.model import *
from objectbox.model.properties import *


class ObjectBox(VectorStore):
    """
    ObjectBox as Vector Store.
    To use, you should have the ``objectbox`` python package installed.
    Args:
        embedding_function: Embedding function to use.
        vector_box: initializing objectbox
    """

    def __init__(self, embedding_function: Embeddings, embedding_dimensions: int):
        self._embedding_function = embedding_function
        self._embedding_dimensions = embedding_dimensions
        self._vector_box = objectbox.Box(
            db, self._create_entity_class(self._embedding_dimensions)
        )
        # TODO: create objectbox db instance properly

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return (
            self.embedding_function
            if isinstance(self.embedding_function, Embeddings)
            else None
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add list of text along with embeddings to the vector store
        Args:
            texts (Iterable[str]): collection of text to add to the database
        Returns:
            List of ids for the newly inserted documents
        """
        return asyncio.run(self.aadd_texts(texts, metadatas, **kwargs))

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add list of text along with embeddings to the vector store asynchronously
        Args:
            texts (Iterable[str]): collection of text to add to the database
        Returns:
            List of ids for the newly inserted documents
        """
        embeddings = self.embedding_function.embed_documents(list(texts))
        ids = []
        with self.ob.read_tx():
            for idx, text in enumerate(texts):
                record = await self._vector_box.put(
                    VectorEntity(text=text, embeddings=embeddings[idx])
                )
                ids.append(record[0]["id"])
        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Run similarity search on query
        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query
        """

        async def _similarity_search() -> List[Document]:
            qb = self._vector_box.query()
            embedded_query = self.embedding_function.embed_query(query)
            qb.nearest_neighbors_f32("embeddings", embedded_query, k)
            query_build = qb.build()
            query_result = query_build.find()
            return [Document(page_content=result.text, metadata={"id": result.id}) for result in query_result]

        return asyncio.run(_similarity_search())

    def _create_entity_class(self, dimensions: int):
        """Dynamically define an Entity class according to the parameters."""

        @Entity(id=1, uid=1)
        class VectorEntity:
            id = Id(id=1, uid=1001)
            text = Property(str, type=PropertyType.string, id=2, uid=1002)
            embeddings = Property(
                np.ndarray,
                type=PropertyType.floatVector,
                id=3,
                uid=1003,
                index=HnswIndex(
                    id=3,
                    uid=10001,
                    dimensions=dimensions,
                    distance_type=HnswDistanceType.EUCLIDEAN,
                ),
            )

        return VectorEntity
