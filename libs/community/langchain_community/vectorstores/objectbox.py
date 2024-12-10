import os
import shutil
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

try:
    import objectbox
    from objectbox.model.entity import Entity
    from objectbox.model.model import IdUid
    from objectbox.model.properties import (
        HnswDistanceType,
        HnswIndex,
        Id,
        Property,
        PropertyType,
    )
except ImportError:
    raise ImportError(
        "Could not import the objectbox package. "
        "Please install it with `pip install objectbox`."
    )

DIRECTORY = "data"


class ObjectBox(VectorStore):
    """
    ObjectBox as Vector Store.
    To use, you should have the `objectbox` python package installed and `flatbuffers`.
    Args:
        embedding_function: Embedding function to use.
        embedding_dimensions: Dimensions of the embeddings.
        db_directory: Path to the database where data will be stored.
        clear_db: Flag for deleting all the data in the database.
        entity_model: Creates an objectbox entity.
        db: Registers the model with objectbox.
        vector_box: Initializing objectbox.
    """

    def __init__(
        self,
        embedding: Embeddings,
        embedding_dimensions: int,
        distance_type: HnswDistanceType = HnswDistanceType.EUCLIDEAN,
        db_directory: Optional[str] = None,
        clear_db: Optional[bool] = False,
    ):
        self._embedding = embedding
        self._embedding_dimensions = embedding_dimensions
        self._distance_type = distance_type
        self.db_directory = db_directory
        self._clear_db = clear_db
        self._entity_model = self._create_entity_class()
        self._db = self._create_objectbox_db()
        self._vector_box = objectbox.Box(self._db, self._entity_model)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding if isinstance(self._embedding, Embeddings) else None

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
        embeddings = self._embedding.embed_documents(list(texts))
        ids = []

        if not metadatas:
            metadatas = [{} for _ in texts]

        with self._db.write_tx():
            for text, metadata, embedding in zip(texts, metadatas, embeddings):
                object_id = self._vector_box.put(
                    self._entity_model(
                        text=text, metadata=metadata, embeddings=embedding
                    )
                )
                ids.append(object_id)
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
        embedded_query = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(embedded_query, k, **kwargs)

    # Overwrite from VectorStore
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return lambda score: self._convert_score(self._distance_type, score)

    @staticmethod
    def _convert_score(type: HnswDistanceType, score: float) -> float:
        # Map ObjectBox distance to LangChain range,
        # in which 0 is dissimilar, 1 is most similar.
        if type == HnswDistanceType.EUCLIDEAN:
            # Not required: score = sqrt(score)  # ObjectBox returns squared Euclidean
            if score > 1.0:  # For now, we assume normalized vectors,
                # which result in scores in the range 0..1
                return 0.0
            return 1.0 - score
        elif type == HnswDistanceType.COSINE:
            return 1.0 - score / 2.0
        elif type == HnswDistanceType.DOT_PRODUCT:
            if score > 2.0:  # For now, we assume normalized vectors,
                # which result in scores in the range 0..2
                return 0.0
            return 1.0 - score / 2.0
        elif type == HnswDistanceType.DOT_PRODUCT_NON_NORMALIZED:
            return 1.0 - score / 2.0
        else:
            raise Exception(f"Unknown distance type: {type.name}")

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance.

        Args:
            query (str): Query
            k (int): Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents with score most similar to the query vector.
        """
        embedded_query = self._embedding.embed_query(query)
        embeddings_prop = self._entity_model.get_property("embeddings")
        qb = self._vector_box.query()
        qb.nearest_neighbors_f32(embeddings_prop, embedded_query, k)
        obx_query = qb.build()
        results = obx_query.find_with_scores()
        return [
            (Document(page_content=obj.text, metadata=obj.metadata), score)
            for obj, score in results
        ]

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding (List[float]): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """
        embeddings_prop = self._entity_model.get_property("embeddings")
        qb = self._vector_box.query()
        qb.nearest_neighbors_f32(embeddings_prop, embedding, k)
        query = qb.build()
        results = query.find_with_scores()
        return [
            Document(page_content=obj.text, metadata=obj.metadata) for obj, _ in results
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> objectbox:
        """Create ObjectBox from list of text
        Args:
            texts (List[str]): list of text to vectorize and store
            embedding (Optional[Embeddings]): Embedding function.
        Returns:
            ObjectBox object initialized and ready for use."""
        ob = cls(embedding, **kwargs)
        ob.add_texts(texts, metadatas)
        return ob

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Delete by vector ID.

        Args:
            ids: List of ids to delete.

        Returns:
            bool: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if ids is None:
            raise ValueError("ids are required for delete()")
        with self._db.write_tx():
            for id in ids:
                try:
                    int_id = int(id)
                except ValueError:
                    raise ValueError("invalid id input")
                self._vector_box.remove(int_id)
        return True

    def _create_objectbox_db(self) -> objectbox:
        """registering the VectorEntity model and setting up objectbox database"""
        db_path = "data" if self.db_directory is None else self.db_directory
        if self._clear_db and os.path.exists(db_path):
            shutil.rmtree(db_path)
        model = objectbox.Model()
        model.entity(self._entity_model, last_property_id=IdUid(4, 1004))
        model.last_entity_id = IdUid(1, 1)
        model.last_index_id = IdUid(3, 10001)
        return objectbox.Builder().model(model).directory(db_path).build()

    def _create_entity_class(self) -> Entity:
        """Dynamically define an Entity class according to the parameters."""

        @Entity(id=1, uid=1)
        class VectorEntity:
            id = Id(id=1, uid=1001)
            text = Property(str, type=PropertyType.string, id=2, uid=1002)
            metadata = Property(dict, type=PropertyType.flex, id=3, uid=1003)
            embeddings = Property(
                np.ndarray,
                type=PropertyType.floatVector,
                id=4,
                uid=1004,
                index=HnswIndex(
                    id=1,
                    uid=10001,
                    dimensions=self._embedding_dimensions,
                    distance_type=self._distance_type,
                ),
            )

        return VectorEntity
