import asyncio
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


@Entity(id=1, uid=1)
class VectorEntity:
    id = Id(id=1, uid=1001)
    text = Property(str, type=PropertyType.string, id=2, uid=1002)
    embeddings = Property(np.ndarray, type=PropertyType.floatVector, id=3, uid=1003,
                          index=HnswIndex(id=3, uid=10001, dimensions=2, distance_type=HnswDistanceType.EUCLIDEAN))
    # TODO: dimensions should be variable -> figure out


class ObjectBox(VectorStore):

    def __init__(self, embedding_function: Embeddings):
        self._embedding_function = embedding_function
        self._vector_box = objectbox.Box(db, VectorEntity)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return (
            self.embedding_function
            if isinstance(self.embedding_function, Embeddings)
            else None
        )

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any, ) -> List[str]:
        return asyncio.run(self.aadd_texts(texts, metadatas, **kwargs))

    async def aadd_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[
        str]:
        embeddings = self.embedding_function.embed_documents(list(texts))
        ids = []
        for idx, text in enumerate(texts):
            record = await self._vector_box.put(
                VectorEntity(text=text, embeddings=embeddings[idx])
            )
            ids.append(record[0]["id"])
        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        async def _similarity_search() -> List[Document]:
            qb = self._vector_box.query()
            embedded_query = self.embedding_function.embed_query(query)
            qb.nearest_neighbors_f32("embeddings", embedded_query, k)
            query_build = qb.build()
            return query_build.find()

        return asyncio.run(_similarity_search())
