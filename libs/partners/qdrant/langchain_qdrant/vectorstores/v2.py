import os
from typing import Union, Sequence, Any, Optional, Tuple, List

from qdrant_client import QdrantClient, AsyncQdrantClient

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexes.types import UpsertResponse
from langchain_core.retrievers.v2 import RetrievalResponse
from langchain_core.vectorstores.v2 import VectorStoreV2


class QdrantV2(VectorStoreV2):

    client: QdrantClient
    collection_name: str
    async_client: Optional[AsyncQdrantClient] = None
    embeddings: Optional[Embeddings] = None
    content_payload_key: str = "page_content"
    metadata_payload_key: str = "metadata"
    distance_strategy: str = "COSINE"
    vector_name: Optional[str] = None

    def __init__(
        self,
        *,
        client: Optional[QdrantClient] = None,
        async_client: Optional[AsyncQdrantClient] = None,
        distance_strategy: str = "COSINE",
        # client kwargs
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:

        if not client:
            if api_key is None:
                api_key = os.getenv("QDRANT_API_KEY")
            client = QdrantClient(
                location=location,
                url=url,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
                https=https,
                api_key=api_key,
                prefix=prefix,
                timeout=timeout,
                host=host,
                path=path,
            )

        if not async_client or location == ":memory:" or path is not None:
            if api_key is None:
                api_key = os.getenv("QDRANT_API_KEY")
            async_client = AsyncQdrantClient(
                location=location,
                url=url,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
                https=https,
                api_key=api_key,
                prefix=prefix,
                timeout=timeout,
                host=host,
                path=path,
            )
        distance_strategy = distance_strategy.upper()
        super().__init__(client=client, async_client=async_client, distance_strategy=distance_strategy, **kwargs)

    def add(
        self,
        documents: Sequence[Document],
        *,
        ids: Optional[Union[List[str], Tuple[str]]] = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> UpsertResponse:

        added_ids = []
        for batch_ids, points in self._generate_rest_batches(
            documents, ids, batch_size
        ):
            self.client.upsert(
                collection_name=self.collection_name, points=points, **kwargs
            )
            added_ids.extend(batch_ids)

        return UpsertResponse(succeeded=added_ids, failed=[])

    def _retrieve(
        self,
        query: Union[str, Sequence[float]],
        *,
        method: str = "similarity",
        metric: str = "cosine_similarity",
        **kwargs: Any,
    ) -> RetrievalResponse:
        ...

    async def _aretrieve(
        self,
        query: Union[str, Sequence[float]],
        *,
        method: str = "similarity",
        metric: str = "cosine_similarity",
        **kwargs: Any,
    ) -> RetrievalResponse:
        ...

