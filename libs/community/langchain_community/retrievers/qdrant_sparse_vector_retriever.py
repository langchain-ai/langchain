import uuid
from itertools import islice
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import pre_init
from pydantic import ConfigDict

from langchain_community.vectorstores.qdrant import Qdrant, QdrantException


@deprecated(
    since="0.2.16",
    alternative=(
        "Qdrant vector store now supports sparse retrievals natively. "
        "Use langchain_qdrant.QdrantVectorStore#as_retriever() instead. "
        "Reference: "
        "https://python.langchain.com/docs/integrations/vectorstores/qdrant/#sparse-vector-search"
    ),
    removal="0.5.0",
)
class QdrantSparseVectorRetriever(BaseRetriever):
    """Qdrant sparse vector retriever."""

    client: Any = None
    """'qdrant_client' instance to use."""
    collection_name: str
    """Qdrant collection name."""
    sparse_vector_name: str
    """Name of the sparse vector to use."""
    sparse_encoder: Callable[[str], Tuple[List[int], List[float]]]
    """Sparse encoder function to use."""
    k: int = 4
    """Number of documents to return per query. Defaults to 4."""
    filter: Optional[Any] = None
    """Qdrant qdrant_client.models.Filter to use for queries. Defaults to None."""
    content_payload_key: str = "content"
    """Payload field containing the document content. Defaults to 'content'"""
    metadata_payload_key: str = "metadata"
    """Payload field containing the document metadata. Defaults to 'metadata'."""
    search_options: Dict[str, Any] = {}
    """Additional search options to pass to qdrant_client.QdrantClient.search()."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that 'qdrant_client' python package exists in environment."""
        try:
            from grpc import RpcError
            from qdrant_client import QdrantClient, models
            from qdrant_client.http.exceptions import UnexpectedResponse
        except ImportError:
            raise ImportError(
                "Could not import qdrant-client python package. "
                "Please install it with `pip install qdrant-client`."
            )

        client = values["client"]
        if not isinstance(client, QdrantClient):
            raise ValueError(
                f"client should be an instance of qdrant_client.QdrantClient, "
                f"got {type(client)}"
            )

        filter = values["filter"]
        if filter is not None and not isinstance(filter, models.Filter):
            raise ValueError(
                f"filter should be an instance of qdrant_client.models.Filter, "
                f"got {type(filter)}"
            )

        client = cast(QdrantClient, client)

        collection_name = values["collection_name"]
        sparse_vector_name = values["sparse_vector_name"]

        try:
            collection_info = client.get_collection(collection_name)
            sparse_vectors_config = collection_info.config.params.sparse_vectors

            if sparse_vector_name not in sparse_vectors_config:
                raise QdrantException(
                    f"Existing Qdrant collection {collection_name} does not "
                    f"contain sparse vector named {sparse_vector_name}."
                    f"Did you mean one of {', '.join(sparse_vectors_config.keys())}?"
                )
        except (UnexpectedResponse, RpcError, ValueError):
            raise QdrantException(
                f"Qdrant collection {collection_name} does not exist."
            )
        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        from qdrant_client import QdrantClient, models

        client = cast(QdrantClient, self.client)
        query_indices, query_values = self.sparse_encoder(query)
        results = client.search(
            self.collection_name,
            query_filter=self.filter,
            query_vector=models.NamedSparseVector(
                name=self.sparse_vector_name,
                vector=models.SparseVector(
                    indices=query_indices,
                    values=query_values,
                ),
            ),
            limit=self.k,
            with_vectors=False,
            **self.search_options,
        )
        return [
            Qdrant._document_from_scored_point(
                point,
                self.collection_name,
                self.content_payload_key,
                self.metadata_payload_key,
            )
            for point in results
        ]

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents (List[Document]: Documents to add to the vectorstore.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> List[str]:
        from qdrant_client import QdrantClient

        added_ids = []
        client = cast(QdrantClient, self.client)
        for batch_ids, points in self._generate_rest_batches(
            texts, metadatas, ids, batch_size
        ):
            client.upsert(self.collection_name, points=points, **kwargs)
            added_ids.extend(batch_ids)

        return added_ids

    def _generate_rest_batches(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 64,
    ) -> Generator[Tuple[List[str], List[Any]], None, None]:
        from qdrant_client import models as rest

        texts_iterator = iter(texts)
        metadatas_iterator = iter(metadatas or [])
        ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])
        while batch_texts := list(islice(texts_iterator, batch_size)):
            # Take the corresponding metadata and id for each text in a batch
            batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
            batch_ids = list(islice(ids_iterator, batch_size))

            # Generate the sparse embeddings for all the texts in a batch
            batch_embeddings: List[Tuple[List[int], List[float]]] = [
                self.sparse_encoder(text) for text in batch_texts
            ]

            points = [
                rest.PointStruct(
                    id=point_id,
                    vector={
                        self.sparse_vector_name: rest.SparseVector(
                            indices=sparse_vector[0],
                            values=sparse_vector[1],
                        )
                    },
                    payload=payload,
                )
                for point_id, sparse_vector, payload in zip(
                    batch_ids,
                    batch_embeddings,
                    Qdrant._build_payloads(
                        batch_texts,
                        batch_metadatas,
                        self.content_payload_key,
                        self.metadata_payload_key,
                    ),
                )
            ]

            yield batch_ids, points
