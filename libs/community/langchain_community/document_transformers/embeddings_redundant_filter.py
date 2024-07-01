"""Transform documents"""

from typing import Any, Callable, List, Sequence

import numpy as np
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.utils.math import cosine_similarity


class _DocumentWithState(Document):
    """Wrapper for a document that includes arbitrary state."""

    state: dict = Field(default_factory=dict)
    """State associated with the document."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    def to_document(self) -> Document:
        """Convert the DocumentWithState to a Document."""
        return Document(page_content=self.page_content, metadata=self.metadata)

    @classmethod
    def from_document(cls, doc: Document) -> "_DocumentWithState":
        """Create a DocumentWithState from a Document."""
        if isinstance(doc, cls):
            return doc
        return cls(page_content=doc.page_content, metadata=doc.metadata)


def get_stateful_documents(
    documents: Sequence[Document],
) -> Sequence[_DocumentWithState]:
    """Convert a list of documents to a list of documents with state.

    Args:
        documents: The documents to convert.

    Returns:
        A list of documents with state.
    """
    return [_DocumentWithState.from_document(doc) for doc in documents]


def _filter_similar_embeddings(
    embedded_documents: List[List[float]], similarity_fn: Callable, threshold: float
) -> List[int]:
    """Filter redundant documents based on the similarity of their embeddings."""
    similarity = np.tril(similarity_fn(embedded_documents, embedded_documents), k=-1)
    redundant = np.where(similarity > threshold)
    redundant_stacked = np.column_stack(redundant)
    redundant_sorted = np.argsort(similarity[redundant])[::-1]
    included_idxs = set(range(len(embedded_documents)))
    for first_idx, second_idx in redundant_stacked[redundant_sorted]:
        if first_idx in included_idxs and second_idx in included_idxs:
            # Default to dropping the second document of any highly similar pair.
            included_idxs.remove(second_idx)
    return list(sorted(included_idxs))


def _get_embeddings_from_stateful_docs(
    embeddings: Embeddings, documents: Sequence[_DocumentWithState]
) -> List[List[float]]:
    if len(documents) and "embedded_doc" in documents[0].state:
        embedded_documents = [doc.state["embedded_doc"] for doc in documents]
    else:
        embedded_documents = embeddings.embed_documents(
            [d.page_content for d in documents]
        )
        for doc, embedding in zip(documents, embedded_documents):
            doc.state["embedded_doc"] = embedding
    return embedded_documents


async def _aget_embeddings_from_stateful_docs(
    embeddings: Embeddings, documents: Sequence[_DocumentWithState]
) -> List[List[float]]:
    if len(documents) and "embedded_doc" in documents[0].state:
        embedded_documents = [doc.state["embedded_doc"] for doc in documents]
    else:
        embedded_documents = await embeddings.aembed_documents(
            [d.page_content for d in documents]
        )
        for doc, embedding in zip(documents, embedded_documents):
            doc.state["embedded_doc"] = embedding
    return embedded_documents


def _filter_cluster_embeddings(
    embedded_documents: List[List[float]],
    num_clusters: int,
    num_closest: int,
    random_state: int,
    remove_duplicates: bool,
) -> List[int]:
    """Filter documents based on proximity of their embeddings to clusters."""

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "sklearn package not found, please install it with "
            "`pip install scikit-learn`"
        )

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state).fit(
        embedded_documents
    )
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(
            embedded_documents - kmeans.cluster_centers_[i], axis=1
        )

        # Find the indices of the two unique closest ones
        # (using argsort to find the smallest 2 distances)
        if remove_duplicates:
            # Only add not duplicated vectors.
            closest_indices_sorted = [
                x
                for x in np.argsort(distances)[:num_closest]
                if x not in closest_indices
            ]
        else:
            # Skip duplicates and add the next closest vector.
            closest_indices_sorted = [
                x for x in np.argsort(distances) if x not in closest_indices
            ][:num_closest]

        # Append that position closest indices list
        closest_indices.extend(closest_indices_sorted)

    return closest_indices


class EmbeddingsRedundantFilter(BaseDocumentTransformer, BaseModel):
    """Filter that drops redundant documents by comparing their embeddings."""

    embeddings: Embeddings
    """Embeddings to use for embedding document contents."""
    similarity_fn: Callable = cosine_similarity
    """Similarity function for comparing documents. Function expected to take as input
    two matrices (List[List[float]]) and return a matrix of scores where higher values
    indicate greater similarity."""
    similarity_threshold: float = 0.95
    """Threshold for determining when two documents are similar enough
    to be considered redundant."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Filter down documents."""
        stateful_documents = get_stateful_documents(documents)
        embedded_documents = _get_embeddings_from_stateful_docs(
            self.embeddings, stateful_documents
        )
        included_idxs = _filter_similar_embeddings(
            embedded_documents, self.similarity_fn, self.similarity_threshold
        )
        return [stateful_documents[i] for i in sorted(included_idxs)]


class EmbeddingsClusteringFilter(BaseDocumentTransformer, BaseModel):
    """Perform K-means clustering on document vectors.
    Returns an arbitrary number of documents closest to center."""

    embeddings: Embeddings
    """Embeddings to use for embedding document contents."""

    num_clusters: int = 5
    """Number of clusters. Groups of documents with similar meaning."""

    num_closest: int = 1
    """The number of closest vectors to return for each cluster center."""

    random_state: int = 42
    """Controls the random number generator used to initialize the cluster centroids.
    If you set the random_state parameter to None, the KMeans algorithm will use a 
    random number generator that is seeded with the current time. This means 
    that the results of the KMeans algorithm will be different each time you 
    run it."""

    sorted: bool = False
    """By default results are re-ordered "grouping" them by cluster, if sorted is true
    result will be ordered by the original position from the retriever"""

    remove_duplicates: bool = False
    """ By default duplicated results are skipped and replaced by the next closest 
    vector in the cluster. If remove_duplicates is true no replacement will be done:
    This could dramatically reduce results when there is a lot of overlap between 
    clusters.
    """

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Filter down documents."""
        stateful_documents = get_stateful_documents(documents)
        embedded_documents = _get_embeddings_from_stateful_docs(
            self.embeddings, stateful_documents
        )
        included_idxs = _filter_cluster_embeddings(
            embedded_documents,
            self.num_clusters,
            self.num_closest,
            self.random_state,
            self.remove_duplicates,
        )
        results = sorted(included_idxs) if self.sorted else included_idxs
        return [stateful_documents[i] for i in results]
