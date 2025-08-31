"""SemanticTextSplitter: Splits text into semantically meaningful chunks.

Uses embeddings and ML clustering to create coherent text chunks.

Dependencies: numpy, scikit-learn.
"""

from collections.abc import Iterable
from typing import Any, Callable, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, KMeans  # type: ignore[import]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter


class SemanticTextSplitter(TextSplitter):
    """A semantic text splitter that uses embeddings + ML clustering.

    Modes:
        - "similarity": splits when adjacent sentence similarity < threshold.
        - "clustering": groups sentences using KMeans or Agglomerative clustering.

    Args:
        embedding_model: Model with `.embed_documents(list[str]) -> list[list[float]]`.
        sentence_splitter: Function to split text into sentences.
        mode: 'similarity' or 'clustering'.
        similarity_threshold: Cosine similarity threshold.
        n_clusters: Number of clusters (optional).
        clustering_method: 'kmeans' or 'agglomerative'.
        max_chunk_size: Max character length (optional).
        random_state: Random seed for clustering.

    Examples:
        >>> from langchain.embeddings import OpenAIEmbeddings
        >>> from langchain_core.documents import Document
        >>> def split_sentences(text: str) -> list[str]:
        ...     return text.split(". ")
        >>> splitter = SemanticTextSplitter(
        ...     embedding_model=OpenAIEmbeddings(),
        ...     sentence_splitter=split_sentences,
        ...     mode="similarity",
        ...     similarity_threshold=0.7
        ... )
        >>> doc = Document(page_content="Hello world.")
        >>> chunks = splitter.split_documents([doc])
        >>> chunks  # doctest: +ELLIPSIS
        [Document(page_content='Hello world.', ...)]
    """

    def __init__(
        self,
        embedding_model: Any,
        sentence_splitter: Callable[[str], list[str]],
        mode: Literal["similarity", "clustering"] = "similarity",
        similarity_threshold: float = 0.7,
        n_clusters: Optional[int] = None,
        clustering_method: Literal["kmeans", "agglomerative"] = "kmeans",
        max_chunk_size: Optional[int] = None,
        random_state: int = 42,
    ):
        self.embedding_model = embedding_model
        self.sentence_splitter = sentence_splitter
        self.mode = mode
        self.similarity_threshold = similarity_threshold
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.max_chunk_size = max_chunk_size
        self.random_state = random_state

    def split_text(self, text: str) -> list[str]:
        """Split text into semantically coherent chunks."""
        sentences = [s for s in self.sentence_splitter(text) if s.strip()]
        if not sentences:
            return []

        if len(sentences) == 1:
            return [sentences[0]]

        embeddings: NDArray[np.float_] = np.array(
            self.embedding_model.embed_documents(sentences)
        )
        if embeddings.ndim != 2:
            msg = "Embeddings should be a 2D array (n_sentences x embedding_dim)"
            raise ValueError(msg)

        if self.mode == "similarity":
            chunks = self._split_by_similarity(sentences, embeddings)
        elif self.mode == "clustering":
            chunks = self._split_by_clustering(sentences, embeddings)
        else:
            msg = f"Unknown mode: {self.mode}"
            raise ValueError(msg)

        return self._apply_max_chunk_size(chunks)

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Split a list of Documents into semantically coherent chunks."""
        return [
            Document(page_content=chunk, metadata=doc.metadata)
            for doc in documents
            for chunk in self.split_text(doc.page_content)
        ]

    def _split_by_similarity(
        self, sentences: list[str], embeddings: NDArray[np.float_]
    ) -> list[str]:
        """Split sentences when cosine similarity drops below threshold."""
        chunks: list[str] = []
        current_chunk: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            if sim < self.similarity_threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        chunks.append(" ".join(current_chunk))
        return chunks

    def _split_by_clustering(
        self, sentences: list[str], embeddings: NDArray[np.float_]
    ) -> list[str]:
        """Split sentences by clustering them into similar groups."""
        n_clusters = self.n_clusters or max(2, len(sentences) // 3)

        if self.clustering_method == "kmeans":
            model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init="auto",
            )
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters)

        labels = model.fit_predict(embeddings)
        clustered: dict[int, list[str]] = {i: [] for i in range(n_clusters)}
        for sent, label in zip(sentences, labels):
            clustered[label].append(sent)

        return [" ".join(clustered[i]) for i in sorted(clustered.keys())]

    def _apply_max_chunk_size(self, chunks: list[str]) -> list[str]:
        """Split chunks further if they exceed max_chunk_size."""
        if not self.max_chunk_size:
            return chunks

        final_chunks: list[str] = []
        for chunk in chunks:
            while len(chunk) > self.max_chunk_size:
                split_pos = chunk.rfind(" ", 0, self.max_chunk_size)
                if split_pos == -1:
                    split_pos = self.max_chunk_size
                final_chunks.append(chunk[:split_pos])
                chunk = chunk[split_pos:].lstrip()
            if chunk:
                final_chunks.append(chunk)

        return final_chunks
