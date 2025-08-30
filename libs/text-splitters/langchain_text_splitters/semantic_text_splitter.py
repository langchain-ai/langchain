"""
langchain/text_splitter/semantic_text_splitter.py
SemanticTextSplitter: Splits text into semantically meaningful chunks
using embeddings + ML clustering.
Dependencies: numpy, scikit-learn
"""

from typing import List, Callable, Optional, Literal, Dict, Any
from langchain.text_splitter import TextSplitter
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np


class SemanticTextSplitter(TextSplitter):
    """
    A semantic text splitter that uses embeddings + ML clustering 
    to split text into semantically coherent chunks.

    Modes:
        - "similarity": splits when adjacent sentence similarity < threshold
        - "clustering": groups sentences using KMeans or Agglomerative clustering

    Args:
        embedding_model: Any model with `.embed_documents(List[str]) -> List[List[float]]`.
        sentence_splitter: Callable[[str], List[str]] to split text into sentences.
        mode: 'similarity' or 'clustering'.
        similarity_threshold: Cosine similarity threshold for similarity mode.
        n_clusters: Number of clusters in clustering mode (optional).
        clustering_method: 'kmeans' or 'agglomerative'.
        max_chunk_size: Max character length of each chunk (optional).
        random_state: Random seed for reproducibility (clustering).

    Example:
        >>> from langchain.embeddings import OpenAIEmbeddings
        >>> from langchain.schema import Document
        >>> def split_sentences(text: str) -> List[str]:
        ...     return text.split('. ')
        >>> splitter = SemanticTextSplitter(
        ...     embedding_model=OpenAIEmbeddings(),
        ...     sentence_splitter=split_sentences,
        ...     mode="similarity",
        ...     similarity_threshold=0.7
        ... )
        >>> doc = Document(page_content="Hello world. This is a test. Another sentence.")
        >>> chunks = splitter.split_documents([doc])
    """

    def __init__(
        self,
        embedding_model: Any,
        sentence_splitter: Callable[[str], List[str]],
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

    def split_text(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks."""
        # Filter out empty sentences
        sentences = [s for s in self.sentence_splitter(text) if s.strip()]
        if not sentences:
            return []

        if len(sentences) == 1:
            return [sentences[0]]

        embeddings = np.array(self.embedding_model.embed_documents(sentences))
        if embeddings.ndim != 2:
            raise ValueError("Embeddings should be a 2D array (n_sentences x embedding_dim)")

        if self.mode == "similarity":
            chunks = self._split_by_similarity(sentences, embeddings)
        elif self.mode == "clustering":
            chunks = self._split_by_clustering(sentences, embeddings)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return self._apply_max_chunk_size(chunks)



    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of Documents into semantically coherent chunks."""
        all_docs: List[Document] = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                all_docs.append(Document(page_content=chunk, metadata=doc.metadata))
        return all_docs

    # ---------- Private Helpers ----------

    def _split_by_similarity(self, sentences: List[str], embeddings: np.ndarray) -> List[str]:
        """Split when cosine similarity between adjacent sentences drops below threshold."""
        chunks, current_chunk = [], [sentences[0]]
        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            if sim < self.similarity_threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        chunks.append(" ".join(current_chunk))
        return chunks

    def _split_by_clustering(self, sentences: List[str], embeddings: np.ndarray) -> List[str]:
        """Split by clustering sentences into semantically similar groups."""
        n_clusters = self.n_clusters or max(2, len(sentences) // 3)

        if self.clustering_method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters)

        labels = model.fit_predict(embeddings)
        clustered: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
        for sent, label in zip(sentences, labels):
            clustered[label].append(sent)

        return [" ".join(clustered[i]) for i in sorted(clustered.keys())]

    def _apply_max_chunk_size(self, chunks: List[str]) -> List[str]:
        if not self.max_chunk_size:
            return chunks

        final_chunks = []
        for chunk in chunks:
            while len(chunk) > self.max_chunk_size:
                # split at nearest space before max_chunk_size
                split_pos = chunk.rfind(" ", 0, self.max_chunk_size)
                if split_pos == -1:
                    split_pos = self.max_chunk_size
                final_chunks.append(chunk[:split_pos])
                chunk = chunk[split_pos:].lstrip()
            if chunk:
                final_chunks.append(chunk)
        return final_chunks
