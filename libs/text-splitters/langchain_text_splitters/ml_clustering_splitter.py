"""ML-based text clustering splitter for LangChain."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from langchain_text_splitters.base import TextSplitter

logger = logging.getLogger(__name__)


class MLClusteringSplitter(TextSplitter):
    """Text splitter using ML clustering techniques.
    
    This splitter uses machine learning clustering algorithms to group
    semantically similar text chunks together, then splits the text
    based on cluster boundaries.
    
    Args:
        chunk_size: Maximum size of chunks to return.
        chunk_overlap: Overlap in characters between chunks.
        length_function: Function that measures the length of given chunks.
        keep_separator: Whether to keep the separator in the chunks.
        add_start_index: If `True`, includes chunk's start index in metadata.
        strip_whitespace: If `True`, strips whitespace from the start and end of 
                         every document.
        model_name: Name of the embedding model to use for clustering.
        n_clusters: Number of clusters to create. If None, will be determined automatically.
        min_cluster_size: Minimum size of clusters.
        clustering_algorithm: Algorithm to use for clustering ('kmeans', 'hierarchical').
    """

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: callable = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
        model_name: str = "all-MiniLM-L6-v2",
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 50,
        clustering_algorithm: str = "kmeans",
    ) -> None:
        """Initialize the ML clustering splitter."""
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            keep_separator=keep_separator,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )
        self._model_name = model_name
        self._n_clusters = n_clusters
        self._min_cluster_size = min_cluster_size
        self._clustering_algorithm = clustering_algorithm
        self._embedding_model = None

    def _get_embedding_model(self) -> Any:
        """Get or initialize the embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self._model_name)
            except ImportError as e:
                logger.error(
                    "Could not import sentence_transformers. "
                    "Please install it with `pip install sentence-transformers`."
                )
                raise ImportError(
                    "Could not import sentence_transformers. "
                    "Please install it with `pip install sentence-transformers`."
                ) from e
        return self._embedding_model

    def _cluster_texts(self, texts: List[str]) -> List[int]:
        """Cluster texts using the specified algorithm."""
        if len(texts) < 2:
            return [0] * len(texts)
        
        try:
            import numpy as np
            from sklearn.cluster import KMeans, AgglomerativeClustering
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as e:
            logger.error(
                "Could not import required ML libraries. "
                "Please install scikit-learn and numpy."
            )
            raise ImportError(
                "Could not import required ML libraries. "
                "Please install with `pip install scikit-learn numpy`."
            ) from e

        # Get embeddings
        model = self._get_embedding_model()
        embeddings = model.encode(texts)
        
        # Determine number of clusters if not specified
        n_clusters = self._n_clusters
        if n_clusters is None:
            n_clusters = max(1, min(len(texts) // self._min_cluster_size, 10))
        
        n_clusters = min(n_clusters, len(texts))
        
        # Perform clustering
        if self._clustering_algorithm == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif self._clustering_algorithm == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(
                f"Unknown clustering algorithm: {self._clustering_algorithm}. "
                "Choose from 'kmeans' or 'hierarchical'."
            )
        
        labels = clusterer.fit_predict(embeddings)
        return labels.tolist()

    def _split_text_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except ImportError:
            # Fallback to simple sentence splitting
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def split_text(self, text: str) -> List[str]:
        """Split text using ML clustering."""
        if not text.strip():
            return []
        
        # First split into sentences
        sentences = self._split_text_into_sentences(text)
        
        if len(sentences) <= 1:
            return sentences
        
        # Cluster sentences
        try:
            cluster_labels = self._cluster_texts(sentences)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}. Falling back to character splitting.")
            # Fallback to character-based splitting
            return self._fallback_split(text)
        
        # Group sentences by cluster
        clusters = {}
        for i, (sentence, label) in enumerate(zip(sentences, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((i, sentence))
        
        # Sort clusters by the position of their first sentence
        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1][0][0])
        
        # Combine sentences within each cluster and respect size limits
        chunks = []
        for _, cluster_sentences in sorted_clusters:
            cluster_text = " ".join([s[1] for s in cluster_sentences])
            
            # If cluster is too large, split it further
            if self._length_function(cluster_text) > self._chunk_size:
                sub_chunks = self._split_large_cluster(cluster_text)
                chunks.extend(sub_chunks)
            else:
                chunks.append(cluster_text)
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _split_large_cluster(self, text: str) -> List[str]:
        """Split a large cluster into smaller chunks."""
        # Use character-based splitting for large clusters
        chunks = []
        current_chunk = ""
        
        sentences = self._split_text_into_sentences(text)
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self._length_function(test_chunk) <= self._chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                
                # Handle very long sentences
                if self._length_function(current_chunk) > self._chunk_size:
                    # Split the sentence by words
                    words = current_chunk.split()
                    temp_chunk = ""
                    
                    for word in words:
                        test_temp = temp_chunk + " " + word if temp_chunk else word
                        if self._length_function(test_temp) <= self._chunk_size:
                            temp_chunk = test_temp
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            temp_chunk = word
                    
                    current_chunk = temp_chunk
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _fallback_split(self, text: str) -> List[str]:
        """Fallback to simple character-based splitting."""
        chunks = []
        current_chunk = ""
        
        for char in text:
            if self._length_function(current_chunk + char) <= self._chunk_size:
                current_chunk += char
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = char
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> MLClusteringSplitter:
        """Create a splitter from a HuggingFace tokenizer."""
        try:
            length_function = lambda text: len(tokenizer.encode(text))
        except AttributeError:
            length_function = len
            
        return cls(length_function=length_function, **kwargs)
