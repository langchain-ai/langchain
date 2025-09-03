"""ML Clustering-based Text Splitter for LangChain

This module implements a text splitter that uses machine learning clustering
to semantically group sentences and create coherent chunks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter

# Optional dependencies with graceful degradation
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLClusteringTextSplitter(TextSplitter):
    """Split text using ML clustering approach.
    
    This splitter uses sentence embeddings and K-means clustering to group
    semantically similar sentences together, creating more coherent chunks
    than traditional character or token-based splitting.
    
    The approach:
    1. Split text into sentences
    2. Generate embeddings for each sentence
    3. Cluster sentences using K-means
    4. Group sentences by cluster to form chunks
    5. Ensure chunks meet size constraints
    
    Attributes:
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters  
        model_name: Sentence transformer model name
        min_chunk_size: Minimum chunk size in characters
        max_clusters: Maximum number of clusters to create
        sentence_splitter: Function to split text into sentences
    """

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        model_name: str = "all-MiniLM-L6-v2",
        min_chunk_size: int = 100,
        max_clusters: Optional[int] = None,
        sentence_splitter: Optional[callable] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ML clustering text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            model_name: Name of the sentence transformer model to use
            min_chunk_size: Minimum size of each chunk in characters
            max_clusters: Maximum number of clusters (auto-determined if None)
            sentence_splitter: Custom sentence splitting function
            **kwargs: Additional arguments passed to parent class
        
        Raises:
            ImportError: If required dependencies are not installed
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for MLClusteringTextSplitter. "
                "Please install it with: pip install sentence-transformers"
            )
            
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for MLClusteringTextSplitter. "
                "Please install it with: pip install scikit-learn"
            )
        
        self._model_name = model_name
        self._min_chunk_size = min_chunk_size
        self._max_clusters = max_clusters
        self._sentence_splitter = sentence_splitter or self._default_sentence_split
        self._model = None
        
    def _load_model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            logger.info(f"Loading sentence transformer model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
        return self._model
    
    def _default_sentence_split(self, text: str) -> List[str]:
        """Default sentence splitting using simple heuristics."""
        # Split on sentence endings, keeping the punctuation
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _determine_optimal_clusters(
        self, 
        embeddings: np.ndarray, 
        min_clusters: int = 2,
        max_clusters: Optional[int] = None
    ) -> int:
        """Determine optimal number of clusters using silhouette analysis."""
        n_samples = len(embeddings)
        
        if max_clusters is None:
            # Heuristic: roughly one cluster per chunk_size worth of text
            max_clusters = min(n_samples // 2, max(2, n_samples // 10))
        
        max_clusters = min(max_clusters, n_samples - 1)
        min_clusters = min(min_clusters, max_clusters)
        
        if min_clusters >= max_clusters:
            return min_clusters
            
        best_score = -1
        best_k = min_clusters
        
        for k in range(min_clusters, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Calculate silhouette score
                score = silhouette_score(embeddings, cluster_labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            except Exception as e:
                logger.warning(f"Error computing clusters for k={k}: {e}")
                continue
        
        logger.info(f"Optimal clusters: {best_k} (silhouette score: {best_score:.3f})")
        return best_k
    
    def _cluster_sentences(self, sentences: List[str]) -> List[List[int]]:
        """Cluster sentences and return grouped indices."""
        if len(sentences) <= 1:
            return [list(range(len(sentences)))]
            
        # Generate embeddings
        model = self._load_model()
        embeddings = model.encode(sentences)
        
        # Determine optimal number of clusters
        n_clusters = self._determine_optimal_clusters(
            embeddings, 
            max_clusters=self._max_clusters
        )
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group sentence indices by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
            
        return list(clusters.values())
    
    def _create_chunks_from_clusters(
        self, 
        sentences: List[str], 
        clusters: List[List[int]]
    ) -> List[str]:
        """Create text chunks from clustered sentences."""
        chunks = []
        
        for cluster_indices in clusters:
            # Sort indices to maintain original order
            cluster_indices.sort()
            cluster_sentences = [sentences[i] for i in cluster_indices]
            cluster_text = " ".join(cluster_sentences)
            
            # Check if cluster exceeds chunk size
            if len(cluster_text) <= self._chunk_size:
                chunks.append(cluster_text)
            else:
                # Split large clusters recursively
                if len(cluster_sentences) > 1:
                    # Recursively cluster this subset
                    sub_clusters = self._cluster_sentences(cluster_sentences)
                    sub_chunks = self._create_chunks_from_clusters(
                        cluster_sentences, sub_clusters
                    )
                    chunks.extend(sub_chunks)
                else:
                    # Single sentence too large, split by character
                    char_chunks = self._split_by_character(cluster_text)
                    chunks.extend(char_chunks)
        
        return chunks
    
    def _split_by_character(self, text: str) -> List[str]:
        """Fallback character-based splitting for oversized content."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self._chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
                
            # Try to break at word boundary
            while end > start and text[end] not in ' \n\t':
                end -= 1
            
            if end == start:  # No good break point found
                end = start + self._chunk_size
                
            chunks.append(text[start:end])
            start = end - self._chunk_overlap if self._chunk_overlap > 0 else end
            
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between chunks if specified."""
        if self._chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
            
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue
                
            # Add overlap from previous chunk
            prev_chunk = chunks[i - 1]
            overlap_text = prev_chunk[-self._chunk_overlap:] if len(prev_chunk) > self._chunk_overlap else prev_chunk
            
            # Find good break point in overlap
            overlap_sentences = self._sentence_splitter(overlap_text)
            if len(overlap_sentences) > 1:
                overlap_text = " ".join(overlap_sentences[-2:])  # Last 1-2 sentences
            
            combined_chunk = overlap_text + " " + chunk
            overlapped_chunks.append(combined_chunk)
            
        return overlapped_chunks
    
    def split_text(self, text: str) -> List[str]:
        """Split text using ML clustering approach.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
            
        # Split into sentences
        sentences = self._sentence_splitter(text)
        
        if not sentences:
            return []
            
        if len(sentences) == 1:
            # Single sentence - check if it needs character splitting
            if len(sentences[0]) > self._chunk_size:
                return self._split_by_character(sentences[0])
            return sentences
        
        # Cluster sentences
        clusters = self._cluster_sentences(sentences)
        
        # Create chunks from clusters
        chunks = self._create_chunks_from_clusters(sentences, clusters)
        
        # Apply overlap
        chunks = self._apply_overlap(chunks)
        
        # Filter out chunks that are too small
        filtered_chunks = [
            chunk for chunk in chunks 
            if len(chunk.strip()) >= self._min_chunk_size
        ]
        
        return filtered_chunks or chunks  # Return original if filtering removes all
    
    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents into chunks while preserving metadata.
        
        Args:
            documents: Iterable of Document objects to split
            
        Returns:
            List of Document objects with split text
        """
        texts = []
        metadatas = []
        
        for doc in documents:
            texts.extend(self.split_text(doc.page_content))
            metadatas.extend([doc.metadata] * len(self.split_text(doc.page_content)))
            
        return [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]


# Convenience function for easy import
def create_ml_clustering_splitter(
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    model_name: str = "all-MiniLM-L6-v2",
    **kwargs: Any,
) -> MLClusteringTextSplitter:
    """Create an ML clustering text splitter with default parameters.
    
    Args:
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks  
        model_name: Name of the sentence transformer model to use
        **kwargs: Additional arguments for MLClusteringTextSplitter
        
    Returns:
        Configured MLClusteringTextSplitter instance
    """
    return MLClusteringTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name,
        **kwargs,
    )
