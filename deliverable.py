from typing import List, Dict, Any, Optional, Tuple
import math
import threading

class VectorStore:
    """
    In-memory vector store with compaction and hybrid BM25 + embedding search.
    """

    def __init__(self, embedding_dim: int) -> None:
        """
        Initialize the vector store.

        Args:
            embedding_dim (int): Dimension of the embedding vectors.
        """
        self.embedding_dim = embedding_dim
        self._lock = threading.Lock()
        self._docs: List[Dict[str, Any]] = []
        self._inverted_index: Dict[str, List[int]] = {}
        self._doc_freq: Dict[str, int] = {}
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0
        self._dirty: bool = False

    def add_document(self, doc_id: str, text: str, embedding: List[float]) -> None:
        """
        Add a document to the store.

        Args:
            doc_id (str): Unique document identifier.
            text (str): Document text.
            embedding (List[float]): Embedding vector.
        """
        if len(embedding) != self.embedding_dim:
            raise ValueError("Embedding dimension mismatch.")

        with self._lock:
            tokens = self._tokenize(text)
            doc_idx = len(self._docs)
            self._docs.append({
                "id": doc_id,
                "text": text,
                "embedding": embedding,
                "tokens": tokens,
            })
            self._doc_lengths.append(len(tokens))
            for token in set(tokens):
                self._inverted_index.setdefault(token, []).append(doc_idx)
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
            self._dirty = True

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by its ID.

        Args:
            doc_id (str): Document identifier.

        Returns:
            bool: True if deleted, False if not found.
        """
        with self._lock:
            for idx, doc in enumerate(self._docs):
                if doc["id"] == doc_id:
                    tokens = doc["tokens"]
                    for token in set(tokens):
                        if token in self._inverted_index:
                            try:
                                self._inverted_index[token].remove(idx)
                            except ValueError:
                                pass
                            if not self._inverted_index[token]:
                                del self._inverted_index[token]
                        if token in self._doc_freq:
                            self._doc_freq[token] -= 1
                            if self._doc_freq[token] == 0:
                                del self._doc_freq[token]
                    del self._docs[idx]
                    del self._doc_lengths[idx]
                    self._dirty = True
                    return True
            return False

    def compact(self) -> None:
        """
        Compact the vector store to remove gaps and update indices.
        """
        with self._lock:
            # Rebuild inverted index and doc_freq
            self._inverted_index.clear()
            self._doc_freq.clear()
            self._doc_lengths = []
            for idx, doc in enumerate(self._docs):
                tokens = doc["tokens"]
                self._doc_lengths.append(len(tokens))
                for token in set(tokens):
                    self._inverted_index.setdefault(token, []).append(idx)
                    self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
            self._dirty = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenizer.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        return text.lower().split()

    def _bm25_score(
        self, query_tokens: List[str], doc_tokens: List[str], k1: float = 1.5, b: float = 0.75
    ) -> float:
        """
        Compute BM25 score for a document.

        Args:
            query_tokens (List[str]): Query tokens.
            doc_tokens (List[str]): Document tokens.
            k1 (float): BM25 k1 parameter.
            b (float): BM25 b parameter.

        Returns:
            float: BM25 score.
        """
        score = 0.0
        doc_len = len(doc_tokens)
        avgdl = self._avg_doc_length if self._avg_doc_length > 0 else 1.0
        N = len(self._docs)
        tf = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1
        for token in query_tokens:
            if token not in self._doc_freq:
                continue
            df = self._doc_freq[token]
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            freq = tf.get(token, 0)
            denom = freq + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * (freq * (k1 + 1)) / (denom + 1e-9)
        return score

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            v1 (List[float]): First vector.
            v2 (List[float]): Second vector.

        Returns:
            float: Cosine similarity.
        """
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _update_avg_doc_length(self) -> None:
        """
        Update average document length.
        """
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        else:
            self._avg_doc_length = 0.0

    def search(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Hybrid search using BM25 and embedding similarity.

        Args:
            query_text (str): Query text.
            query_embedding (Optional[List[float]]): Query embedding vector.
            top_k (int): Number of top results to return.
            alpha (float): Weight for BM25 (0.0 = only embedding, 1.0 = only BM25).

        Returns:
            List[Tuple[str, float]]: List of (doc_id, score) tuples.
        """
        with self._lock:
            if self._dirty:
                self.compact()
            self._update_avg_doc_length()
            query_tokens = self._tokenize(query_text)
            bm25_scores = []
            emb_scores = []
            for doc in self._docs:
                bm25 = self._bm25_score(query_tokens, doc["tokens"])
                bm25_scores.append(bm25)
                if query_embedding is not None:
                    emb = self._cosine_similarity(query_embedding, doc["embedding"])
                else:
                    emb = 0.0
                emb_scores.append(emb)
            # Normalize scores
            bm25_max = max(bm25_scores) if bm25_scores else 1.0
            emb_max = max(emb_scores) if emb_scores else 1.0
            results = []
            for idx, doc in enumerate(self._docs):
                bm25_norm = bm25_scores[idx] / (bm25_max + 1e-9) if bm25_max > 0 else 0.0
                emb_norm = emb_scores[idx] / (emb_max + 1e-9) if emb_max > 0 else 0.0
                score = alpha * bm25_norm + (1 - alpha) * emb_norm
                results.append((doc["id"], score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.

        Args:
            doc_id (str): Document identifier.

        Returns:
            Optional[Dict[str, Any]]: Document dict or None.
        """
        with self._lock:
            for doc in self._docs:
                if doc["id"] == doc_id:
                    return doc.copy()
            return None

# --- TESTS ---

def _test_vector_store() -> None:
    store = VectorStore(embedding_dim=3)
    store.add_document("doc1", "the quick brown fox", [0.1, 0.2, 0.3])
    store.add_document("doc2", "jumps over the lazy dog", [0.2, 0.1, 0.4])
    store.add_document("doc3", "the quick blue hare", [0.3, 0.2, 0.1])
    # Hybrid search
    results = store.search("quick fox", [0.1, 0.2, 0.3], top_k=2, alpha=0.5)
    assert results[0][0] == "doc1", "doc1 should be most relevant"
    # BM25 only
    results_bm25 = store.search("lazy dog", None, top_k=1, alpha=1.0)
    assert results_bm25[0][0] == "doc2", "doc2 should be most relevant for BM25"
    # Embedding only
    results_emb = store.search("irrelevant", [0.3, 0.2, 0.1], top_k=1, alpha=0.0)
    assert results_emb[0][0] == "doc3", "doc3 should be most relevant for embedding"
    # Delete and compact
    assert store.delete_document("doc2") is True
    store.compact()
    assert store.get_document("doc2") is None
    print("All tests passed.")

if __name__ == "__main__":
    _test_vector_store()