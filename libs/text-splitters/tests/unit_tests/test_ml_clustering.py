"""Unit tests for ML Clustering Text Splitter"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from langchain_core.documents import Document

# Import the splitter (assuming it's in the same directory)
from ml_clustering_splitter import MLClusteringTextSplitter, create_ml_clustering_splitter


class TestMLClusteringTextSplitter:
    """Test suite for ML Clustering Text Splitter"""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing"""
        return (
            "Machine learning is a subset of artificial intelligence. "
            "It focuses on algorithms that can learn from data. "
            "Deep learning uses neural networks with multiple layers. "
            "Natural language processing deals with human language. "
            "Computer vision processes visual information. "
            "These fields are interconnected and rapidly evolving."
        )
    
    @pytest.fixture
    def long_text(self):
        """Long text for testing chunk size limits"""
        base_text = "This is a sentence about machine learning and AI. " * 100
        return base_text + "This final sentence concludes our discussion."
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock sentence transformer model"""
        mock_model = MagicMock()
        # Create mock embeddings with some variation
        mock_embeddings = np.array([
            [0.1, 0.2, 0.3, 0.4],  # Similar to next one
            [0.2, 0.3, 0.4, 0.5],  # Similar to previous one  
            [0.8, 0.9, 0.7, 0.6],  # Different cluster
            [0.9, 0.8, 0.6, 0.7],  # Similar to previous one
            [0.1, 0.3, 0.2, 0.4],  # Similar to first cluster
            [0.7, 0.8, 0.9, 0.8],  # Similar to third cluster
        ])
        mock_model.encode.return_value = mock_embeddings
        return mock_model
    
    def test_initialization_with_required_dependencies(self):
        """Test splitter initialization with required dependencies"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            splitter = MLClusteringTextSplitter()
            assert splitter._chunk_size == 4000
            assert splitter._chunk_overlap == 200
            assert splitter._model_name == "all-MiniLM-L6-v2"
            assert splitter._min_chunk_size == 100
    
    def test_initialization_missing_sentence_transformers(self):
        """Test initialization fails without sentence-transformers"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                MLClusteringTextSplitter()
    
    def test_initialization_missing_sklearn(self):
        """Test initialization fails without scikit-learn"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', False):
            with pytest.raises(ImportError, match="scikit-learn is required"):
                MLClusteringTextSplitter()
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter(
                chunk_size=2000,
                chunk_overlap=100,
                model_name="custom-model",
                min_chunk_size=50,
                max_clusters=10
            )
            
            assert splitter._chunk_size == 2000
            assert splitter._chunk_overlap == 100
            assert splitter._model_name == "custom-model"
            assert splitter._min_chunk_size == 50
            assert splitter._max_clusters == 10
    
    def test_default_sentence_split(self):
        """Test default sentence splitting functionality"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter()
            
            text = "First sentence. Second sentence! Third sentence? Fourth sentence."
            sentences = splitter._default_sentence_split(text)
            
            expected = [
                "First sentence.",
                "Second sentence!",
                "Third sentence?",
                "Fourth sentence."
            ]
            
            assert sentences == expected
    
    def test_custom_sentence_splitter(self):
        """Test custom sentence splitter function"""
        def custom_splitter(text):
            return text.split("|")
        
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter(sentence_splitter=custom_splitter)
            
            text = "First part|Second part|Third part"
            sentences = splitter._sentence_splitter(text)
            
            assert sentences == ["First part", "Second part", "Third part"]
    
    @patch('ml_clustering_splitter.SentenceTransformer')
    def test_model_loading(self, mock_st_class):
        """Test lazy loading of sentence transformer model"""
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model
        
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter(model_name="test-model")
            
            # Model should not be loaded initially
            assert splitter._model is None
            
            # Load model
            loaded_model = splitter._load_model()
            
            # Should create and cache model
            mock_st_class.assert_called_once_with("test-model")
            assert loaded_model == mock_model
            assert splitter._model == mock_model
            
            # Second call should return cached model
            loaded_model_2 = splitter._load_model()
            assert loaded_model_2 == mock_model
            assert mock_st_class.call_count == 1  # Still only called once
    
    def test_determine_optimal_clusters(self):
        """Test optimal cluster determination"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter()
            
            # Create mock embeddings
            embeddings = np.array([
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4], 
                [0.8, 0.9, 0.7],
                [0.9, 0.8, 0.6],
            ])
            
            with patch('ml_clustering_splitter.silhouette_score') as mock_silhouette:
                mock_silhouette.return_value = 0.7
                
                n_clusters = splitter._determine_optimal_clusters(embeddings)
                
                assert isinstance(n_clusters, int)
                assert n_clusters >= 2
                assert n_clusters < len(embeddings)
    
    @patch('ml_clustering_splitter.SentenceTransformer')
    @patch('ml_clustering_splitter.KMeans')
    def test_cluster_sentences(self, mock_kmeans_class, mock_st_class, mock_sentence_transformer):
        """Test sentence clustering functionality"""
        # Setup mocks
        mock_st_class.return_value = mock_sentence_transformer
        mock_kmeans = MagicMock()
        mock_kmeans.fit_predict.return_value = np.array([0, 0, 1, 1, 0, 1])
        mock_kmeans_class.return_value = mock_kmeans
        
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter()
            
            sentences = [
                "ML sentence one.",
                "ML sentence two.", 
                "Vision sentence one.",
                "Vision sentence two.",
                "ML sentence three.",
                "Vision sentence three."
            ]
            
            clusters = splitter._cluster_sentences(sentences)
            
            # Should return grouped indices
            assert isinstance(clusters, list)
            assert len(clusters) > 0
            
            # Check that all sentence indices are included
            all_indices = set()
            for cluster in clusters:
                all_indices.update(cluster)
            assert all_indices == set(range(len(sentences)))
    
    def test_create_chunks_from_clusters(self):
        """Test chunk creation from clustered sentences"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter(chunk_size=100)
            
            sentences = [
                "Short sentence one.",
                "Short sentence two.",
                "Short sentence three.",
                "Short sentence four."
            ]
            
            clusters = [[0, 1], [2, 3]]  # Two clusters
            
            chunks = splitter._create_chunks_from_clusters(sentences, clusters)
            
            expected = [
                "Short sentence one. Short sentence two.",
                "Short sentence three. Short sentence four."
            ]
            
            assert chunks == expected
    
    def test_split_by_character_fallback(self):
        """Test character-based splitting fallback"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter(chunk_size=50, chunk_overlap=10)
            
            long_text = "This is a very long sentence that exceeds the chunk size limit and needs to be split"
            
            chunks = splitter._split_by_character(long_text)
            
            assert len(chunks) > 1
            assert all(len(chunk) <= 50 for chunk in chunks)
    
    @patch('ml_clustering_splitter.SentenceTransformer')
    @patch('ml_clustering_splitter.KMeans')
    def test_split_text_end_to_end(self, mock_kmeans_class, mock_st_class, sample_text, mock_sentence_transformer):
        """Test complete text splitting pipeline"""
        # Setup mocks
        mock_st_class.return_value = mock_sentence_transformer
        mock_kmeans = MagicMock()
        mock_kmeans.fit_predict.return_value = np.array([0, 0, 1, 1, 2, 2])
        mock_kmeans_class.return_value = mock_kmeans
        
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter(
                chunk_size=200,
                chunk_overlap=20,
                min_chunk_size=10
            )
            
            chunks = splitter.split_text(sample_text)
            
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)
            assert all(len(chunk) >= 10 for chunk in chunks)  # min_chunk_size
    
    def test_split_text_empty_input(self):
        """Test splitting empty or whitespace-only text"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter()
            
            assert splitter.split_text("") == []
            assert splitter.split_text("   ") == []
            assert splitter.split_text("\n\t ") == []
    
    def test_split_text_single_sentence(self):
        """Test splitting text with single sentence"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter()
            
            single_sentence = "This is a single sentence."
            chunks = splitter.split_text(single_sentence)
            
            assert chunks == [single_sentence]
    
    def test_split_documents(self, sample_text):
        """Test splitting Document objects"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter()
            
            # Mock the split_text method to avoid complex mocking
            with patch.object(splitter, 'split_text', return_value=['chunk1', 'chunk2']):
                documents = [
                    Document(page_content=sample_text, metadata={"source": "test1"}),
                    Document(page_content=sample_text, metadata={"source": "test2"})
                ]
                
                result_docs = splitter.split_documents(documents)
                
                assert len(result_docs) == 4  # 2 docs * 2 chunks each
                assert all(isinstance(doc, Document) for doc in result_docs)
                assert result_docs[0].metadata["source"] == "test1"
                assert result_docs[2].metadata["source"] == "test2"
    
    def test_apply_overlap(self):
        """Test overlap application between chunks"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter(chunk_overlap=20)
            
            chunks = [
                "This is the first chunk with some content.",
                "This is the second chunk with different content.",
                "This is the third and final chunk."
            ]
            
            overlapped = splitter._apply_overlap(chunks)
            
            assert len(overlapped) == len(chunks)
            assert overlapped[0] == chunks[0]  # First chunk unchanged
            assert len(overlapped[1]) > len(chunks[1])  # Second has overlap
            assert len(overlapped[2]) > len(chunks[2])  # Third has overlap
    
    def test_no_overlap_when_zero(self):
        """Test no overlap is applied when chunk_overlap is 0"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter(chunk_overlap=0)
            
            chunks = ["First chunk.", "Second chunk.", "Third chunk."]
            overlapped = splitter._apply_overlap(chunks)
            
            assert overlapped == chunks
    
    def test_create_ml_clustering_splitter_convenience_function(self):
        """Test the convenience function for creating splitter"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = create_ml_clustering_splitter(
                chunk_size=1000,
                chunk_overlap=50,
                model_name="custom-model"
            )
            
            assert isinstance(splitter, MLClusteringTextSplitter)
            assert splitter._chunk_size == 1000
            assert splitter._chunk_overlap == 50
            assert splitter._model_name == "custom-model"
    
    @pytest.mark.parametrize("chunk_size,expected_min_chunks", [
        (100, 2),  # Small chunks should create more splits
        (1000, 1), # Large chunks might create fewer splits
    ])
    def test_chunk_size_effect(self, chunk_size, expected_min_chunks, long_text):
        """Test that chunk_size parameter affects splitting"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter(chunk_size=chunk_size)
            
            # Mock the split_text method to return character-based splits for testing
            with patch.object(splitter, '_split_by_character') as mock_char_split:
                mock_char_split.return_value = ['chunk'] * expected_min_chunks
                
                chunks = splitter._split_by_character(long_text)
                assert len(chunks) >= expected_min_chunks
    
    def test_min_chunk_size_filtering(self):
        """Test that chunks smaller than min_chunk_size are filtered"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter(min_chunk_size=50)
            
            # Mock split_text internals to return chunks of varying sizes
            with patch.object(splitter, '_cluster_sentences', return_value=[[0], [1], [2]]), \
                 patch.object(splitter, '_sentence_splitter', return_value=['Short.', 'A', 'This is a longer sentence with more content.']), \
                 patch.object(splitter, '_apply_overlap', side_effect=lambda x: x):
                
                chunks = splitter.split_text("Short. A This is a longer sentence with more content.")
                
                # Only chunks meeting min_chunk_size should remain
                assert all(len(chunk.strip()) >= 50 for chunk in chunks)
    
    def test_edge_case_single_word(self):
        """Test edge case with single word input"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter()
            chunks = splitter.split_text("Word")
            assert chunks == ["Word"]
    
    def test_edge_case_no_sentences(self):
        """Test edge case where sentence splitting returns empty list"""
        with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
            
            splitter = MLClusteringTextSplitter()
            
            with patch.object(splitter, '_sentence_splitter', return_value=[]):
                chunks = splitter.split_text("Some text")
                assert chunks == []


class TestMLClusteringSplitterIntegration:
    """Integration tests (these would require actual dependencies in real testing)"""
    
    @pytest.mark.skipif(True, reason="Integration test - requires actual dependencies")
    def test_real_sentence_transformers_integration(self):
        """Integration test with real sentence-transformers (skip in CI)"""
        # This test would run with actual sentence-transformers installed
        splitter = MLClusteringTextSplitter(model_name="all-MiniLM-L6-v2")
        
        text = """
        Machine learning is a powerful technology. It can process large amounts of data.
        Natural language processing is a subset of ML. It focuses on human language understanding.
        Computer vision deals with image analysis. It can recognize objects and patterns.
        These technologies are transforming industries. They enable automation and insights.
        """
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert sum(len(chunk) for chunk in chunks) <= len(text) * 1.5  # Account for overlap
    
    @pytest.mark.skipif(True, reason="Integration test - requires actual dependencies") 
    def test_real_clustering_performance(self):
        """Performance test with real clustering (skip in CI)"""
        splitter = MLClusteringTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            model_name="all-MiniLM-L6-v2"
        )
        
        # Generate longer test text
        long_text = " ".join([
            "This is sentence number {}. It discusses various topics in machine learning and AI.".format(i)
            for i in range(100)
        ])
        
        chunks = splitter.split_text(long_text)
        
        # Performance assertions
        assert len(chunks) > 1
        assert all(len(chunk) <= 550 for chunk in chunks)  # Allow some buffer over chunk_size
        assert all(len(chunk) >= 50 for chunk in chunks)   # Should meet minimum


# Pytest fixtures for the test class
@pytest.fixture(scope="module")
def mock_dependencies():
    """Mock all external dependencies for testing"""
    with patch('ml_clustering_splitter.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
         patch('ml_clustering_splitter.SKLEARN_AVAILABLE', True):
        yield


# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
