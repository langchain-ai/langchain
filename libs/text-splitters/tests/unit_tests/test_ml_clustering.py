"""Tests for ML clustering splitter."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from langchain_text_splitters.ml_clustering_splitter import MLClusteringSplitter


class TestMLClusteringSplitter:
    """Test cases for MLClusteringSplitter."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        splitter = MLClusteringSplitter()
        assert splitter._chunk_size == 4000
        assert splitter._chunk_overlap == 200
        assert splitter._model_name == "all-MiniLM-L6-v2"
        assert splitter._n_clusters is None
        assert splitter._min_cluster_size == 50
        assert splitter._clustering_algorithm == "kmeans"

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        splitter = MLClusteringSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            model_name="custom-model",
            n_clusters=5,
            min_cluster_size=25,
            clustering_algorithm="hierarchical",
        )
        assert splitter._chunk_size == 1000
        assert splitter._chunk_overlap == 100
        assert splitter._model_name == "custom-model"
        assert splitter._n_clusters == 5
        assert splitter._min_cluster_size == 25
        assert splitter._clustering_algorithm == "hierarchical"

    def test_split_empty_text(self) -> None:
        """Test splitting empty or whitespace-only text."""
        splitter = MLClusteringSplitter()
        assert splitter.split_text("") == []
        assert splitter.split_text("   ") == []
        assert splitter.split_text("\n\n") == []

    def test_split_single_sentence(self) -> None:
        """Test splitting text with single sentence."""
        splitter = MLClusteringSplitter()
        text = "This is a single sentence."
        result = splitter.split_text(text)
        assert len(result) == 1
        assert result[0] == text

    @patch('langchain_text_splitters.ml_clustering_splitter.SentenceTransformer')
    @patch('sklearn.cluster.KMeans')
    def test_split_multiple_sentences_with_mocking(self, mock_kmeans, mock_transformer) -> None:
        """Test splitting multiple sentences with mocked dependencies."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_transformer.return_value = mock_model
        
        # Mock KMeans
        mock_clusterer = Mock()
        mock_clusterer.fit_predict.return_value = [0, 0, 1]
        mock_kmeans.return_value = mock_clusterer
        
        splitter = MLClusteringSplitter()
        text = "First sentence. Second sentence. Third sentence."
        
        with patch('nltk.download'), \
             patch('nltk.tokenize.sent_tokenize') as mock_sent_tokenize, \
             patch('numpy.array'), \
             patch('sklearn.metrics.pairwise.cosine_similarity'):
            
            mock_sent_tokenize.return_value = ["First sentence.", "Second sentence.", "Third sentence."]
            
            result = splitter.split_text(text)
            
            assert isinstance(result, list)
            assert len(result) > 0
            mock_transformer.assert_called_once()
            mock_model.encode.assert_called_once()

    def test_fallback_split_text_into_sentences_without_nltk(self) -> None:
        """Test sentence splitting fallback when NLTK is not available."""
        splitter = MLClusteringSplitter()
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'nltk'")):
            sentences = splitter._split_text_into_sentences("First sentence. Second sentence! Third question?")
            
            # Should still split on sentence boundaries
            assert len(sentences) >= 1
            assert any("First sentence" in s for s in sentences)

    def test_clustering_algorithm_validation(self) -> None:
        """Test validation of clustering algorithm parameter."""
        with pytest.raises(ValueError, match="Unknown clustering algorithm"):
            splitter = MLClusteringSplitter(clustering_algorithm="invalid")
            # Trigger the error by calling a method that uses clustering
            with patch('langchain_text_splitters.ml_clustering_splitter.SentenceTransformer'), \
                 patch('sklearn.cluster.KMeans'), \
                 patch('nltk.download'), \
                 patch('nltk.tokenize.sent_tokenize', return_value=["Test sentence."]):
                splitter._cluster_texts(["Test sentence."])

    def test_missing_dependencies_sentence_transformer(self) -> None:
        """Test handling of missing sentence-transformers dependency."""
        splitter = MLClusteringSplitter()
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'sentence_transformers'")):
            with pytest.raises(ImportError, match="Could not import sentence_transformers"):
                splitter._get_embedding_model()

    def test_missing_dependencies_sklearn(self) -> None:
        """Test handling of missing scikit-learn dependency."""
        splitter = MLClusteringSplitter()
        
        with patch('langchain_text_splitters.ml_clustering_splitter.SentenceTransformer'), \
             patch('builtins.__import__', side_effect=lambda name, *args: Mock() if name != 'sklearn.cluster' else ImportError()):
            
            # Mock the sklearn import to raise ImportError
            with patch('sklearn.cluster.KMeans', side_effect=ImportError("No module named 'sklearn'")):
                with pytest.raises(ImportError, match="Could not import required ML libraries"):
                    splitter._cluster_texts(["Test sentence."])

    def test_cluster_texts_single_text(self) -> None:
        """Test clustering with single text returns single cluster."""
        splitter = MLClusteringSplitter()
        result = splitter._cluster_texts(["Single text"])
        assert result == [0]

    def test_cluster_texts_empty_list(self) -> None:
        """Test clustering with empty list."""
        splitter = MLClusteringSplitter()
        result = splitter._cluster_texts([])
        assert result == []

    @patch('langchain_text_splitters.ml_clustering_splitter.SentenceTransformer')
    @patch('sklearn.cluster.AgglomerativeClustering')
    def test_hierarchical_clustering(self, mock_hierarchical, mock_transformer) -> None:
        """Test hierarchical clustering algorithm."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_transformer.return_value = mock_model
        
        # Mock AgglomerativeClustering
        mock_clusterer = Mock()
        mock_clusterer.fit_predict.return_value = [0, 1]
        mock_hierarchical.return_value = mock_clusterer
        
        splitter = MLClusteringSplitter(clustering_algorithm="hierarchical")
        
        with patch('numpy.array'), \
             patch('sklearn.metrics.pairwise.cosine_similarity'):
            result = splitter._cluster_texts(["Text one", "Text two"])
            
            assert result == [0, 1]
            mock_hierarchical.assert_called_once()

    def test_split_large_cluster(self) -> None:
        """Test splitting of large clusters."""
        splitter = MLClusteringSplitter(chunk_size=50)  # Small chunk size
        
        long_text = "This is a very long sentence that exceeds the chunk size limit. " * 10
        
        with patch.object(splitter, '_split_text_into_sentences') as mock_split:
            mock_split.return_value = [long_text]
            
            result = splitter._split_large_cluster(long_text)
            
            assert isinstance(result, list)
            assert len(result) > 0
            # Each chunk should be within size limits (with some tolerance for word boundaries)
            for chunk in result:
                assert len(chunk) <= splitter._chunk_size + 50  # Allow some tolerance

    def test_fallback_split(self) -> None:
        """Test fallback splitting mechanism."""
        splitter = MLClusteringSplitter(chunk_size=20)  # Small chunk size
        text = "This is a test text for fallback splitting."
        
        result = splitter._fallback_split(text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        for chunk in result:
            assert len(chunk) <= splitter._chunk_size

    def test_from_huggingface_tokenizer(self) -> None:
        """Test creation from HuggingFace tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        splitter = MLClusteringSplitter.from_huggingface_tokenizer(
            mock_tokenizer, 
            chunk_size=1000
        )
        
        assert splitter._chunk_size == 1000
        # Test that the length function uses tokenizer
        test_text = "Test text"
        length = splitter._length_function(test_text)
        assert length == 5  # Length of mock tokenizer output

    def test_from_huggingface_tokenizer_no_encode(self) -> None:
        """Test creation from tokenizer without encode method."""
        mock_tokenizer = Mock()
        del mock_tokenizer.encode  # Remove encode attribute
        
        splitter = MLClusteringSplitter.from_huggingface_tokenizer(mock_tokenizer)
        
        # Should fall back to len function
        test_text = "Test"
        length = splitter._length_function(test_text)
        assert length == 4  # Length of string

    def test_clustering_with_exception_fallback(self) -> None:
        """Test that clustering exceptions trigger fallback splitting."""
        splitter = MLClusteringSplitter()
        text = "First sentence. Second sentence. Third sentence."
        
        with patch.object(splitter, '_cluster_texts', side_effect=Exception("Clustering failed")):
            with patch.object(splitter, '_fallback_split', return_value=["fallback chunk"]) as mock_fallback:
                result = splitter.split_text(text)
                
                mock_fallback.assert_called_once_with(text)
                assert result == ["fallback chunk"]

    def test_automatic_cluster_number_determination(self) -> None:
        """Test automatic determination of cluster numbers."""
        splitter = MLClusteringSplitter(n_clusters=None, min_cluster_size=2)
        
        with patch('langchain_text_splitters.ml_clustering_splitter.SentenceTransformer') as mock_transformer, \
             patch('sklearn.cluster.KMeans') as mock_kmeans:
            
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2]] * 10  # 10 texts
            mock_transformer.return_value = mock_model
            
            mock_clusterer = Mock()
            mock_clusterer.fit_predict.return_value = [0] * 10
            mock_kmeans.return_value = mock_clusterer
            
            with patch('numpy.array'), \
                 patch('sklearn.metrics.pairwise.cosine_similarity'):
                
                texts = ["text"] * 10
                result = splitter._cluster_texts(texts)
                
                # Should determine n_clusters automatically
                mock_kmeans.assert_called_once()
                args = mock_kmeans.call_args[1]
                assert args['n_clusters'] == 5  # 10 // 2 = 5

    def test_integration_split_text_complete_flow(self) -> None:
        """Test complete flow of split_text method."""
        splitter = MLClusteringSplitter(chunk_size=100)
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        
        # Mock all dependencies
        with patch('langchain_text_splitters.ml_clustering_splitter.SentenceTransformer') as mock_transformer, \
             patch('sklearn.cluster.KMeans') as mock_kmeans, \
             patch('nltk.download'), \
             patch('nltk.tokenize.sent_tokenize') as mock_sent_tokenize, \
             patch('numpy.array'), \
             patch('sklearn.metrics.pairwise.cosine_similarity'):
            
            # Setup mocks
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4], [0.1, 0.2]]
            mock_transformer.return_value = mock_model
            
            mock_clusterer = Mock()
            mock_clusterer.fit_predict.return_value = [0, 1, 0]  # First and third in cluster 0, second in cluster 1
            mock_kmeans.return_value = mock_clusterer
            
            mock_sent_tokenize.return_value = [
                "This is the first sentence.",
                "This is the second sentence.",
                "This is the third sentence."
            ]
            
            result = splitter.split_text(text)
            
            assert isinstance(result, list)
            assert len(result) >= 1
            # Verify that clustering was attempted
            mock_transformer.assert_called_once()
            mock_model.encode.assert_called_once()


# Integration test that doesn't require external dependencies
class TestMLClusteringSplitterIntegration:
    """Integration tests that work without external ML libraries."""
    
    def test_basic_functionality_without_ml_libs(self) -> None:
        """Test that the splitter works even without ML libraries by falling back."""
        splitter = MLClusteringSplitter()
        
        # Test with a simple text that should trigger fallback
        text = "Short text."
        result = splitter.split_text(text)
        
        assert isinstance(result, list)
        assert len(result) >= 1
        assert "Short text." in result[0]

    def test_very_long_text_handling(self) -> None:
        """Test handling of very long text that exceeds chunk size."""
        splitter = MLClusteringSplitter(chunk_size=50)
        
        # Create a long text
        long_text = "This is a very long sentence. " * 20
        
        result = splitter.split_text(long_text)
        
        assert isinstance(result, list)
        assert len(result) > 1  # Should be split into multiple chunks
        
        # Verify no chunk exceeds the size limit significantly
        for chunk in result:
            # Allow some tolerance for word boundaries
            assert len(chunk) <= splitter._chunk_size + 100

    def test_strip_whitespace_functionality(self) -> None:
        """Test that whitespace stripping works correctly."""
        splitter = MLClusteringSplitter(strip_whitespace=True)
        
        text = "  \n\n  First sentence.  \n  Second sentence.  \n\n  "
        result = splitter.split_text(text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        for chunk in result:
            assert chunk == chunk.strip()  # Should not have leading/trailing whitespace


if __name__ == "__main__":
    pytest.main([__file__])
