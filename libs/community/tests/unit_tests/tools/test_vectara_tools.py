"""Unit tests for Vectara tools."""

import json
import unittest
from unittest.mock import MagicMock, patch

import pytest
from langchain_vectara import (
    CorpusConfig,
    GenerationConfig,
    SearchConfig,
    VectaraQueryConfig,
)

from langchain_community.tools.vectara import (
    VectaraGeneration,
    VectaraIngest,
    VectaraSearch,
)


@pytest.mark.requires("langchain-vectara")
class TestVectaraTools(unittest.TestCase):
    """Test Vectara tools functionality."""

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_search_with_config(self, mock_vectara: MagicMock) -> None:
        """Test VectaraSearch tool with config parameter."""

        mock_vectorstore = mock_vectara.return_value
        mock_vectorstore.similarity_search_with_score.return_value = [
            (MagicMock(page_content="Test content", metadata={"source": "test"}), 0.85)
        ]

        tool = VectaraSearch(
            name="test_search",
            description="Test search",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        corpus_config = CorpusConfig(
            corpus_key="test-corpus-123",
            metadata_filter="doc.type = 'article'",
            lexical_interpolation=0.2,
        )

        search_config = SearchConfig(corpora=[corpus_config], limit=10)

        query_config = VectaraQueryConfig(search=search_config)

        result = tool._run(
            query="test query",
            config=query_config,
        )

        mock_vectorstore.similarity_search_with_score.assert_called_once()
        call_kwargs = mock_vectorstore.similarity_search_with_score.call_args[1]
        assert "config" in call_kwargs
        assert isinstance(call_kwargs["config"], VectaraQueryConfig)

        # Verify result is valid JSON
        results_data = json.loads(result)
        assert isinstance(results_data, list)
        assert len(results_data) == 1
        assert results_data[0]["content"] == "Test content"
        assert results_data[0]["source"] == "test"
        assert results_data[0]["score"] == 0.85
        assert "index" in results_data[0]
        assert "metadata" in results_data[0]

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_search_without_config(self, mock_vectara: MagicMock) -> None:
        """Test VectaraSearch tool with default configuration."""

        mock_vectorstore = mock_vectara.return_value
        mock_vectorstore.similarity_search_with_score.return_value = [
            (MagicMock(page_content="Test content", metadata={"source": "test"}), 0.92)
        ]

        tool = VectaraSearch(
            name="test_search",
            description="Test search",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        result = tool._run(
            query="test query",
        )

        mock_vectorstore.similarity_search_with_score.assert_called_once()
        call_kwargs = mock_vectorstore.similarity_search_with_score.call_args[1]

        # Verify the config contains the corpus_key we provided
        config = call_kwargs["config"]
        assert config.search.corpora is not None
        assert len(config.search.corpora) == 1
        assert config.search.corpora[0].corpus_key == "test-corpus-123"

        # Verify result is valid JSON
        results_data = json.loads(result)
        assert isinstance(results_data, list)
        assert len(results_data) == 1
        assert results_data[0]["content"] == "Test content"
        assert results_data[0]["source"] == "test"
        assert results_data[0]["score"] == 0.92

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_generation_with_config(self, mock_vectara: MagicMock) -> None:
        """Test VectaraGeneration tool with config parameter."""

        mock_vectorstore = mock_vectara.return_value
        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {
            "answer": "Summary text",
            "fcs": 0.95,
            "context": [
                (
                    MagicMock(
                        page_content="Document content", metadata={"source": "test"}
                    ),
                    0.9,
                )
            ],
        }
        mock_vectorstore.as_rag.return_value = mock_rag

        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        # Test with config parameter containing search and generation settings
        corpus_config = CorpusConfig(
            corpus_key="test-corpus-123",
            metadata_filter="doc.type = 'article'",
            lexical_interpolation=0.2,
        )

        search_config = SearchConfig(corpora=[corpus_config], limit=10)

        generation_config = GenerationConfig(
            max_used_search_results=8,
            response_language="eng",
            generation_preset_name="test-prompt",
            enable_factual_consistency_score=True,
        )

        query_config = VectaraQueryConfig(
            search=search_config, generation=generation_config
        )

        result = tool._run(
            query="test query",
            config=query_config,
        )

        mock_vectorstore.as_rag.assert_called_once_with(query_config)
        mock_rag.invoke.assert_called_once_with("test query")

        # Verify result is valid JSON with summary
        summary_data = json.loads(result)
        assert isinstance(summary_data, dict)
        assert summary_data["summary"] == "Summary text"
        assert summary_data["factual_consistency_score"] == 0.95

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_generation_with_defaults(self, mock_vectara: MagicMock) -> None:
        """Test VectaraGeneration tool with default configuration."""

        mock_vectorstore = mock_vectara.return_value
        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {
            "answer": "Summary text",
            "fcs": 0.95,
            "context": [
                (
                    MagicMock(
                        page_content="Document content", metadata={"source": "test"}
                    ),
                    0.9,
                )
            ],
        }
        mock_vectorstore.as_rag.return_value = mock_rag

        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        result = tool._run(
            query="test query",
        )

        mock_vectorstore.as_rag.assert_called_once()

        # Get the VectaraQueryConfig passed to as_rag
        config = mock_vectorstore.as_rag.call_args[0][0]
        assert isinstance(config, VectaraQueryConfig)
        assert config.search.corpora is not None
        assert len(config.search.corpora) == 1
        assert config.search.corpora[0].corpus_key == "test-corpus-123"
        assert config.generation is not None
        assert (
            config.generation.generation_preset_name
            == "vectara-summary-ext-24-05-med-omni"
        )

        # Verify result is valid JSON with summary
        summary_data = json.loads(result)
        assert isinstance(summary_data, dict)
        assert summary_data["summary"] == "Summary text"
        assert summary_data["factual_consistency_score"] == 0.95

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_generation_without_answer(self, mock_vectara: MagicMock) -> None:
        """Test VectaraGeneration tool with no answer in response."""

        mock_vectorstore = mock_vectara.return_value
        mock_rag = MagicMock()
        # Response with no answer key
        mock_rag.invoke.return_value = {
            "fcs": None,
            "context": [
                (
                    MagicMock(
                        page_content="Document content", metadata={"source": "test"}
                    ),
                    0.9,
                )
            ],
        }
        mock_vectorstore.as_rag.return_value = mock_rag

        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        result = tool._run(
            query="test query",
        )

        # Verify result is valid JSON with summary=None
        summary_data = json.loads(result)
        assert isinstance(summary_data, dict)
        assert summary_data["summary"] is None
        assert summary_data["factual_consistency_score"] == "N/A"

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_generation_with_no_results(self, mock_vectara: MagicMock) -> None:
        """Test VectaraGeneration tool with empty response."""

        mock_vectorstore = mock_vectara.return_value
        mock_rag = MagicMock()

        mock_rag.invoke.return_value = None
        mock_vectorstore.as_rag.return_value = mock_rag

        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        result = tool._run(
            query="test query",
        )

        # Should return a message about no results
        assert result == "No results found"

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_ingest(self, mock_vectara: MagicMock) -> None:
        """Test VectaraIngest tool with basic parameters."""
        mock_vectorstore = mock_vectara.return_value
        mock_vectorstore.add_texts.return_value = ["doc1", "doc2"]

        tool = VectaraIngest(
            name="test_ingest",
            description="Test ingest",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        documents = ["Document 1", "Document 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]

        result = tool._run(
            documents=documents,
            metadatas=metadatas,
        )

        mock_vectorstore.add_texts.assert_called_once_with(
            texts=documents,
            metadatas=metadatas,
            ids=None,
            corpus_key="test-corpus-123",
        )

        # Verify result
        assert "Successfully ingested 2 documents" in result
        assert "test-corpus-123" in result
        assert "doc1, doc2" in result

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_ingest_with_all_parameters(self, mock_vectara: MagicMock) -> None:
        """Test VectaraIngest tool with all available parameters."""

        mock_vectorstore = mock_vectara.return_value
        mock_vectorstore.add_texts.return_value = ["custom1", "custom2"]

        # Create tool with required corpus_key
        tool = VectaraIngest(
            name="test_ingest",
            description="Test ingest",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        # Test ingest with all parameters
        documents = ["Document 1", "Document 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        ids = ["custom1", "custom2"]
        doc_metadata = {"batch": "test-batch", "department": "engineering"}
        doc_type = "structured"

        result = tool._run(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            doc_metadata=doc_metadata,
            doc_type=doc_type,
        )

        # Verify calls with all parameters
        mock_vectorstore.add_texts.assert_called_once_with(
            texts=documents,
            metadatas=metadatas,
            ids=ids,
            corpus_key="test-corpus-123",
            doc_metadata=doc_metadata,
            doc_type=doc_type,
        )

        # Verify result
        assert "Successfully ingested 2 documents" in result
        assert "test-corpus-123" in result
        assert "custom1, custom2" in result

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_ingest_with_override_corpus(self, mock_vectara: MagicMock) -> None:
        """Test VectaraIngest tool with corpus_key override."""
        # Set up mock
        mock_vectorstore = mock_vectara.return_value
        mock_vectorstore.add_texts.return_value = ["doc1", "doc2"]

        # Create tool with default corpus_key
        tool = VectaraIngest(
            name="test_ingest",
            description="Test ingest",
            vectorstore=mock_vectorstore,
            corpus_key="default-corpus",
        )

        # Test ingest with a different corpus_key
        documents = ["Document 1", "Document 2"]

        result = tool._run(documents=documents, corpus_key="override-corpus")

        # Verify calls uses the override corpus_key
        mock_vectorstore.add_texts.assert_called_once_with(
            texts=documents, metadatas=None, ids=None, corpus_key="override-corpus"
        )

        # Verify result uses the override corpus
        assert "Successfully ingested 2 documents" in result
        assert "override-corpus" in result

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_search_without_corpus_key(self, mock_vectara: MagicMock) -> None:
        """Test VectaraSearch tool with no corpus_key provided."""
        # Set up mock
        mock_vectorstore = mock_vectara.return_value

        # Create tool without corpus_key
        tool = VectaraSearch(
            name="test_search",
            description="Test search",
            vectorstore=mock_vectorstore,
            corpus_key=None,  # Explicitly set to None
        )

        # Test without config
        result = tool._run(
            query="test query",
        )

        # Should return error about missing corpus_key
        assert "Error: A corpus_key is required for search" in result
        assert (
            "provide it either directly to the tool or in the config object" in result
        )

        # Test with empty config
        empty_config = VectaraQueryConfig(search=SearchConfig())

        result = tool._run(query="test query", config=empty_config)

        # Should still return error about missing corpus_key
        assert "Error: A corpus_key is required for search" in result

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_generation_without_corpus_key(
        self, mock_vectara: MagicMock
    ) -> None:
        """Test VectaraGeneration tool with no corpus_key provided."""
        # Set up mock
        mock_vectorstore = mock_vectara.return_value

        # Create tool without corpus_key
        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key=None,  # Explicitly set to None
        )

        # Test without config
        result = tool._run(
            query="test query",
        )

        # Should return error about missing corpus_key
        assert "Error: A corpus_key is required for generation" in result
        assert (
            "provide it either directly to the tool or in the config object" in result
        )

        # Test with empty config
        empty_config = VectaraQueryConfig(
            search=SearchConfig(), generation=GenerationConfig()
        )

        result = tool._run(query="test query", config=empty_config)

        # Should still return error about missing corpus_key
        assert "Error: A corpus_key is required for generation" in result

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_generation_without_context(self, mock_vectara: MagicMock) -> None:
        """Test VectaraGeneration tool with response missing context field."""
        # Set up mock
        mock_vectorstore = mock_vectara.return_value
        mock_rag = MagicMock()
        # Response with answer but no context field
        mock_rag.invoke.return_value = {
            "answer": "Summary text without context",
            "fcs": 0.80,
            # No context field
        }
        mock_vectorstore.as_rag.return_value = mock_rag

        # Create tool
        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        # Test with a response that has no context
        result = tool._run(
            query="test query",
        )

        # Verify result is valid JSON with just the summary
        summary_data = json.loads(result)
        assert isinstance(summary_data, dict)
        assert summary_data["summary"] == "Summary text without context"
        assert summary_data["factual_consistency_score"] == 0.80

    @patch("langchain_community.tools.vectara.tool.Vectara")
    def test_vectara_generation_with_empty_dict(self, mock_vectara: MagicMock) -> None:
        """Test VectaraGeneration tool with empty dictionary response."""
        # Set up mock
        mock_vectorstore = mock_vectara.return_value
        mock_rag = MagicMock()
        # Empty dictionary response
        mock_rag.invoke.return_value = {}
        mock_vectorstore.as_rag.return_value = mock_rag

        # Create tool
        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        # Test with empty dictionary response
        result = tool._run(
            query="test query",
        )

        # Verify result is valid JSON with summary=None
        summary_data = json.loads(result)
        assert isinstance(summary_data, dict)
        assert summary_data["summary"] is None
        assert summary_data["factual_consistency_score"] == "N/A"


if __name__ == "__main__":
    unittest.main()
