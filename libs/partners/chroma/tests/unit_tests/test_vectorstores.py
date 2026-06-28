import sys
from unittest.mock import MagicMock, patch

import pytest

# Build a complete mock chromadb hierarchy before any real import
mock_chromadb = MagicMock()
mock_chromadb.config = MagicMock()
mock_chromadb.api = MagicMock()
mock_chromadb.api.CreateCollectionConfiguration = MagicMock()
mock_chromadb.Settings = MagicMock()
mock_chromadb.Search = MagicMock()
mock_chromadb.Collection = MagicMock()

sys.modules["chromadb"] = mock_chromadb
sys.modules["chromadb.config"] = mock_chromadb.config
sys.modules["chromadb.api"] = mock_chromadb.api

import uuid  # noqa: E402

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings.fake import FakeEmbeddings  # noqa: E402

from langchain_chroma.vectorstores import Chroma  # noqa: E402

mock_collection = MagicMock()
mock_collection.name = "test_collection"
mock_collection.configuration = {"hnsw": {"space": "ip"}}


@pytest.fixture
def chroma_instance():
    instance = Chroma(
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(size=10),
    )
    instance._chroma_collection = mock_collection
    return instance


class TestChromaRelevanceScores:
    def test_chroma_max_inner_product_relevance_score_fn(self):
        fn = Chroma._chroma_max_inner_product_relevance_score_fn
        assert fn(0.0) == 1.0
        assert fn(0.5) == 0.5
        assert fn(1.0) == 0.0
        assert fn(2.0) == -1.0

    def test_select_relevance_score_fn_ip_distance(self, chroma_instance):
        fn = chroma_instance._select_relevance_score_fn()
        assert fn is Chroma._chroma_max_inner_product_relevance_score_fn
        assert fn(0.0) == 1.0
        assert fn(1.0) == 0.0

    def test_select_relevance_score_fn_with_override(self, chroma_instance):
        custom_fn = MagicMock(return_value=0.5)
        chroma_instance.override_relevance_score_fn = custom_fn
        fn = chroma_instance._select_relevance_score_fn()
        assert fn == custom_fn
        assert fn(123.0) == 0.5

    @patch.object(Chroma, "_select_relevance_score_fn", return_value=lambda d: 1.0 - d)
    @patch.object(Chroma, "_Chroma__query_collection")
    def test_similarity_search_by_vector_with_relevance_scores(
        self, mock_query, mock_select_fn, chroma_instance
    ):
        mock_query.return_value = (
            [["doc1", "doc2"]],
            [[0.0, 0.5]],
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
        )
        mock_partial = MagicMock()
        mock_partial.return_value = [
            (Document(page_content="doc1"), 0.0),
            (Document(page_content="doc2"), 0.5),
        ]

        with patch(
            "langchain_chroma.vectorstores._results_to_docs_and_scores",
            mock_partial,
        ):
            results = chroma_instance.similarity_search_by_vector_with_relevance_scores(
                embedding=[0.1, 0.2, 0.3], k=2
            )

        assert len(results) == 2
        assert results[0][1] == 1.0
        assert results[1][1] == 0.5
        assert mock_select_fn.called
