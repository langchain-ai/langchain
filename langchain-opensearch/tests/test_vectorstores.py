import unittest
from unittest.mock import MagicMock, patch
from langchain_opensearch.vectorstores import OpenSearchVectorStore
from langchain_core.embeddings import Embeddings

try:
    from opensearchpy.helpers import bulk as opensearch_bulk_helper
except ImportError:
    opensearch_bulk_helper = None


class TestOpenSearchVectorStore(unittest.TestCase):
    def test_import(self):
        """Test that the OpenSearchVectorStore class can be imported."""
        self.assertTrue(hasattr(OpenSearchVectorStore, "add_texts"))

    @patch("opensearchpy.helpers.bulk")
    def test_add_texts_mock(self, mock_bulk_helper):
        """Test add_texts, mocking the opensearchpy bulk helper."""

        mock_client = MagicMock()
        mock_embeddings = MagicMock(spec=Embeddings)
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        vector_store = OpenSearchVectorStore(
            client=mock_client,
            index_name="test_index",
            embedding_function=mock_embeddings
        )

        texts = ["Hello", "World"]
        vector_store.add_texts(texts)

        mock_embeddings.embed_documents.assert_called_once_with(texts)
        mock_bulk_helper.assert_called_once()

        args, kwargs = mock_bulk_helper.call_args
        self.assertIs(args[0], mock_client)
        self.assertEqual(len(args[1]), len(texts))
        self.assertEqual(args[1][0]['_op_type'], 'index')
        self.assertIn('_source', args[1][0])
        self.assertEqual(args[1][0]['_source']['text'], texts[0])


if __name__ == '__main__':
    unittest.main()