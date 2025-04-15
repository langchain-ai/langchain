import unittest
from langchain_opensearch.vectorstores import OpenSearchVectorStore

class TestOpenSearchVectorStore(unittest.TestCase):
    def test_import(self):
        """Test that the OpenSearchVectorStore class can be imported."""
        self.assertTrue(hasattr(OpenSearchVectorStore, "add_texts"))