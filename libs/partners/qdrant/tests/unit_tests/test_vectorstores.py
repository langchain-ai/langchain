import unittest
import uuid

from langchain_core.embeddings import FakeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_qdrant import QdrantVectorStore


class TestQdrantDeleteIntegration(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test fixtures with in-memory Qdrant client."""
        self.client = QdrantClient(":memory:")
        self.collection_name = "demo_collection"

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=10, distance=Distance.COSINE),
        )

        self.embeddings = FakeEmbeddings(size=10)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        self.uuids = [str(uuid.uuid4()) for _ in range(3)]
        self.vector_store.add_texts(
            texts=["first", "second", "third"],
            ids=self.uuids,
            metadatas=[{"type": "image"}, {"type": "text"}, {"owner": "admin"}],
        )

    def test_delete_by_ids(self) -> None:
        """Test deletion by document IDs."""
        result = self.vector_store.delete(ids=[self.uuids[0]])
        self.assertTrue(result)

        res = self.client.scroll(self.collection_name)
        remaining_ids = [str(point.id) for point in res[0]]
        self.assertNotIn(self.uuids[0], remaining_ids)

    def test_delete_by_filters(self) -> None:
        """Test deletion by metadata filters."""
        result = self.vector_store.delete(filters={"owner": "admin"})
        self.assertTrue(result)

        res = self.client.scroll(self.collection_name)
        remaining_ids = [str(point.id) for point in res[0]]
        self.assertNotIn(self.uuids[2], remaining_ids)

    def test_delete_by_multiple_filters(self) -> None:
        """Test deletion by multiple metadata filters."""
        # Add a document with multiple metadata fields
        test_id = str(uuid.uuid4())
        self.vector_store.add_texts(
            texts=["test document"],
            ids=[test_id],
            metadatas=[{"owner": "admin", "type": "document"}],
        )

        result = self.vector_store.delete(
            filters={"owner": "admin", "type": "document"}
        )
        self.assertTrue(result)

        res = self.client.scroll(self.collection_name)
        remaining_ids = [str(point.id) for point in res[0]]
        self.assertNotIn(test_id, remaining_ids)

    def test_delete_nonexistent_filters(self) -> None:
        """Test deletion with filters that match no documents."""
        # Get initial count
        initial_res = self.client.scroll(self.collection_name)
        initial_count = len(initial_res[0])

        result = self.vector_store.delete(filters={"owner": "nonexistent"})
        self.assertTrue(result)

        # Verify no documents were actually deleted
        res = self.client.scroll(self.collection_name)
        remaining_ids = [str(point.id) for point in res[0]]
        self.assertEqual(len(remaining_ids), initial_count)
        self.assertCountEqual(remaining_ids, self.uuids)

    def test_delete_no_parameters(self) -> None:
        """Test that delete raises error when no parameters are provided."""
        with self.assertRaises(ValueError):
            self.vector_store.delete()

    def test_delete_both_ids_and_filters(self) -> None:
        """Test that when both ids and filters are provided, ids take precedence."""
        result = self.vector_store.delete(
            ids=[self.uuids[0]], filters={"owner": "admin"}
        )
        self.assertTrue(result)

        res = self.client.scroll(self.collection_name)
        remaining_ids = [str(point.id) for point in res[0]]
        # Only the ID should be deleted, not the filter match
        self.assertNotIn(self.uuids[0], remaining_ids)
        self.assertIn(self.uuids[2], remaining_ids)  # admin document should still exist
