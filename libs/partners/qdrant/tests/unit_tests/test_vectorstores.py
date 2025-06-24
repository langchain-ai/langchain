import unittest
import uuid
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FakeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter


class TestQdrantDeleteIntegration(unittest.TestCase):
    def setUp(self):
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

    def test_delete_by_ids(self):
        result = self.vector_store.delete(ids=[self.uuids[0]])
        self.assertTrue(result)

        res = self.client.scroll(self.collection_name)
        remaining_ids = [str(point.id) for point in res[0]]
        self.assertNotIn(self.uuids[0], remaining_ids)

    def test_delete_by_metadata(self):
        result = self.vector_store.delete(owner="admin")
        self.assertTrue(result)

        res = self.client.scroll(self.collection_name)
        remaining_ids = [str(point.id) for point in res[0]]
        self.assertNotIn(self.uuids[2], remaining_ids)

    def test_delete_failure_invalid_filter(self):
        
        # Get initial count
        initial_res = self.client.scroll(self.collection_name)
        initial_count = len(initial_res[0])
        
        result = self.vector_store.delete(owner="nonexistent")
        self.assertTrue(result)  

        # Verify no documents were actually deleted
        res = self.client.scroll(self.collection_name)
        remaining_ids = [str(point.id) for point in res[0]]
        self.assertEqual(len(remaining_ids), initial_count) 
        self.assertCountEqual(remaining_ids, self.uuids)  


if __name__ == "__main__":
    unittest.main()




