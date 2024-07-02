"""Run standard read write tests on the PineconeVectorStore."""
import os
import time

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_standard_tests.integration_tests.vectorstores import ReadWriteTestSuite

from langchain_pinecone.vectorstores import PineconeVectorStore

INDEX_NAME = "langchain-standard-tests-index"
NAMESPACE_NAME = "langchain-standard-tests-namespace"
DIMENSION = 6

DEFAULT_SLEEP = 20


class TestRWAPI(ReadWriteTestSuite):
    @pytest.fixture()
    def vectorstore(self) -> VectorStore:
        """Get an empty vectorstore."""
        import pinecone
        from pinecone import PodSpec

        client = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_list = client.list_indexes()
        for i in index_list:
            if i["name"] == INDEX_NAME:
                client.delete_index(INDEX_NAME)
                break
        if len(index_list) > 0:
            time.sleep(DEFAULT_SLEEP)  # prevent race with creation

        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=PodSpec(environment="gcp-starter"),
        )

        index = client.Index(INDEX_NAME)

        # insure the index is empty
        index_stats = index.describe_index_stats()
        assert index_stats["dimension"] == DIMENSION

        if index_stats["namespaces"].get(NAMESPACE_NAME) is not None:
            assert index_stats["namespaces"][NAMESPACE_NAME]["vector_count"] == 0
        try:
            yield PineconeVectorStore(
                index_name=INDEX_NAME, embedding=self.get_embeddings()
            )
        finally:
            client.delete_index(INDEX_NAME)
