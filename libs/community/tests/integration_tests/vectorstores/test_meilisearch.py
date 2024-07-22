"""Test Meilisearch functionality."""

from typing import TYPE_CHECKING, Any, Dict, Generator

import pytest
import requests
from langchain_core.documents import Document

from langchain_community.vectorstores import Meilisearch
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

if TYPE_CHECKING:
    import meilisearch

INDEX_NAME = "test-langchain-demo"
TEST_MEILI_HTTP_ADDR = "http://localhost:7700"
TEST_MEILI_MASTER_KEY = "masterKey"


class TestMeilisearchVectorSearch:
    @pytest.fixture(scope="class", autouse=True)
    def enable_vector_search(self) -> Generator[str, None, None]:
        requests.patch(
            f"{TEST_MEILI_HTTP_ADDR}/experimental-features",
            headers={"Authorization": f"Bearer {TEST_MEILI_MASTER_KEY}"},
            json={"vectorStore": True},
            timeout=10,
        )
        yield "done"
        requests.patch(
            f"{TEST_MEILI_HTTP_ADDR}/experimental-features",
            headers={"Authorization": f"Bearer {TEST_MEILI_MASTER_KEY}"},
            json={"vectorStore": False},
            timeout=10,
        )

    @pytest.fixture
    def new_embedders(self) -> Dict[str, Dict[str, Any]]:
        return {
            "default": {
                "source": "userProvided",
                # Dimension defined in FakeEmbeddings as [float(1.0)] * 9 + [float(0.0)]
                "dimensions": 10,
            }
        }

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.delete_all_indexes()

    @pytest.fixture(scope="class", autouse=True)
    def teardown_test(self) -> Generator[str, None, None]:
        # Yields back to the test function.
        yield "done"
        self.delete_all_indexes()

    def delete_all_indexes(self) -> None:
        client = self.client()
        # Deletes all the indexes in the Meilisearch instance.
        indexes = client.get_indexes()
        for index in indexes["results"]:
            task = client.index(index.uid).delete()
            client.wait_for_task(task.task_uid)

    def client(self) -> "meilisearch.Client":
        import meilisearch

        return meilisearch.Client(TEST_MEILI_HTTP_ADDR, TEST_MEILI_MASTER_KEY)

    def _wait_last_task(self) -> None:
        client = self.client()
        # Get the last task
        tasks = client.get_tasks()
        # Wait for the last task to be completed
        client.wait_for_task(tasks.results[0].uid)

    def test_meilisearch(self, new_embedders: Dict[str, Any]) -> None:
        """Test end to end construction and search."""
        texts = ["foo", "bar", "baz"]
        vectorstore = Meilisearch.from_texts(
            texts=texts,
            embedding=FakeEmbeddings(),
            embedders=new_embedders,
            embedder_name=list(new_embedders)[0],
            url=TEST_MEILI_HTTP_ADDR,
            api_key=TEST_MEILI_MASTER_KEY,
            index_name=INDEX_NAME,
        )
        self._wait_last_task()
        output = vectorstore.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_meilisearch_with_client(self, new_embedders: Dict[str, Any]) -> None:
        """Test end to end construction and search."""
        texts = ["foo", "bar", "baz"]
        vectorstore = Meilisearch.from_texts(
            texts=texts,
            embedding=FakeEmbeddings(),
            embedders=new_embedders,
            embedder_name=list(new_embedders)[0],
            client=self.client(),
            index_name=INDEX_NAME,
        )
        self._wait_last_task()
        output = vectorstore.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_meilisearch_with_metadatas(self, new_embedders: Dict[str, Any]) -> None:
        """Test end to end construction and search."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = Meilisearch.from_texts(
            texts=texts,
            embedding=FakeEmbeddings(),
            embedders=new_embedders,
            embedder_name=list(new_embedders)[0],
            url=TEST_MEILI_HTTP_ADDR,
            api_key=TEST_MEILI_MASTER_KEY,
            index_name=INDEX_NAME,
            metadatas=metadatas,
        )
        self._wait_last_task()
        output = docsearch.similarity_search("foo", k=1)
        assert len(output) == 1
        assert output[0].page_content == "foo"
        assert output[0].metadata["page"] == 0
        assert output == [Document(page_content="foo", metadata={"page": 0})]

    def test_meilisearch_with_metadatas_with_scores(
        self, new_embedders: Dict[str, Any]
    ) -> None:
        """Test end to end construction and scored search."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        docsearch = Meilisearch.from_texts(
            texts=texts,
            embedding=FakeEmbeddings(),
            embedders=new_embedders,
            embedder_name=list(new_embedders)[0],
            url=TEST_MEILI_HTTP_ADDR,
            api_key=TEST_MEILI_MASTER_KEY,
            index_name=INDEX_NAME,
            metadatas=metadatas,
        )
        self._wait_last_task()
        output = docsearch.similarity_search_with_score("foo", k=1)
        assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]

    def test_meilisearch_with_metadatas_with_scores_using_vector(
        self, new_embedders: Dict[str, Any]
    ) -> None:
        """Test end to end construction and scored search, using embedding vector."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        embeddings = FakeEmbeddings()

        docsearch = Meilisearch.from_texts(
            texts=texts,
            embedding=FakeEmbeddings(),
            embedders=new_embedders,
            embedder_name=list(new_embedders)[0],
            url=TEST_MEILI_HTTP_ADDR,
            api_key=TEST_MEILI_MASTER_KEY,
            index_name=INDEX_NAME,
            metadatas=metadatas,
        )
        embedded_query = embeddings.embed_query("foo")
        self._wait_last_task()
        output = docsearch.similarity_search_by_vector_with_scores(
            embedding=embedded_query, k=1
        )
        assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]
