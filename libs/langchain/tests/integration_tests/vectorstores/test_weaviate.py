"""Test Weaviate V4 functionality under libs/langchain/

# Launch the Weaviate pod
cd tests/integration_tests/vectorstores/docker-compose
docker compose -f weaviate.yml up

# Run the integration tests inside libs/langchain/
pytest -sv tests/integration_tests/vectorstores/test_weaviate.py

"""

import logging
import os
import pytest
from typing import Generator, Union

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate
from weaviate.classes.config import DataType, Property
from weaviate.classes.query import Filter

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(scope="module")
def embedding_openai() -> OpenAIEmbeddings:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAIEmbeddings()


class TestWeaviate:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

    @pytest.fixture(scope="class", autouse=True)
    def weaviate_url(self) -> Union[str, Generator[str, None, None]]:  # type: ignore[return]
        """Return the weaviate url."""
        from weaviate import Client

        url = "http://localhost:8080"
        yield url

        # Clear the test index
        client = Client(url)
        client.schema.delete_all()

    @pytest.fixture(scope="class", autouse=True)
    def weaviate_host(self) -> str:  # type: ignore[return]
        # localhost or weaviate
        host = "localhost"
        yield host

    @pytest.fixture(scope="class", autouse=True)
    def weaviate_port(self) -> str:  # type: ignore[return]
        port = "8080"
        yield port

    @pytest.fixture(scope="class", autouse=True)
    def weaviate_client(
        self, weaviate_host: str, weaviate_port: str
    ) -> weaviate.WeaviateClient:
        weaviate_client = weaviate.connect_to_local(
            host=weaviate_host, port=weaviate_port
        )
        # Create the collection if not exists
        # The collection name is assumed to be "test" throughout the test module
        if not weaviate_client.collections.exists("test"):
            weaviate_client.collections.create(
                name="test", properties=[Property(name="name", data_type=DataType.TEXT)]
            )
        yield weaviate_client

    @pytest.mark.vcr(ignore_localhost=True)
    def test_similarity_search_with_metadata(
        self, weaviate_client: str, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = WeaviateVectorStore.from_texts(
            texts, embedding_openai, metadatas=metadatas, client=weaviate_client
        )
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={"page": 0})]

    @pytest.mark.vcr(ignore_localhost=True)
    def test_similarity_search_with_metadata_and_filter(
        self,
        weaviate_client: weaviate.WeaviateClient,
        embedding_openai: OpenAIEmbeddings,
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        search_filter = Filter.by_property("page").equal(0)

        docsearch = WeaviateVectorStore.from_texts(
            texts, embedding_openai, metadatas=metadatas, client=weaviate_client
        )
        # Even if k=2, only one result should be returned because of the filter
        output = docsearch.similarity_search("foo", k=2, filters=search_filter)
        assert output == [Document(page_content="foo", metadata={"page": 0})]

    @pytest.mark.vcr(ignore_localhost=True)
    def test_max_marginal_relevance_search(
        self,
        weaviate_client: weaviate.WeaviateClient,
        embedding_openai: OpenAIEmbeddings,
    ) -> None:
        """Test end to end construction and MRR search."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]

        docsearch = WeaviateVectorStore.from_texts(
            texts, embedding_openai, metadatas=metadatas, client=weaviate_client
        )
        # if lambda=1 the algorithm should be equivalent to standard ranking
        standard_ranking = docsearch.similarity_search("foo", k=2)
        output = docsearch.max_marginal_relevance_search(
            "foo", k=2, fetch_k=3, lambda_mult=1.0
        )
        assert output == standard_ranking

        # if lambda=0 the algorithm should favour maximal diversity
        output = docsearch.max_marginal_relevance_search(
            "foo", k=2, fetch_k=3, lambda_mult=0.0
        )
        assert output == [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="bar", metadata={"page": 1}),
        ]

    @pytest.mark.vcr(ignore_localhost=True)
    def test_max_marginal_relevance_search_by_vector(
        self,
        weaviate_client: weaviate.WeaviateClient,
        embedding_openai: OpenAIEmbeddings,
    ) -> None:
        """Test end to end construction and MRR search by vector."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]

        docsearch = WeaviateVectorStore.from_texts(
            texts, embedding_openai, metadatas=metadatas, client=weaviate_client
        )
        foo_embedding = embedding_openai.embed_query("foo")

        # if lambda=1 the algorithm should be equivalent to standard ranking
        standard_ranking = docsearch.similarity_search("foo", k=2)
        output = docsearch.max_marginal_relevance_search_by_vector(
            foo_embedding, k=2, fetch_k=3, lambda_mult=1.0
        )
        assert output == standard_ranking

        # if lambda=0 the algorithm should favour maximal diversity
        output = docsearch.max_marginal_relevance_search_by_vector(
            foo_embedding, k=2, fetch_k=3, lambda_mult=0.0
        )
        assert output == [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="bar", metadata={"page": 1}),
        ]

    def test_add_texts_with_given_embedding(
        self,
        weaviate_client: weaviate.WeaviateClient,
        embedding_openai: OpenAIEmbeddings,
    ) -> None:
        texts = ["foo", "bar", "baz"]
        embedding = embedding_openai

        docsearch = WeaviateVectorStore.from_texts(
            texts, embedding=embedding, client=weaviate_client
        )

        # similarity_search_by_vector is not implemented
        with pytest.raises(NotImplementedError):
            docsearch.add_texts(["foo"])
            output = docsearch.similarity_search_by_vector(
                embedding.embed_query("foo"), k=2
            )
            assert output == [
                Document(page_content="foo"),
                Document(page_content="foo"),
            ]
