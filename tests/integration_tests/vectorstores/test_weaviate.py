"""Test Weaviate functionality."""
import logging
from typing import Generator, Union

import pytest
from weaviate import Client

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.weaviate import Weaviate

logging.basicConfig(level=logging.DEBUG)

"""
cd tests/integration_tests/vectorstores/docker-compose
docker compose -f weaviate.yml up
"""


class TestWeaviate:
    @pytest.fixture(scope="class", autouse=True)
    def weaviate_url(self) -> Union[str, Generator[str, None, None]]:
        """Return the weaviate url."""
        url = "http://localhost:8080"
        yield url

        # Clear the test index
        client = Client(url)
        client.schema.delete_all()

    def test_similarity_search_without_metadata(self, weaviate_url: str) -> None:
        """Test end to end construction and search without metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = Weaviate.from_texts(
            texts,
            OpenAIEmbeddings(),
            weaviate_url=weaviate_url,
        )

        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_similarity_search_with_metadata(self, weaviate_url: str) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = Weaviate.from_texts(
            texts, OpenAIEmbeddings(), metadatas=metadatas, weaviate_url=weaviate_url
        )
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={"page": 0})]
