"""Test Weaviate functionality."""
import logging
import os
from typing import Generator, List, Union

import pytest
from weaviate import Client

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.weaviate import Weaviate
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

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

    @pytest.fixture(scope="class")
    def documents(self) -> Generator[List[Document], None, None]:
        """Return a generator that yields a list of documents."""
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        documents = TextLoader(
            os.path.join(os.path.dirname(__file__), "fixtures", "sharks.txt")
        ).load()
        yield text_splitter.split_documents(documents)

    def test_similarity_search_without_metadata(self, weaviate_url: str) -> None:
        """Test end to end construction and search without metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = Weaviate.from_texts(
            texts,
            FakeEmbeddings(),
            weaviate_url=weaviate_url,
        )

        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_similarity_search_with_metadata(self, weaviate_url: str) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = Weaviate.from_texts(
            texts, FakeEmbeddings(), metadatas=metadatas, weaviate_url=weaviate_url
        )
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={"page": 0})]
