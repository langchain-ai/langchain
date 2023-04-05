"""Test ElasticSearch functionality."""
import logging
import os
from typing import Generator, List, Union

import pytest
from elasticsearch import Elasticsearch

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

logging.basicConfig(level=logging.DEBUG)

"""
cd tests/integration_tests/vectorstores/docker-compose
docker-compose -f elasticsearch.yml up
"""


class TestElasticsearch:
    @pytest.fixture(scope="class", autouse=True)
    def elasticsearch_url(self) -> Union[str, Generator[str, None, None]]:
        """Return the elasticsearch url."""
        url = "http://localhost:9200"
        yield url
        es = Elasticsearch(hosts=url)

        # Clear all indexes
        index_names = es.indices.get(index="_all").keys()
        for index_name in index_names:
            # print(index_name)
            es.indices.delete(index=index_name)

    @pytest.fixture(scope="class", autouse=True)
    def openai_api_key(self) -> Union[str, Generator[str, None, None]]:
        """Return the OpenAI API key."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        yield openai_api_key

    @pytest.fixture(scope="class")
    def documents(self) -> Generator[List[Document], None, None]:
        """Return a generator that yields a list of documents."""
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        documents = TextLoader(
            os.path.join(os.path.dirname(__file__), "fixtures", "sharks.txt")
        ).load()
        yield text_splitter.split_documents(documents)

    def test_similarity_search_without_metadata(self, elasticsearch_url: str) -> None:
        """Test end to end construction and search without metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticVectorSearch.from_texts(
            texts, FakeEmbeddings(), elasticsearch_url=elasticsearch_url
        )
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_similarity_search_with_metadata(self, elasticsearch_url: str) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticVectorSearch.from_texts(
            texts,
            FakeEmbeddings(),
            metadatas=metadatas,
            elasticsearch_url=elasticsearch_url,
        )
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={"page": 0})]

    @pytest.mark.vcr()
    def test_default_index_from_documents(
        self, documents: List[Document], openai_api_key: str, elasticsearch_url: str
    ) -> None:
        """This test checks the construction of a default
        ElasticSearch index using the 'from_documents'."""
        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

        elastic_vector_search = ElasticVectorSearch.from_documents(
            documents=documents,
            embedding=embedding,
            elasticsearch_url=elasticsearch_url,
        )

        search_result = elastic_vector_search.similarity_search("sharks")

        print(search_result)
        assert len(search_result) != 0

    def test_custom_index_from_documents(
        self, documents: List[Document], openai_api_key: str, elasticsearch_url: str
    ) -> None:
        """This test checks the construction of a custom
        ElasticSearch index using the 'from_documents'."""
        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        elastic_vector_search = ElasticVectorSearch.from_documents(
            documents=documents,
            embedding=embedding,
            elasticsearch_url=elasticsearch_url,
            index_name="custom_index",
        )
        es = Elasticsearch(hosts=elasticsearch_url)
        index_names = es.indices.get(index="_all").keys()
        assert "custom_index" in index_names

        search_result = elastic_vector_search.similarity_search("sharks")
        print(search_result)

        assert len(search_result) != 0

    def test_custom_index_add_documents(
        self, documents: List[Document], openai_api_key: str, elasticsearch_url: str
    ) -> None:
        """This test checks the construction of a custom
        ElasticSearch index using the 'add_documents'."""
        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        elastic_vector_search = ElasticVectorSearch(
            embedding=embedding,
            elasticsearch_url=elasticsearch_url,
            index_name="custom_index",
        )
        es = Elasticsearch(hosts=elasticsearch_url)
        index_names = es.indices.get(index="_all").keys()
        assert "custom_index" in index_names

        elastic_vector_search.add_documents(documents)
        search_result = elastic_vector_search.similarity_search("sharks")
        print(search_result)

        assert len(search_result) != 0
