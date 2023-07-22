"""Test ElasticSearch functionality."""
import logging
import os
import uuid
from typing import Generator, List, Union

import pytest
from elasticsearch import Elasticsearch

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

logging.basicConfig(level=logging.DEBUG)

"""
cd tests/integration_tests/vectorstores/docker-compose
docker-compose -f elasticsearch.yml up
"""


class TestElasticsearch:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

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

    def test_similarity_search_without_metadata(self, elasticsearch_url: str) -> None:
        """Test end to end construction and search without metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticVectorSearch.from_texts(
            texts, FakeEmbeddings(), elasticsearch_url=elasticsearch_url
        )
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_similarity_search_with_ssl_verify(self, elasticsearch_url: str) -> None:
        """Test end to end construction and search with ssl verify."""
        ssl_verify = {
            "verify_certs": True,
            "basic_auth": ("ES_USER", "ES_PASSWORD"),
            "ca_certs": "ES_CA_CERTS_PATH",
        }
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticVectorSearch.from_texts(
            texts,
            FakeEmbeddings(),
            elasticsearch_url=elasticsearch_url,
            ssl_verify=ssl_verify,
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

    @pytest.mark.vcr(ignore_localhost=True)
    def test_default_index_from_documents(
        self,
        documents: List[Document],
        embedding_openai: OpenAIEmbeddings,
        elasticsearch_url: str,
    ) -> None:
        """This test checks the construction of a default
        ElasticSearch index using the 'from_documents'."""

        elastic_vector_search = ElasticVectorSearch.from_documents(
            documents=documents,
            embedding=embedding_openai,
            elasticsearch_url=elasticsearch_url,
        )

        search_result = elastic_vector_search.similarity_search("sharks")

        print(search_result)
        assert len(search_result) != 0

    @pytest.mark.vcr(ignore_localhost=True)
    def test_custom_index_from_documents(
        self,
        documents: List[Document],
        embedding_openai: OpenAIEmbeddings,
        elasticsearch_url: str,
    ) -> None:
        """This test checks the construction of a custom
        ElasticSearch index using the 'from_documents'."""

        index_name = f"custom_index_{uuid.uuid4().hex}"
        elastic_vector_search = ElasticVectorSearch.from_documents(
            documents=documents,
            embedding=embedding_openai,
            elasticsearch_url=elasticsearch_url,
            index_name=index_name,
        )
        es = Elasticsearch(hosts=elasticsearch_url)
        index_names = es.indices.get(index="_all").keys()
        assert index_name in index_names

        search_result = elastic_vector_search.similarity_search("sharks")
        print(search_result)

        assert len(search_result) != 0

    @pytest.mark.vcr(ignore_localhost=True)
    def test_custom_index_add_documents(
        self,
        documents: List[Document],
        embedding_openai: OpenAIEmbeddings,
        elasticsearch_url: str,
    ) -> None:
        """This test checks the construction of a custom
        ElasticSearch index using the 'add_documents'."""

        index_name = f"custom_index_{uuid.uuid4().hex}"
        elastic_vector_search = ElasticVectorSearch(
            embedding=embedding_openai,
            elasticsearch_url=elasticsearch_url,
            index_name=index_name,
        )
        es = Elasticsearch(hosts=elasticsearch_url)
        elastic_vector_search.add_documents(documents)

        index_names = es.indices.get(index="_all").keys()
        assert index_name in index_names

        search_result = elastic_vector_search.similarity_search("sharks")
        print(search_result)

        assert len(search_result) != 0

    def test_custom_index_add_documents_to_exists_store(self) -> None:
        # TODO: implement it
        pass
