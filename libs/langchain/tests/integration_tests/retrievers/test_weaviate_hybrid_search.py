"""Test Weaviate functionality."""
import logging
import os
import uuid
from typing import Generator, Union
from uuid import uuid4

import pytest

from langchain.docstore.document import Document
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever

logging.basicConfig(level=logging.DEBUG)

"""
cd tests/integration_tests/vectorstores/docker-compose
docker compose -f weaviate.yml up
"""


class TestWeaviateHybridSearchRetriever:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

    @pytest.fixture(scope="class", autouse=True)
    def weaviate_url(self) -> Union[str, Generator[str, None, None]]:
        """Return the weaviate url."""
        from weaviate import Client

        url = "http://localhost:8080"
        yield url

        # Clear the test index
        client = Client(url)
        client.schema.delete_all()

    @pytest.mark.vcr(ignore_localhost=True)
    def test_get_relevant_documents(self, weaviate_url: str) -> None:
        """Test end to end construction and MRR search."""
        from weaviate import Client

        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]

        client = Client(weaviate_url)

        retriever = WeaviateHybridSearchRetriever(
            client=client,
            index_name=f"LangChain_{uuid4().hex}",
            text_key="text",
            attributes=["page"],
        )
        for i, text in enumerate(texts):
            retriever.add_documents(
                [Document(page_content=text, metadata=metadatas[i])]
            )

        output = retriever.get_relevant_documents("foo")
        assert output == [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="baz", metadata={"page": 2}),
            Document(page_content="bar", metadata={"page": 1}),
        ]

    @pytest.mark.vcr(ignore_localhost=True)
    def test_get_relevant_documents_with_score(self, weaviate_url: str) -> None:
        """Test end to end construction and MRR search."""
        from weaviate import Client

        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]

        client = Client(weaviate_url)

        retriever = WeaviateHybridSearchRetriever(
            client=client,
            index_name=f"LangChain_{uuid4().hex}",
            text_key="text",
            attributes=["page"],
        )
        for i, text in enumerate(texts):
            retriever.add_documents(
                [Document(page_content=text, metadata=metadatas[i])]
            )

        output = retriever.get_relevant_documents("foo", score=True)
        for doc in output:
            assert "_additional" in doc.metadata

    @pytest.mark.vcr(ignore_localhost=True)
    def test_get_relevant_documents_with_filter(self, weaviate_url: str) -> None:
        """Test end to end construction and MRR search."""
        from weaviate import Client

        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]

        client = Client(weaviate_url)

        retriever = WeaviateHybridSearchRetriever(
            client=client,
            index_name=f"LangChain_{uuid4().hex}",
            text_key="text",
            attributes=["page"],
        )
        for i, text in enumerate(texts):
            retriever.add_documents(
                [Document(page_content=text, metadata=metadatas[i])]
            )

        where_filter = {"path": ["page"], "operator": "Equal", "valueNumber": 0}

        output = retriever.get_relevant_documents("foo", where_filter=where_filter)
        assert output == [
            Document(page_content="foo", metadata={"page": 0}),
        ]

    @pytest.mark.vcr(ignore_localhost=True)
    def test_get_relevant_documents_with_uuids(self, weaviate_url: str) -> None:
        """Test end to end construction and MRR search."""
        from weaviate import Client

        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        # Weaviate replaces the object if the UUID already exists
        uuids = [uuid.uuid5(uuid.NAMESPACE_DNS, "same-name") for text in texts]

        client = Client(weaviate_url)

        retriever = WeaviateHybridSearchRetriever(
            client=client,
            index_name=f"LangChain_{uuid4().hex}",
            text_key="text",
            attributes=["page"],
        )
        for i, text in enumerate(texts):
            retriever.add_documents(
                [Document(page_content=text, metadata=metadatas[i])], uuids=[uuids[i]]
            )

        output = retriever.get_relevant_documents("foo")
        assert len(output) == 1
