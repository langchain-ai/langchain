"""Test Xata vector store functionality."""
import importlib
import os

import pytest
from xata.client import XataClient

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.xata import XataVectorStore

class TestXata:
    @classmethod
    def setup_class(cls) -> None:
        assert os.getenv("XATA_API_KEY"), "XATA_API_KEY environment variable is not set"
        assert os.getenv("XATA_DB_URL"), "XATA_DB_URL environment variable is not set"

    @pytest.fixture(scope="class", autouse=True)
    def xata(self) -> XataClient:
        """Return the xata client."""
        client = XataClient(
            api_key=os.getenv("XATA_API_KEY"),
            db_url=os.getenv("XATA_DB_URL"),
            branch_name=os.getenv("XATA_BRANCH_NAME", "main"))
        yield client

    def test_similarity_search_without_metadata(
        self, xata: XataClient, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end constructions and search without metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = XataVectorStore.from_texts(
            texts=texts, 
            embedding=embedding_openai,
            client=xata)
        docsearch.wait_for_indexing(ndocs=3)

        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]
        docsearch.delete(delete_all=True)

    def test_similarity_search_with_metadata(
        self, xata: XataClient, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search with a metadata filter.

        This test requires a column named "a" of type integer to be present
        in the Xata table. """
        texts = ["foo", "foo", "foo"]
        metadatas = [{"a": i} for i in range(len(texts))]
        docsearch = XataVectorStore.from_texts(
            texts=texts, 
            embedding=embedding_openai,
            metadatas=metadatas,
            client=xata)
        docsearch.wait_for_indexing(ndocs=3)
        output = docsearch.similarity_search("foo", k=1, filter={"a": 1})
        assert output == [Document(page_content="foo", metadata={"a": 1})]
        docsearch.delete(delete_all=True)

