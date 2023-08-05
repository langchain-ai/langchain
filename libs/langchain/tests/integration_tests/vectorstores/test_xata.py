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

        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]