"""Test HANA vectorstore functionality."""
from typing import List

import numpy as np
import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores.hanavector import HanaDB
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings
from hdbcli import dbapi


try:
    from hdbcli import dbapi
    hanadb_installed = True
except ImportError:
    hanadb_installed = False

embedding = FakeEmbeddings()
connection = dbapi.connect(address="<address>",
                     port=30015,
                     user="<user>",
                     password="<password>",
                     autocommit=False,
                     sslValidateCertificate=False
                     )
vectordb = HanaDB(connection=connection, embedding=embedding, distance_strategy = DistanceStrategy.COSINE)

@pytest.fixture
def texts() -> List[str]:
    return ["foo", "bar", "baz"]

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_add_texts(texts: List[str]) -> None:
    """Test end to end construction and search."""
    vectordb.add_texts(texts)
    results = vectordb.similarity_search("foo", k=2)
    assert results == [Document(page_content="foo"), Document(page_content="bar")]