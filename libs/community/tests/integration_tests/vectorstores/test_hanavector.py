"""Test HANA vectorstore functionality."""
from typing import List

import numpy as np
import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores.hanavector import HanaDB
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import ConsistentFakeEmbeddings
from hdbcli import dbapi
import os


try:
    from hdbcli import dbapi
    hanadb_installed = True
except ImportError:
    hanadb_installed = False

embedding = ConsistentFakeEmbeddings()
connection = dbapi.connect(
        address=os.environ.get("DB_ADDRESS"),
        port=30015,
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        autocommit=False,
        sslValidateCertificate=False
)

@pytest.fixture
def texts() -> List[str]:
    return ["foo", "bar", "baz"]

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_add_texts(texts: List[str]) -> None:
    """Test end to end construction and search."""
    vectordb = HanaDB(connection=connection, embedding=embedding, distance_strategy = DistanceStrategy.COSINE, table_name="WTF")
    vectordb.add_texts(texts)
    results = vectordb.similarity_search("foo", k=1)
    assert results == [Document(page_content="foo")]

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_non_existing_table(texts: List[str]) -> None:
    """Test end to end construction and search."""
    table_name = "NON_EXISTING"
    # Delete table if it exists
    try:
        cur = connection.cursor()
        sql_str = f"DROP TABLE {table_name}"
        cur.execute(sql_str)
    except:
        pass        
    finally:
        cur.close()

    # Check if table is created
    vectordb = HanaDB(connection=connection, embedding=embedding, distance_strategy = DistanceStrategy.COSINE, table_name=table_name)

    assert vectordb._table_exists(table_name)

