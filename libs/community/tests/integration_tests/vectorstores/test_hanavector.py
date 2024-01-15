"""Test HANA vectorstore functionality."""
from typing import List

import numpy as np
import pytest

from langchain_community.vectorstores import HanaDB
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import ConsistentFakeEmbeddings
from hdbcli import dbapi
import os

try:
    from hdbcli import dbapi
    hanadb_installed = True
except ImportError:
    hanadb_installed = False

class NormalizedFakeEmbeddings(ConsistentFakeEmbeddings):
    """Fake embeddings with normalization. For testing purposes."""

    def normalize(self, vector: List[float]) -> List[float]:
        """Normalize vector."""
        return [float(v / np.linalg.norm(vector)) for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.normalize(v) for v in super().embed_documents(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.normalize(super().embed_query(text))

embedding = NormalizedFakeEmbeddings()
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

@pytest.fixture
def metadatas() -> List[str]:
    return [{"start": 0, "end": 100}, {"start": 100, "end": 200},  {"start": 200, "end": 300}]

def drop_table(connection, table_name):
    try:
        cur = connection.cursor()
        sql_str = f"DROP TABLE {table_name}"
        cur.execute(sql_str)
    except:
        pass        
    finally:
        cur.close()

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_non_existing_table() -> None:
    """Test end to end construction and search."""
    table_name = "NON_EXISTING"
    # Delete table if it exists
    drop_table(connection, table_name)

    # Check if table is created
    vectordb = HanaDB(connection=connection, embedding=embedding, distance_strategy = DistanceStrategy.COSINE, table_name=table_name)

    assert vectordb._table_exists(table_name)

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_table_with_missing_columns() -> None:
    table_name = "EXISTING_WRONG"
    try:
        drop_table(connection, table_name)
        cur = connection.cursor()
        sql_str = f"CREATE TABLE {table_name}(WRONG_COL NVARCHAR(500));"
        cur.execute(sql_str)
    except:
        pass        
    finally:
        cur.close()

    # Check if table is created
    exception_occured = False
    try:
        vectordb = HanaDB(connection=connection, embedding=embedding, distance_strategy = DistanceStrategy.COSINE, table_name=table_name)
        exception_occured = False
    except:
        exception_occured = True
    assert exception_occured

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_table_with_wrong_typed_columns() -> None:
    table_name = "EXISTING_WRONG"
    content_field = "DOC_TEXT"
    metadata_field = "DOC_META"
    vector_field = "DOC_VECTOR"
    try:
        drop_table(connection, table_name)
        cur = connection.cursor()
        sql_str = f"CREATE TABLE {table_name}({content_field} INTEGER, {metadata_field} INTEGER, {vector_field} INTEGER);"
        cur.execute(sql_str)
    except:
        pass        
    finally:
        cur.close()

    # Check if table is created
    exception_occured = False
    try:
        vectordb = HanaDB(connection=connection, embedding=embedding, distance_strategy = DistanceStrategy.COSINE, table_name=table_name)
        exception_occured = False
    except AttributeError as err:
        print(err)
        exception_occured = True
    assert exception_occured

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_non_existing_table_fixed_vector_length() -> None:
    """Test end to end construction and search."""
    table_name = "NON_EXISTING"
    vector_field = "MY_VECTOR"
    vector_field_length = 42
    # Delete table if it exists
    drop_table(connection, table_name)

    # Check if table is created
    vectordb = HanaDB(connection=connection, embedding=embedding, distance_strategy = DistanceStrategy.COSINE, table_name=table_name, 
                      vector_field = vector_field, vector_field_length = vector_field_length)

    assert vectordb._table_exists(table_name)
    vectordb._check_column(table_name, vector_field, "REAL_VECTOR", vector_field_length)

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_add_texts(texts: List[str]) -> None:
    table_name = "TEST_TABLE"
    # Delete table if it exists
    drop_table(connection, table_name)

    # Check if table is created
    vectordb = HanaDB(connection=connection, embedding=embedding, table_name=table_name)

    vectordb.add_texts(texts = texts)

    # chech that embeddings have been created in the table
    number_of_texts = len(texts)
    number_of_rows = -1
    sql_str = f"SELECT COUNT(*) FROM {table_name}"
    cur = connection.cursor()
    cur.execute(sql_str)
    if cur.has_result_set():
        rows = cur.fetchall()
        number_of_rows = rows[0][0] 
    assert number_of_rows == number_of_texts


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_from_texts(texts: List[str]) -> None:
    table_name = "TEST_TABLE"
    # Delete table if it exists
    drop_table(connection, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(connection=connection, texts = texts, embedding=embedding, table_name=table_name)
    # test if vectorDB is instance of HanaDB
    assert isinstance(vectorDB, HanaDB) 

    # chech that embeddings have been created in the table
    number_of_texts = len(texts)
    number_of_rows = -1
    sql_str = f"SELECT COUNT(*) FROM {table_name}"
    cur = connection.cursor()
    cur.execute(sql_str)
    if cur.has_result_set():
        rows = cur.fetchall()
        number_of_rows = rows[0][0] 
    assert number_of_rows == number_of_texts

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_simple(texts: List[str]) -> None:
    table_name = "TEST_TABLE"
    # Delete table if it exists
    drop_table(connection, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(connection=connection, texts = texts, embedding=embedding, table_name=table_name)

    assert texts[0] == vectorDB.similarity_search(texts[0], 1)[0].page_content
    assert texts[1] != vectorDB.similarity_search(texts[0], 1)[0].page_content

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_simple_euclidean_distance(texts: List[str]) -> None:
    table_name = "TEST_TABLE"
    # Delete table if it exists
    drop_table(connection, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(connection=connection, texts = texts, embedding=embedding, table_name=table_name, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)

    assert texts[0] == vectorDB.similarity_search(texts[0], 1)[0].page_content
    assert texts[1] != vectorDB.similarity_search(texts[0], 1)[0].page_content

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_metadata(texts: List[str], metadatas: List[dict]) -> None:
    table_name = "TEST_TABLE"
    # Delete table if it exists
    drop_table(connection, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(connection=connection, texts = texts, metadatas=metadatas, embedding=embedding, table_name=table_name)

    search_result = vectorDB.similarity_search(texts[0], 3)

    assert texts[0] == search_result[0].page_content
    assert metadatas[0]["start"] == search_result[0].metadata["start"]
    assert metadatas[0]["end"] == search_result[0].metadata["end"]
    assert texts[1] != search_result[0].page_content
    assert metadatas[1]["start"] != search_result[0].metadata["start"]
    assert metadatas[1]["end"] != search_result[0].metadata["end"]

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_metadata_filter(texts: List[str], metadatas: List[dict]) -> None:
    table_name = "TEST_TABLE"
    # Delete table if it exists
    drop_table(connection, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(connection=connection, texts = texts, metadatas=metadatas, embedding=embedding, table_name=table_name)

    search_result = vectorDB.similarity_search(texts[0], 3, filter={"start": 100})

    assert len(search_result) == 1
    assert texts[1] == search_result[0].page_content
    assert metadatas[1]["start"] == search_result[0].metadata["start"]
    assert metadatas[1]["end"] == search_result[0].metadata["end"]

    search_result = vectorDB.similarity_search(texts[0], 3, filter={"start": 100, "end": 150})
    assert len(search_result) == 0

    search_result = vectorDB.similarity_search(texts[0], 3, filter={"start": 100, "end": 200})
    assert len(search_result) == 1
    assert texts[1] == search_result[0].page_content
    assert metadatas[1]["start"] == search_result[0].metadata["start"]
    assert metadatas[1]["end"] == search_result[0].metadata["end"]

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_score(texts: List[str], metadatas: List[dict]) -> None:
    table_name = "TEST_TABLE"
    # Delete table if it exists
    drop_table(connection, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(connection=connection, texts = texts, embedding=embedding, table_name=table_name)

    search_result = vectorDB.similarity_search_with_score(texts[0], 3)

    assert search_result[0][0].page_content == texts[0]
    assert search_result[0][1] == 1.0
    assert search_result[1][1] <= search_result[0][1]
    assert search_result[2][1] <= search_result[1][1]
    assert search_result[2][1] >= 0.0

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_score_with_euclidian_distance(texts: List[str], metadatas: List[dict]) -> None:
    table_name = "TEST_TABLE"
    # Delete table if it exists
    drop_table(connection, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(connection=connection, texts = texts, embedding=embedding, table_name=table_name, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)

    search_result = vectorDB.similarity_search_with_score(texts[0], 3)

    assert search_result[0][0].page_content == texts[0]
    assert search_result[0][1] == 1.0
    assert search_result[1][1] <= search_result[0][1]
    assert search_result[2][1] <= search_result[1][1]
    assert search_result[2][1] >= 0.0









