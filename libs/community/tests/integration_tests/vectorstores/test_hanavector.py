"""Test HANA vectorstore functionality."""

import os
import random
from typing import Any, Dict, List

import numpy as np
import pytest

from langchain_community.vectorstores import HanaDB
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)
from tests.integration_tests.vectorstores.fixtures.filtering_test_cases import (
    DOCUMENTS,
    TYPE_1_FILTERING_TEST_CASES,
    TYPE_2_FILTERING_TEST_CASES,
    TYPE_3_FILTERING_TEST_CASES,
    TYPE_4_FILTERING_TEST_CASES,
    TYPE_5_FILTERING_TEST_CASES,
)

TYPE_4B_FILTERING_TEST_CASES = [
    # Test $nin, which is missing in TYPE_4_FILTERING_TEST_CASES
    (
        {"name": {"$nin": ["adam", "bob"]}},
        [3],
    ),
]


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


class ConfigData:
    def __init__(self):  # type: ignore[no-untyped-def]
        self.conn = None
        self.schema_name = ""


test_setup = ConfigData()


def generateSchemaName(cursor):  # type: ignore[no-untyped-def]
    # return "Langchain"
    cursor.execute(
        "SELECT REPLACE(CURRENT_UTCDATE, '-', '') || '_' || BINTOHEX(SYSUUID) FROM "
        "DUMMY;"
    )
    if cursor.has_result_set():
        rows = cursor.fetchall()
        uid = rows[0][0]
    else:
        uid = random.randint(1, 100000000)
    return f"VEC_{uid}"


def setup_module(module):  # type: ignore[no-untyped-def]
    test_setup.conn = dbapi.connect(
        address=os.environ.get("HANA_DB_ADDRESS"),
        port=os.environ.get("HANA_DB_PORT"),
        user=os.environ.get("HANA_DB_USER"),
        password=os.environ.get("HANA_DB_PASSWORD"),
        autocommit=True,
        sslValidateCertificate=False,
        # encrypt=True
    )
    try:
        cur = test_setup.conn.cursor()
        test_setup.schema_name = generateSchemaName(cur)
        sql_str = f"CREATE SCHEMA {test_setup.schema_name}"
        cur.execute(sql_str)
        sql_str = f"SET SCHEMA {test_setup.schema_name}"
        cur.execute(sql_str)
    except dbapi.ProgrammingError:
        pass
    finally:
        cur.close()


def teardown_module(module):  # type: ignore[no-untyped-def]
    # return
    try:
        cur = test_setup.conn.cursor()
        sql_str = f"DROP SCHEMA {test_setup.schema_name} CASCADE"
        cur.execute(sql_str)
    except dbapi.ProgrammingError:
        pass
    finally:
        cur.close()


@pytest.fixture
def texts() -> List[str]:
    return ["foo", "bar", "baz", "bak", "cat"]


@pytest.fixture
def metadatas() -> List[str]:
    return [
        {"start": 0, "end": 100, "quality": "good", "ready": True},  # type: ignore[list-item]
        {"start": 100, "end": 200, "quality": "bad", "ready": False},  # type: ignore[list-item]
        {"start": 200, "end": 300, "quality": "ugly", "ready": True},  # type: ignore[list-item]
        {"start": 200, "quality": "ugly", "ready": True, "Owner": "Steve"},  # type: ignore[list-item]
        {"start": 300, "quality": "ugly", "Owner": "Steve"},  # type: ignore[list-item]
    ]


def drop_table(connection, table_name):  # type: ignore[no-untyped-def]
    try:
        cur = connection.cursor()
        sql_str = f"DROP TABLE {table_name}"
        cur.execute(sql_str)
    except dbapi.ProgrammingError:
        pass
    finally:
        cur.close()


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_non_existing_table() -> None:
    """Test end to end construction and search."""
    table_name = "NON_EXISTING"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectordb = HanaDB(
        connection=test_setup.conn,
        embedding=embedding,
        distance_strategy=DistanceStrategy.COSINE,
        table_name=table_name,
    )

    assert vectordb._table_exists(table_name)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_table_with_missing_columns() -> None:
    table_name = "EXISTING_MISSING_COLS"
    try:
        drop_table(test_setup.conn, table_name)
        cur = test_setup.conn.cursor()
        sql_str = f"CREATE TABLE {table_name}(WRONG_COL NVARCHAR(500));"
        cur.execute(sql_str)
    finally:
        cur.close()

    # Check if table is created
    exception_occured = False
    try:
        HanaDB(
            connection=test_setup.conn,
            embedding=embedding,
            distance_strategy=DistanceStrategy.COSINE,
            table_name=table_name,
        )
        exception_occured = False
    except AttributeError:
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_table_with_nvarchar_content(texts: List[str]) -> None:
    table_name = "EXISTING_NVARCHAR"
    content_column = "TEST_TEXT"
    metadata_column = "TEST_META"
    vector_column = "TEST_VECTOR"
    try:
        drop_table(test_setup.conn, table_name)
        cur = test_setup.conn.cursor()
        sql_str = (
            f"CREATE TABLE {table_name}({content_column} NVARCHAR(2048), "
            f"{metadata_column} NVARCHAR(2048), {vector_column} REAL_VECTOR);"
        )
        cur.execute(sql_str)
    finally:
        cur.close()

    vectordb = HanaDB(
        connection=test_setup.conn,
        embedding=embedding,
        distance_strategy=DistanceStrategy.COSINE,
        table_name=table_name,
        content_column=content_column,
        metadata_column=metadata_column,
        vector_column=vector_column,
    )

    vectordb.add_texts(texts=texts)

    # check that embeddings have been created in the table
    number_of_texts = len(texts)
    number_of_rows = -1
    sql_str = f"SELECT COUNT(*) FROM {table_name}"
    cur = test_setup.conn.cursor()
    cur.execute(sql_str)
    if cur.has_result_set():
        rows = cur.fetchall()
        number_of_rows = rows[0][0]
    assert number_of_rows == number_of_texts


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_table_with_wrong_typed_columns() -> None:
    table_name = "EXISTING_WRONG_TYPES"
    content_column = "DOC_TEXT"
    metadata_column = "DOC_META"
    vector_column = "DOC_VECTOR"
    try:
        drop_table(test_setup.conn, table_name)
        cur = test_setup.conn.cursor()
        sql_str = (
            f"CREATE TABLE {table_name}({content_column} INTEGER, "
            f"{metadata_column} INTEGER, {vector_column} INTEGER);"
        )
        cur.execute(sql_str)
    finally:
        cur.close()

    # Check if table is created
    exception_occured = False
    try:
        HanaDB(
            connection=test_setup.conn,
            embedding=embedding,
            distance_strategy=DistanceStrategy.COSINE,
            table_name=table_name,
        )
        exception_occured = False
    except AttributeError as err:
        print(err)  # noqa: T201
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_non_existing_table_fixed_vector_length() -> None:
    """Test end to end construction and search."""
    table_name = "NON_EXISTING"
    vector_column = "MY_VECTOR"
    vector_column_length = 42
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectordb = HanaDB(
        connection=test_setup.conn,
        embedding=embedding,
        distance_strategy=DistanceStrategy.COSINE,
        table_name=table_name,
        vector_column=vector_column,
        vector_column_length=vector_column_length,
    )

    assert vectordb._table_exists(table_name)
    vectordb._check_column(
        table_name, vector_column, "REAL_VECTOR", vector_column_length
    )


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_add_texts(texts: List[str]) -> None:
    table_name = "TEST_TABLE_ADD_TEXTS"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectordb = HanaDB(
        connection=test_setup.conn, embedding=embedding, table_name=table_name
    )

    vectordb.add_texts(texts=texts)

    # check that embeddings have been created in the table
    number_of_texts = len(texts)
    number_of_rows = -1
    sql_str = f"SELECT COUNT(*) FROM {table_name}"
    cur = test_setup.conn.cursor()
    cur.execute(sql_str)
    if cur.has_result_set():
        rows = cur.fetchall()
        number_of_rows = rows[0][0]
    assert number_of_rows == number_of_texts


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_from_texts(texts: List[str]) -> None:
    table_name = "TEST_TABLE_FROM_TEXTS"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )
    # test if vectorDB is instance of HanaDB
    assert isinstance(vectorDB, HanaDB)

    # check that embeddings have been created in the table
    number_of_texts = len(texts)
    number_of_rows = -1
    sql_str = f"SELECT COUNT(*) FROM {table_name}"
    cur = test_setup.conn.cursor()
    cur.execute(sql_str)
    if cur.has_result_set():
        rows = cur.fetchall()
        number_of_rows = rows[0][0]
    assert number_of_rows == number_of_texts


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_simple(texts: List[str]) -> None:
    table_name = "TEST_TABLE_SEARCH_SIMPLE"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    assert texts[0] == vectorDB.similarity_search(texts[0], 1)[0].page_content
    assert texts[1] != vectorDB.similarity_search(texts[0], 1)[0].page_content


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_by_vector_simple(texts: List[str]) -> None:
    table_name = "TEST_TABLE_SEARCH_SIMPLE_VECTOR"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    vector = embedding.embed_query(texts[0])
    assert texts[0] == vectorDB.similarity_search_by_vector(vector, 1)[0].page_content
    assert texts[1] != vectorDB.similarity_search_by_vector(vector, 1)[0].page_content


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_simple_euclidean_distance(
    texts: List[str],
) -> None:
    table_name = "TEST_TABLE_SEARCH_EUCLIDIAN"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    assert texts[0] == vectorDB.similarity_search(texts[0], 1)[0].page_content
    assert texts[1] != vectorDB.similarity_search(texts[0], 1)[0].page_content


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_metadata(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_METADATA"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.similarity_search(texts[0], 3)

    assert texts[0] == search_result[0].page_content
    assert metadatas[0]["start"] == search_result[0].metadata["start"]
    assert metadatas[0]["end"] == search_result[0].metadata["end"]
    assert texts[1] != search_result[0].page_content
    assert metadatas[1]["start"] != search_result[0].metadata["start"]
    assert metadatas[1]["end"] != search_result[0].metadata["end"]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_metadata_filter(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_FILTER"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.similarity_search(texts[0], 3, filter={"start": 100})

    assert len(search_result) == 1
    assert texts[1] == search_result[0].page_content
    assert metadatas[1]["start"] == search_result[0].metadata["start"]
    assert metadatas[1]["end"] == search_result[0].metadata["end"]

    search_result = vectorDB.similarity_search(
        texts[0], 3, filter={"start": 100, "end": 150}
    )
    assert len(search_result) == 0

    search_result = vectorDB.similarity_search(
        texts[0], 3, filter={"start": 100, "end": 200}
    )
    assert len(search_result) == 1
    assert texts[1] == search_result[0].page_content
    assert metadatas[1]["start"] == search_result[0].metadata["start"]
    assert metadatas[1]["end"] == search_result[0].metadata["end"]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_metadata_filter_string(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_FILTER_STRING"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.similarity_search(texts[0], 3, filter={"quality": "bad"})

    assert len(search_result) == 1
    assert texts[1] == search_result[0].page_content


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_metadata_filter_bool(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_FILTER_BOOL"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.similarity_search(texts[0], 3, filter={"ready": False})

    assert len(search_result) == 1
    assert texts[1] == search_result[0].page_content


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_metadata_filter_invalid_type(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_FILTER_INVALID_TYPE"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
    )

    exception_occured = False
    try:
        vectorDB.similarity_search(texts[0], 3, filter={"wrong_type": 0.1})
    except ValueError:
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_score(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_SCORE"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.similarity_search_with_score(texts[0], 3)

    assert search_result[0][0].page_content == texts[0]
    assert search_result[0][1] == 1.0
    assert search_result[1][1] <= search_result[0][1]
    assert search_result[2][1] <= search_result[1][1]
    assert search_result[2][1] >= 0.0


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_relevance_score(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_REL_SCORE"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.similarity_search_with_relevance_scores(texts[0], 3)

    assert search_result[0][0].page_content == texts[0]
    assert search_result[0][1] == 1.0
    assert search_result[1][1] <= search_result[0][1]
    assert search_result[2][1] <= search_result[1][1]
    assert search_result[2][1] >= 0.0


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_relevance_score_with_euclidian_distance(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_REL_SCORE_EUCLIDIAN"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    search_result = vectorDB.similarity_search_with_relevance_scores(texts[0], 3)

    assert search_result[0][0].page_content == texts[0]
    assert search_result[0][1] == 1.0
    assert search_result[1][1] <= search_result[0][1]
    assert search_result[2][1] <= search_result[1][1]
    assert search_result[2][1] >= 0.0


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_score_with_euclidian_distance(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_SCORE_DISTANCE"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    search_result = vectorDB.similarity_search_with_score(texts[0], 3)

    assert search_result[0][0].page_content == texts[0]
    assert search_result[0][1] == 0.0
    assert search_result[1][1] >= search_result[0][1]
    assert search_result[2][1] >= search_result[1][1]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_delete_with_filter(texts: List[str], metadatas: List[dict]) -> None:
    table_name = "TEST_TABLE_DELETE_FILTER"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Fill table
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.similarity_search(texts[0], 10)
    assert len(search_result) == 5

    # Delete one of the three entries
    assert vectorDB.delete(filter={"start": 100, "end": 200})

    search_result = vectorDB.similarity_search(texts[0], 10)
    assert len(search_result) == 4


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
async def test_hanavector_delete_with_filter_async(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_DELETE_FILTER_ASYNC"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Fill table
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.similarity_search(texts[0], 10)
    assert len(search_result) == 5

    # Delete one of the three entries
    assert await vectorDB.adelete(filter={"start": 100, "end": 200})

    search_result = vectorDB.similarity_search(texts[0], 10)
    assert len(search_result) == 4


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_delete_all_with_empty_filter(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_DELETE_ALL"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Fill table
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.similarity_search(texts[0], 3)
    assert len(search_result) == 3

    # Delete all entries
    assert vectorDB.delete(filter={})

    search_result = vectorDB.similarity_search(texts[0], 3)
    assert len(search_result) == 0


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_delete_called_wrong(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_DELETE_FILTER_WRONG"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Fill table
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
    )

    # Delete without filter parameter
    exception_occured = False
    try:
        vectorDB.delete()
    except ValueError:
        exception_occured = True
    assert exception_occured

    # Delete with ids parameter
    exception_occured = False
    try:
        vectorDB.delete(ids=["id1", "id"], filter={"start": 100, "end": 200})
    except ValueError:
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_max_marginal_relevance_search(texts: List[str]) -> None:
    table_name = "TEST_TABLE_MAX_RELEVANCE"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.max_marginal_relevance_search(texts[0], k=2, fetch_k=20)

    assert len(search_result) == 2
    assert search_result[0].page_content == texts[0]
    assert search_result[1].page_content != texts[0]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_max_marginal_relevance_search_vector(texts: List[str]) -> None:
    table_name = "TEST_TABLE_MAX_RELEVANCE_VECTOR"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.max_marginal_relevance_search_by_vector(
        embedding.embed_query(texts[0]), k=2, fetch_k=20
    )

    assert len(search_result) == 2
    assert search_result[0].page_content == texts[0]
    assert search_result[1].page_content != texts[0]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
async def test_hanavector_max_marginal_relevance_search_async(texts: List[str]) -> None:
    table_name = "TEST_TABLE_MAX_RELEVANCE_ASYNC"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = await vectorDB.amax_marginal_relevance_search(
        texts[0], k=2, fetch_k=20
    )

    assert len(search_result) == 2
    assert search_result[0].page_content == texts[0]
    assert search_result[1].page_content != texts[0]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_filter_prepared_statement_params(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "TEST_TABLE_FILTER_PARAM"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    # Check if table is created
    HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
    )

    cur = test_setup.conn.cursor()
    sql_str = (
        f"SELECT * FROM {table_name} WHERE JSON_VALUE(VEC_META, '$.start') = '100'"
    )
    cur.execute(sql_str)
    rows = cur.fetchall()
    assert len(rows) == 1

    query_value = 100
    sql_str = f"SELECT * FROM {table_name} WHERE JSON_VALUE(VEC_META, '$.start') = ?"
    cur.execute(sql_str, (query_value))
    rows = cur.fetchall()
    assert len(rows) == 1

    sql_str = (
        f"SELECT * FROM {table_name} WHERE JSON_VALUE(VEC_META, '$.quality') = 'good'"
    )
    cur.execute(sql_str)
    rows = cur.fetchall()
    assert len(rows) == 1

    query_value = "good"  # type: ignore[assignment]
    sql_str = f"SELECT * FROM {table_name} WHERE JSON_VALUE(VEC_META, '$.quality') = ?"
    cur.execute(sql_str, (query_value))
    rows = cur.fetchall()
    assert len(rows) == 1

    sql_str = (
        f"SELECT * FROM {table_name} WHERE JSON_VALUE(VEC_META, '$.ready') = false"
    )
    cur.execute(sql_str)
    rows = cur.fetchall()
    assert len(rows) == 1

    # query_value = True
    query_value = "true"  # type: ignore[assignment]
    sql_str = f"SELECT * FROM {table_name} WHERE JSON_VALUE(VEC_META, '$.ready') = ?"
    cur.execute(sql_str, (query_value))
    rows = cur.fetchall()
    assert len(rows) == 3

    # query_value = False
    query_value = "false"  # type: ignore[assignment]
    sql_str = f"SELECT * FROM {table_name} WHERE JSON_VALUE(VEC_META, '$.ready') = ?"
    cur.execute(sql_str, (query_value))
    rows = cur.fetchall()
    assert len(rows) == 1


def test_invalid_metadata_keys(texts: List[str], metadatas: List[dict]) -> None:
    table_name = "TEST_TABLE_INVALID_METADATA"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    invalid_metadatas = [
        {"sta rt": 0, "end": 100, "quality": "good", "ready": True},
    ]
    exception_occured = False
    try:
        HanaDB.from_texts(
            connection=test_setup.conn,
            texts=texts,
            metadatas=invalid_metadatas,
            embedding=embedding,
            table_name=table_name,
        )
    except ValueError:
        exception_occured = True
    assert exception_occured

    invalid_metadatas = [
        {"sta/nrt": 0, "end": 100, "quality": "good", "ready": True},
    ]
    exception_occured = False
    try:
        HanaDB.from_texts(
            connection=test_setup.conn,
            texts=texts,
            metadatas=invalid_metadatas,
            embedding=embedding,
            table_name=table_name,
        )
    except ValueError:
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_table_mixed_case_names(texts: List[str]) -> None:
    table_name = "MyTableName"
    content_column = "TextColumn"
    metadata_column = "MetaColumn"
    vector_column = "VectorColumn"

    vectordb = HanaDB(
        connection=test_setup.conn,
        embedding=embedding,
        distance_strategy=DistanceStrategy.COSINE,
        table_name=table_name,
        content_column=content_column,
        metadata_column=metadata_column,
        vector_column=vector_column,
    )

    vectordb.add_texts(texts=texts)

    # check that embeddings have been created in the table
    number_of_texts = len(texts)
    number_of_rows = -1
    sql_str = f'SELECT COUNT(*) FROM "{table_name}"'
    cur = test_setup.conn.cursor()
    cur.execute(sql_str)
    if cur.has_result_set():
        rows = cur.fetchall()
        number_of_rows = rows[0][0]
    assert number_of_rows == number_of_texts

    # check results of similarity search
    assert texts[0] == vectordb.similarity_search(texts[0], 1)[0].page_content


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_enhanced_filter_1() -> None:
    table_name = "TEST_TABLE_ENHANCED_FILTER_1"
    # Delete table if it exists
    drop_table(test_setup.conn, table_name)

    vectorDB = HanaDB(
        connection=test_setup.conn,
        embedding=embedding,
        table_name=table_name,
    )

    vectorDB.add_documents(DOCUMENTS)


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_1_FILTERING_TEST_CASES)
@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_pgvector_with_with_metadata_filters_1(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    table_name = "TEST_TABLE_ENHANCED_FILTER_1"
    drop_table(test_setup.conn, table_name)

    vectorDB = HanaDB(
        connection=test_setup.conn,
        embedding=embedding,
        table_name=table_name,
    )

    vectorDB.add_documents(DOCUMENTS)

    docs = vectorDB.similarity_search("meow", k=5, filter=test_filter)
    ids = [doc.metadata["id"] for doc in docs]
    assert len(ids) == len(expected_ids), test_filter
    assert set(ids).issubset(expected_ids), test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_2_FILTERING_TEST_CASES)
@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_pgvector_with_with_metadata_filters_2(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    table_name = "TEST_TABLE_ENHANCED_FILTER_2"
    drop_table(test_setup.conn, table_name)

    vectorDB = HanaDB(
        connection=test_setup.conn,
        embedding=embedding,
        table_name=table_name,
    )

    vectorDB.add_documents(DOCUMENTS)

    docs = vectorDB.similarity_search("meow", k=5, filter=test_filter)
    ids = [doc.metadata["id"] for doc in docs]
    assert len(ids) == len(expected_ids), test_filter
    assert set(ids).issubset(expected_ids), test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_3_FILTERING_TEST_CASES)
@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_pgvector_with_with_metadata_filters_3(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    table_name = "TEST_TABLE_ENHANCED_FILTER_3"
    drop_table(test_setup.conn, table_name)

    vectorDB = HanaDB(
        connection=test_setup.conn,
        embedding=embedding,
        table_name=table_name,
    )

    vectorDB.add_documents(DOCUMENTS)

    docs = vectorDB.similarity_search("meow", k=5, filter=test_filter)
    ids = [doc.metadata["id"] for doc in docs]
    assert len(ids) == len(expected_ids), test_filter
    assert set(ids).issubset(expected_ids), test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_4_FILTERING_TEST_CASES)
@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_pgvector_with_with_metadata_filters_4(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    table_name = "TEST_TABLE_ENHANCED_FILTER_4"
    drop_table(test_setup.conn, table_name)

    vectorDB = HanaDB(
        connection=test_setup.conn,
        embedding=embedding,
        table_name=table_name,
    )

    vectorDB.add_documents(DOCUMENTS)

    docs = vectorDB.similarity_search("meow", k=5, filter=test_filter)
    ids = [doc.metadata["id"] for doc in docs]
    assert len(ids) == len(expected_ids), test_filter
    assert set(ids).issubset(expected_ids), test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_4B_FILTERING_TEST_CASES)
@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_pgvector_with_with_metadata_filters_4b(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    table_name = "TEST_TABLE_ENHANCED_FILTER_4B"
    drop_table(test_setup.conn, table_name)

    vectorDB = HanaDB(
        connection=test_setup.conn,
        embedding=embedding,
        table_name=table_name,
    )

    vectorDB.add_documents(DOCUMENTS)

    docs = vectorDB.similarity_search("meow", k=5, filter=test_filter)
    ids = [doc.metadata["id"] for doc in docs]
    assert len(ids) == len(expected_ids), test_filter
    assert set(ids).issubset(expected_ids), test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_5_FILTERING_TEST_CASES)
@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_pgvector_with_with_metadata_filters_5(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    table_name = "TEST_TABLE_ENHANCED_FILTER_5"
    drop_table(test_setup.conn, table_name)

    vectorDB = HanaDB(
        connection=test_setup.conn,
        embedding=embedding,
        table_name=table_name,
    )

    vectorDB.add_documents(DOCUMENTS)

    docs = vectorDB.similarity_search("meow", k=5, filter=test_filter)
    ids = [doc.metadata["id"] for doc in docs]
    assert len(ids) == len(expected_ids), test_filter
    assert set(ids).issubset(expected_ids), test_filter


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_preexisting_specific_columns_for_metadata_fill(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "PREEXISTING_FILTER_COLUMNS"
    # drop_table(test_setup.conn, table_name)

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"Owner" NVARCHAR(100), '
        f'"quality" NVARCHAR(100));'
    )
    try:
        cur = test_setup.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
        specific_metadata_columns=["Owner", "quality"],
    )

    c = 0
    try:
        sql_str = f'SELECT COUNT(*) FROM {table_name} WHERE "quality"=' f"'ugly'"
        cur = test_setup.conn.cursor()
        cur.execute(sql_str)
        if cur.has_result_set():
            rows = cur.fetchall()
            c = rows[0][0]
    finally:
        cur.close()
    assert c == 3

    docs = vectorDB.similarity_search("hello", k=5, filter={"quality": "good"})
    assert len(docs) == 1
    assert docs[0].page_content == "foo"

    docs = vectorDB.similarity_search("hello", k=5, filter={"start": 100})
    assert len(docs) == 1
    assert docs[0].page_content == "bar"

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 100, "quality": "good"}
    )
    assert len(docs) == 0

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 0, "quality": "good"}
    )
    assert len(docs) == 1
    assert docs[0].page_content == "foo"


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_preexisting_specific_columns_for_metadata_via_array(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "PREEXISTING_FILTER_COLUMNS_VIA_ARRAY"
    # drop_table(test_setup.conn, table_name)

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"Owner" NVARCHAR(100), '
        f'"quality" NVARCHAR(100));'
    )
    try:
        cur = test_setup.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
        specific_metadata_columns=["quality"],
    )

    c = 0
    try:
        sql_str = f'SELECT COUNT(*) FROM {table_name} WHERE "quality"=' f"'ugly'"
        cur = test_setup.conn.cursor()
        cur.execute(sql_str)
        if cur.has_result_set():
            rows = cur.fetchall()
            c = rows[0][0]
    finally:
        cur.close()
    assert c == 3

    try:
        sql_str = f'SELECT COUNT(*) FROM {table_name} WHERE "Owner"=' f"'Steve'"
        cur = test_setup.conn.cursor()
        cur.execute(sql_str)
        if cur.has_result_set():
            rows = cur.fetchall()
            c = rows[0][0]
    finally:
        cur.close()
    assert c == 0

    docs = vectorDB.similarity_search("hello", k=5, filter={"quality": "good"})
    assert len(docs) == 1
    assert docs[0].page_content == "foo"

    docs = vectorDB.similarity_search("hello", k=5, filter={"start": 100})
    assert len(docs) == 1
    assert docs[0].page_content == "bar"

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 100, "quality": "good"}
    )
    assert len(docs) == 0

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 0, "quality": "good"}
    )
    assert len(docs) == 1
    assert docs[0].page_content == "foo"


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_preexisting_specific_columns_for_metadata_multiple_columns(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "PREEXISTING_FILTER_MULTIPLE_COLUMNS"
    # drop_table(test_setup.conn, table_name)

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"quality" NVARCHAR(100), '
        f'"start" INTEGER);'
    )
    try:
        cur = test_setup.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
        specific_metadata_columns=["quality", "start"],
    )

    docs = vectorDB.similarity_search("hello", k=5, filter={"quality": "good"})
    assert len(docs) == 1
    assert docs[0].page_content == "foo"

    docs = vectorDB.similarity_search("hello", k=5, filter={"start": 100})
    assert len(docs) == 1
    assert docs[0].page_content == "bar"

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 100, "quality": "good"}
    )
    assert len(docs) == 0

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 0, "quality": "good"}
    )
    assert len(docs) == 1
    assert docs[0].page_content == "foo"


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_preexisting_specific_columns_for_metadata_empty_columns(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "PREEXISTING_FILTER_MULTIPLE_COLUMNS_EMPTY"
    # drop_table(test_setup.conn, table_name)

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"quality" NVARCHAR(100), '
        f'"ready" BOOLEAN, '
        f'"start" INTEGER);'
    )
    try:
        cur = test_setup.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
        specific_metadata_columns=["quality", "ready", "start"],
    )

    docs = vectorDB.similarity_search("hello", k=5, filter={"quality": "good"})
    assert len(docs) == 1
    assert docs[0].page_content == "foo"

    docs = vectorDB.similarity_search("hello", k=5, filter={"start": 100})
    assert len(docs) == 1
    assert docs[0].page_content == "bar"

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 100, "quality": "good"}
    )
    assert len(docs) == 0

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 0, "quality": "good"}
    )
    assert len(docs) == 1
    assert docs[0].page_content == "foo"

    docs = vectorDB.similarity_search("hello", k=5, filter={"ready": True})
    assert len(docs) == 3


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_preexisting_specific_columns_for_metadata_wrong_type_or_non_existing(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "PREEXISTING_FILTER_COLUMNS_WRONG_TYPE"
    # drop_table(test_setup.conn, table_name)

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"quality" INTEGER); '
    )
    try:
        cur = test_setup.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    # Check if table is created
    exception_occured = False
    try:
        HanaDB.from_texts(
            connection=test_setup.conn,
            texts=texts,
            metadatas=metadatas,
            embedding=embedding,
            table_name=table_name,
            specific_metadata_columns=["quality"],
        )
        exception_occured = False
    except dbapi.Error:  # Nothing we should do here, hdbcli will throw an error
        exception_occured = True
    assert exception_occured  # Check if table is created

    exception_occured = False
    try:
        HanaDB.from_texts(
            connection=test_setup.conn,
            texts=texts,
            metadatas=metadatas,
            embedding=embedding,
            table_name=table_name,
            specific_metadata_columns=["NonExistingColumn"],
        )
        exception_occured = False
    except AttributeError:  # Nothing we should do here, hdbcli will throw an error
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_preexisting_specific_columns_for_returned_metadata_completeness(
    texts: List[str], metadatas: List[dict]
) -> None:
    table_name = "PREEXISTING_FILTER_COLUMNS_METADATA_COMPLETENESS"
    # drop_table(test_setup.conn, table_name)

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"quality" NVARCHAR(100), '
        f'"NonExisting" NVARCHAR(100), '
        f'"ready" BOOLEAN, '
        f'"start" INTEGER);'
    )
    try:
        cur = test_setup.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
        specific_metadata_columns=["quality", "ready", "start", "NonExisting"],
    )

    docs = vectorDB.similarity_search("hello", k=5, filter={"quality": "good"})
    assert len(docs) == 1
    assert docs[0].page_content == "foo"
    assert docs[0].metadata["end"] == 100
    assert docs[0].metadata["start"] == 0
    assert docs[0].metadata["quality"] == "good"
    assert docs[0].metadata["ready"]
    assert "NonExisting" not in docs[0].metadata.keys()


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_create_hnsw_index_with_default_values(texts: List[str]) -> None:
    table_name = "TEST_TABLE_HNSW_INDEX_DEFAULT"

    # Delete table if it exists (cleanup from previous tests)
    drop_table(test_setup.conn, table_name)

    # Create table and insert data
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    # Test the creation of HNSW index
    try:
        vectorDB.create_hnsw_index()
    except Exception as e:
        pytest.fail(f"Failed to create HNSW index: {e}")

    # Perform a search using the index to confirm its correctness
    search_result = vectorDB.max_marginal_relevance_search(texts[0], k=2, fetch_k=20)

    assert len(search_result) == 2
    assert search_result[0].page_content == texts[0]
    assert search_result[1].page_content != texts[0]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_create_hnsw_index_with_defined_values(texts: List[str]) -> None:
    table_name = "TEST_TABLE_HNSW_INDEX_DEFINED"

    # Delete table if it exists (cleanup from previous tests)
    drop_table(test_setup.conn, table_name)

    # Create table and insert data
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    # Test the creation of HNSW index with specific values
    try:
        vectorDB.create_hnsw_index(
            index_name="my_L2_index", ef_search=500, m=100, ef_construction=200
        )
    except Exception as e:
        pytest.fail(f"Failed to create HNSW index with defined values: {e}")

    # Perform a search using the index to confirm its correctness
    search_result = vectorDB.max_marginal_relevance_search(texts[0], k=2, fetch_k=20)

    assert len(search_result) == 2
    assert search_result[0].page_content == texts[0]
    assert search_result[1].page_content != texts[0]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_create_hnsw_index_after_initialization(texts: List[str]) -> None:
    table_name = "TEST_TABLE_HNSW_INDEX_AFTER_INIT"

    drop_table(test_setup.conn, table_name)

    # Initialize HanaDB without adding documents yet
    vectorDB = HanaDB(
        connection=test_setup.conn,
        embedding=embedding,
        table_name=table_name,
    )

    # Create HNSW index before adding documents
    vectorDB.create_hnsw_index(
        index_name="index_pre_add", ef_search=400, m=50, ef_construction=150
    )

    # Add texts after index creation
    vectorDB.add_texts(texts=texts)

    # Perform similarity search using the index
    search_result = vectorDB.similarity_search(texts[0], k=3)

    # Assert that search result is valid and has expected length
    assert len(search_result) == 3
    assert search_result[0].page_content == texts[0]
    assert search_result[1].page_content != texts[0]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_duplicate_hnsw_index_creation(texts: List[str]) -> None:
    table_name = "TEST_TABLE_HNSW_DUPLICATE_INDEX"

    # Delete table if it exists (cleanup from previous tests)
    drop_table(test_setup.conn, table_name)

    # Create table and insert data
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    # Create HNSW index for the first time
    vectorDB.create_hnsw_index(
        index_name="index_cosine",
        ef_search=300,
        m=80,
        ef_construction=100,
    )

    with pytest.raises(Exception):
        vectorDB.create_hnsw_index(ef_search=300, m=80, ef_construction=100)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_create_hnsw_index_invalid_m_value(texts: List[str]) -> None:
    table_name = "TEST_TABLE_HNSW_INVALID_M"

    # Cleanup: drop the table if it exists
    drop_table(test_setup.conn, table_name)

    # Create table and insert data
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    # Test invalid `m` value (too low)
    with pytest.raises(ValueError):
        vectorDB.create_hnsw_index(m=3)

    # Test invalid `m` value (too high)
    with pytest.raises(ValueError):
        vectorDB.create_hnsw_index(m=1001)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_create_hnsw_index_invalid_ef_construction(texts: List[str]) -> None:
    table_name = "TEST_TABLE_HNSW_INVALID_EF_CONSTRUCTION"

    # Cleanup: drop the table if it exists
    drop_table(test_setup.conn, table_name)

    # Create table and insert data
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    # Test invalid `ef_construction` value (too low)
    with pytest.raises(ValueError):
        vectorDB.create_hnsw_index(ef_construction=0)

    # Test invalid `ef_construction` value (too high)
    with pytest.raises(ValueError):
        vectorDB.create_hnsw_index(ef_construction=100001)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_create_hnsw_index_invalid_ef_search(texts: List[str]) -> None:
    table_name = "TEST_TABLE_HNSW_INVALID_EF_SEARCH"

    # Cleanup: drop the table if it exists
    drop_table(test_setup.conn, table_name)

    # Create table and insert data
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    # Test invalid `ef_search` value (too low)
    with pytest.raises(ValueError):
        vectorDB.create_hnsw_index(ef_search=0)

    # Test invalid `ef_search` value (too high)
    with pytest.raises(ValueError):
        vectorDB.create_hnsw_index(ef_search=100001)
