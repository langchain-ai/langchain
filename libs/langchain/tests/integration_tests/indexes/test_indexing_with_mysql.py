import os
from typing import List
import pytest
from langchain_core.documents import Document
from sqlalchemy import text
from langchain.indexes import SQLRecordManager, index
from tests.unit_tests.indexes.test_indexing import InMemoryVectorStore
collection_name = os.environ.get("collection_name")
namespace = os.environ.get("namespace")
DB_URL = os.environ.get('DB_URL')
@pytest.fixture
def documents() -> List[Document]:
    documents = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
        Document(
            page_content="This is another document.",
            metadata={"source": "2"},
        ),
    ]
    return documents
@pytest.fixture
def record_manager() -> None:
    record_manager = SQLRecordManager(namespace, db_url=DB_URL)
    record_manager.create_schema()
    # enusre the index table does not contain any remains from past
    with record_manager.engine.connect() as connection:
        drop_sql = text("TRUNCATE TABLE [upsertion_record]")
        connection.execute(drop_sql)
        connection.commit()
    return record_manager
def test_create_index(record_manager: SQLRecordManager) -> None:
    with record_manager.engine.connect() as connection:
        # confirm schema was created
        get_cols_sql = text(
            "SELECT COLUMN_NAME, DATA_TYPE "
            "FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_NAME = 'upsertion_record';"
        )
        res = connection.execute(get_cols_sql)
        columns_list = [row[0] for row in res]
        assert columns_list == ["uuid", "key", "namespace", "group_id", "updated_at"]
def test_indexing_documents(
    record_manager: SQLRecordManager, documents: List[Document]
) -> None:
    vector_store = InMemoryVectorStore()
    # test adding documents
    indexing_result = index(
        documents, record_manager, vector_store, cleanup="incremental", source_id_key="source"
    )
    assert indexing_result == {
        "num_added": 2,
        "num_updated": 0,
        "num_skipped": 0,
        "num_deleted": 0,
    }
    docs_input = [i.metadata["source"] for i in documents]
    docs_in_store = [i[1].metadata["source"] for i in vector_store.store.items()]
    assert sorted(docs_input) == sorted(docs_in_store)
    # testing deletion of document via indexing with cleanup="full"
    indexing_result = index(
        documents[1:],
        record_manager,
        vector_store,
        cleanup="full",
        source_id_key="source",
    )
    assert indexing_result == {
        "num_added": 0,
        "num_updated": 0,
        "num_skipped": 1,
        "num_deleted": 1,
    }
    docs_in_store = [i[1].metadata["source"] for i in vector_store.store.items()]
    assert sorted(docs_input[1:]) == sorted(docs_in_store)