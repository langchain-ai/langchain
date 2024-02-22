import os

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine

from langchain_community.document_loaders import TiDBLoader

try:
    CONNECTION_STRING = os.getenv("TEST_TiDB_CONNECTION_URL", "")

    if CONNECTION_STRING == "":
        raise OSError("TEST_TiDB_URL environment variable is not set")

    tidb_available = True
except (OSError, ImportError):
    tidb_available = False


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_load_documents() -> None:
    """Test loading documents from TiDB."""

    # Connect to the database
    engine = create_engine(CONNECTION_STRING)
    metadata = MetaData()
    table_name = "tidb_loader_intergration_test"

    # Create a test table
    test_table = Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(255)),
        Column("description", String(255)),
    )
    metadata.create_all(engine)

    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(
                test_table.insert(),
                [
                    {"name": "Item 1", "description": "Description of Item 1"},
                    {"name": "Item 2", "description": "Description of Item 2"},
                    {"name": "Item 3", "description": "Description of Item 3"},
                ],
            )
            transaction.commit()
        except:
            transaction.rollback()
            raise

    loader = TiDBLoader(
        connection_string=CONNECTION_STRING,
        query=f"SELECT * FROM {table_name};",
        page_content_columns=["name", "description"],
        metadata_columns=["id"],
    )
    documents = loader.load()
    test_table.drop(bind=engine)

    # check
    assert len(documents) == 3
    assert (
        documents[0].page_content == "name: Item 1\ndescription: Description of Item 1"
    )
    assert documents[0].metadata == {"id": 1}
    assert (
        documents[1].page_content == "name: Item 2\ndescription: Description of Item 2"
    )
    assert documents[1].metadata == {"id": 2}
    assert (
        documents[2].page_content == "name: Item 3\ndescription: Description of Item 3"
    )
    assert documents[2].metadata == {"id": 3}
