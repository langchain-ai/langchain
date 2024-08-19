"""Test SQLServer_VectorStore functionality."""

import os
from typing import Generator, List

import pytest
from langchain_core.documents import Document
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.exc import DBAPIError

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.sqlserver import SQLServer_VectorStore

_CONNECTION_STRING = str(os.environ.get("TEST_AZURESQLSERVER_CONNECTION_STRING"))
_SCHEMA = "lc_test"
_SYS_TABLE_QUERY = """
select object_id from sys.tables where name = '%s'
and schema_name(schema_id) = '%s'"""
_TABLE_NAME = "langchain_vector_store_tests"
_TABLE_DOES_NOT_EXIST = "Table %s.%s does not exist."


@pytest.fixture
def store() -> Generator[SQLServer_VectorStore, None, None]:
    """Setup resources that are needed for the duration of the test."""
    store = SQLServer_VectorStore(
        connection_string=_CONNECTION_STRING,
        embedding_function=FakeEmbeddings(size=1536),
        table_name="langchain_vector_store_tests",
    )
    yield store  # provide this data to the test

    # Drop store after it's done being used in the test case.
    store.drop()


@pytest.fixture
def texts() -> List[str]:
    """Definition of texts used in the tests."""
    query = [
        """I have bought several of the Vitality canned dog food products and have 
        found them all to be of good quality. The product looks more like a stew 
        than a processed meat and it smells better. My Labrador is finicky and she 
        appreciates this product better than  most.""",
        """The candy is just red , No flavor . Just  plan and chewy .
        I would never buy them again""",
        "Arrived in 6 days and were so stale i could not eat any of the 6 bags!!",
        """Got these on sale for roughly 25 cents per cup, which is half the price 
        of my local grocery stores, plus they rarely stock the spicy flavors. These 
        things are a GREAT snack for my office where time is constantly crunched and 
        sometimes you can't escape for a real meal. This is one of my favorite flavors 
        of Instant Lunch and will be back to buy every time it goes on sale.""",
        """If you are looking for a less messy version of licorice for the children, 
        then be sure to try these!  They're soft, easy to chew, and they don't get your 
        hands all sticky and gross in the car, in the summer, at the beach, etc. 
        We love all the flavos and sometimes mix these in with the chocolate to have a 
        very nice snack! Great item, great price too, highly recommend!""",
    ]
    return query  # provide this data to the test.


def test_sqlserver_add_texts(store: SQLServer_VectorStore) -> None:
    """Test that add text returns equivalent number of ids of input texts."""
    texts = ["rabbit", "cherry", "hamster", "cat", "elderberry"]
    metadatas = [
        {"color": "black", "type": "pet", "length": 6},
        {"color": "red", "type": "fruit", "length": 6},
        {"color": "brown", "type": "pet", "length": 7},
        {"color": "black", "type": "pet", "length": 3},
        {"color": "blue", "type": "fruit", "length": 10},
    ]
    result = store.add_texts(texts, metadatas)
    assert len(result) == len(texts)


def test_sqlserver_add_texts_when_no_metadata_is_provided(
    store: SQLServer_VectorStore,
) -> None:
    """Test that when user calls the add_texts function without providing metadata,
    the embedded text still get added to the vector store."""
    texts = [
        "Good review",
        "new books",
        "table",
        "Sunglasses are a form of protective eyewear.",
    ]
    result = store.add_texts(texts)
    assert len(result) == len(texts)


def test_sqlserver_add_texts_when_text_length_and_metadata_length_vary(
    store: SQLServer_VectorStore,
) -> None:
    """Test that all texts provided are added into the vector store
    even when metadata is not available for all the texts."""
    # The text 'elderberry' and its embedded value should be added to the vector store.
    texts = ["rabbit", "cherry", "hamster", "cat", "elderberry"]
    metadatas = [
        {"color": "black", "type": "pet", "length": 6},
        {"color": "red", "type": "fruit", "length": 6},
        {"color": "brown", "type": "pet", "length": 7},
        {"color": "black", "type": "pet", "length": 3},
    ]
    result = store.add_texts(texts, metadatas)
    assert len(result) == len(texts)


def test_sqlserver_add_texts_when_list_of_given_id_is_less_than_list_of_texts(
    store: SQLServer_VectorStore,
) -> None:
    """Test that when length of given id is less than length of texts,
    random ids are created."""
    texts = [
        "Good review",
        "new books",
        "table",
        "Sunglasses are a form of protective eyewear.",
        "It's a new year.",
    ]

    # List of ids is 3 and is less than len(texts) which is 5.
    metadatas = [
        {"id": 1, "soure": "book review", "length": 11},
        {"id": 2, "source": "random texts", "length": 9},
        {"source": "household list", "length": 5},
        {"id": 6, "source": "newspaper page", "length": 44},
        {"source": "random texts", "length": 16},
    ]
    result = store.add_texts(texts, metadatas)

    # Length of ids returned by add_texts function should be equal to length of texts.
    assert len(result) == len(texts)


def test_add_document_with_sqlserver(store: SQLServer_VectorStore) -> None:
    """Test that when add_document function is used, it integrates well
    with the add_text function in SQLServer Vector Store."""
    docs = [
        Document(
            page_content="rabbit",
            metadata={"color": "black", "type": "pet", "length": 6},
        ),
        Document(
            page_content="cherry",
            metadata={"color": "red", "type": "fruit", "length": 6},
        ),
        Document(
            page_content="hamster",
            metadata={"color": "brown", "type": "pet", "length": 7},
        ),
        Document(
            page_content="cat", metadata={"color": "black", "type": "pet", "length": 3}
        ),
        Document(
            page_content="elderberry",
            metadata={"color": "blue", "type": "fruit", "length": 10},
        ),
    ]
    result = store.add_documents(docs)
    assert len(result) == len(docs)


def test_that_a_document_entry_without_metadata_will_be_added_to_vectorstore(
    store: SQLServer_VectorStore,
) -> None:
    docs = [
        Document(
            page_content="rabbit",
            metadata={"color": "black", "type": "pet", "length": 6},
        ),
        Document(
            page_content="cherry",
        ),
        Document(
            page_content="hamster",
            metadata={"color": "brown", "type": "pet", "length": 7},
        ),
        Document(page_content="cat"),
        Document(
            page_content="elderberry",
            metadata={"color": "blue", "type": "fruit", "length": 10},
        ),
    ]
    result = store.add_documents(docs)
    assert len(result) == len(docs)


def test_that_drop_deletes_vector_store(
    store: SQLServer_VectorStore,
    texts: List[str],
) -> None:
    """Test that when drop is called, vector store is deleted
    and a call to add_text raises an exception.
    """
    store.drop()
    with pytest.raises(DBAPIError):
        store.add_texts(texts)


def test_sqlserver_delete_text_by_id_valid_ids_provided(
    store: SQLServer_VectorStore,
) -> None:
    """Test that delete API deletes texts by id."""
    texts = [
        "Good review",
        "new books",
        "table",
        "Sunglasses are a form of protective eyewear.",
        "It's a new year.",
    ]

    metadatas = [
        {"id": 100, "source": "book review", "length": 11},
        {"id": 200, "source": "random texts", "length": 9},
        {"id": 200, "source": "household list", "length": 5},
        {"id": 600, "source": "newspaper page", "length": 44},
        {"id": 300, "source": "random texts", "length": 16},
    ]
    store.add_texts(texts, metadatas)

    result = store.delete(["100", "200", "600"])
    # Should return true since valid ids are given
    if result:
        pass


def test_sqlserver_delete_text_by_id_valid_id_and_invalid_ids_provided(
    store: SQLServer_VectorStore,
) -> None:
    """Test that delete API deletes texts by id."""
    texts = [
        "Good review",
        "new books",
        "table",
        "Sunglasses are a form of protective eyewear.",
        "It's a new year.",
    ]

    metadatas = [
        {"id": 100, "source": "book review", "length": 11},
        {"id": 200, "source": "random texts", "length": 9},
        {"id": 200, "source": "household list", "length": 5},
        {"id": 600, "source": "newspaper page", "length": 44},
        {"id": 300, "source": "random texts", "length": 16},
    ]
    store.add_texts(texts, metadatas)

    result = store.delete(["100", "200", "600", "900"])
    # Should return true since valid ids are given
    if result:
        pass


def test_sqlserver_delete_text_by_id_invalid_ids_provided(
    store: SQLServer_VectorStore,
) -> None:
    """Test that delete API deletes texts by id."""
    texts = [
        "Good review",
        "new books",
        "table",
        "Sunglasses are a form of protective eyewear.",
        "It's a new year.",
    ]

    metadatas = [
        {"id": 100, "source": "book review", "length": 11},
        {"id": 200, "source": "random texts", "length": 9},
        {"id": 200, "source": "household list", "length": 5},
        {"id": 600, "source": "newspaper page", "length": 44},
        {"id": 300, "source": "random texts", "length": 16},
    ]
    store.add_texts(texts, metadatas)

    result = store.delete(["100000"])
    # Should return False since given id is not in DB
    if not result:
        pass


def test_sqlserver_delete_text_by_id_no_ids_provided(
    store: SQLServer_VectorStore,
) -> None:
    """Test that delete API deletes texts by id."""
    texts = [
        "Good review",
        "new books",
        "table",
        "Sunglasses are a form of protective eyewear.",
        "It's a new year.",
    ]

    metadatas = [
        {"id": 100, "source": "book review", "length": 11},
        {"id": 200, "source": "random texts", "length": 9},
        {"id": 200, "source": "household list", "length": 5},
        {"id": 600, "source": "newspaper page", "length": 44},
        {"id": 300, "source": "random texts", "length": 16},
    ]
    result = store.add_texts(texts, metadatas)

    # Should return False, since empty list of ids given
    if not result:
        pass


def test_that_multiple_vector_stores_can_be_created(
    store: SQLServer_VectorStore,
) -> None:
    """Tests that when multiple SQLServer_VectorStore objects are
    created, the first created vector store is not reused, but
    multiple vector stores are created."""

    # Create another vector store with a different table name.
    new_store = SQLServer_VectorStore(
        connection_string=_CONNECTION_STRING,
        embedding_function=FakeEmbeddings(size=1536),
        table_name="langchain_vector_store_tests_2",
    )

    # Check that the name of the table being created for the embeddingstore
    # is what is expected.
    assert new_store._embedding_store.__table__.name == "langchain_vector_store_tests_2"

    # Drop the new_store table to clean up this test run.
    new_store.drop()


def test_that_schema_input_is_used() -> None:
    """Tests that when a schema is given as input to the SQLServer_VectorStore object,
    a vector store is created within the schema."""
    connection = create_schema()

    # Create a vector store in the DB with the schema just created
    sqlserver_vectorstore = SQLServer_VectorStore(
        connection=connection,
        connection_string=_CONNECTION_STRING,
        db_schema=_SCHEMA,
        embedding_function=FakeEmbeddings(size=1536),
        table_name=_TABLE_NAME,
    )
    sqlserver_vectorstore.add_texts(["cats"])

    # Confirm table in that schema exists.
    result = connection.execute(text(_SYS_TABLE_QUERY % (_TABLE_NAME, _SCHEMA)))
    assert result.fetchone() is not None, _TABLE_DOES_NOT_EXIST % (_SCHEMA, _TABLE_NAME)
    connection.close()


def test_that_same_name_vector_store_can_be_created_in_different_schemas() -> None:
    """Tests that vector stores can be created with same name in different
    schemas even with the same connection."""
    connection = create_schema()

    # Create a vector store in the DB with the schema just created
    sqlserver_vectorstore = SQLServer_VectorStore(
        connection=connection,
        connection_string=_CONNECTION_STRING,
        db_schema=_SCHEMA,
        embedding_function=FakeEmbeddings(size=1536),
        table_name=_TABLE_NAME,
    )

    # Create a vector store in the DB with the default schema
    sqlserver_vectorstore_default_schema = SQLServer_VectorStore(
        connection=connection,
        connection_string=_CONNECTION_STRING,
        embedding_function=FakeEmbeddings(size=1536),
        table_name=_TABLE_NAME,
    )

    sqlserver_vectorstore.add_texts(["cats"])
    result_with_schema = connection.execute(
        text(_SYS_TABLE_QUERY % (_TABLE_NAME, _SCHEMA))
    )
    assert result_with_schema.fetchone() is not None, _TABLE_DOES_NOT_EXIST % (
        _SCHEMA,
        _TABLE_NAME,
    )

    sqlserver_vectorstore_default_schema.add_texts(["cats"])
    result_with_default = connection.execute(
        text(_SYS_TABLE_QUERY % (_TABLE_NAME, "dbo"))
    )
    assert result_with_default.fetchone() is not None, _TABLE_DOES_NOT_EXIST % (
        "dbo",
        _TABLE_NAME,
    )
    connection.close()


def create_schema() -> Connection:
    engine = create_engine(url=_CONNECTION_STRING)
    connection = engine.connect()
    # Create a schema in the DB
    connection.execute(text(f"create schema {_SCHEMA}"))
    return connection
