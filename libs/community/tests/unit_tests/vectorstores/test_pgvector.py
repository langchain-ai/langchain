"""Test PGVector functionality."""
from unittest import mock
from unittest.mock import Mock

import pytest

from langchain.embeddings import FakeEmbeddings
from langchain.pydantic_v1 import ValidationError
from langchain.schema import Document
from langchain.vectorstores import pgvector

CONNECTION_STRING = pgvector.PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="postgres",
)

connection_pool_config = pgvector.ConnectionPoolConfig(
    size=1,
    max_overflow=0,
    recycle_time=3600,
    use_lifo=True,
    pre_ping=True,
    wait_timeout=10,
)

embedding_function = FakeEmbeddings(size=1536)

texts = [
    "Madam Speaker, Madam Vice President, our First Lady and Second Gentleman.",
    "Members of Congress and the Cabinet.",
    "Justices of the Supreme Court.",
    "My fellow Americans.",
]


@pytest.mark.requires("pgvector")
def test_invalid_connection_pool_size() -> None:
    """When an invalid pool size is set then validation must fails."""
    with pytest.raises(ValidationError) as excinfo:
        pgvector.ConnectionPoolConfig(
            size=-1,
        )

    assert "ensure this value is greater than or equal to 0" in str(excinfo.value)


@pytest.mark.requires("pgvector")
def test_invalid_connection_pool_max_overflow() -> None:
    """When an invalid maximum overflow is set then validation must fails."""
    with pytest.raises(ValidationError) as excinfo:
        pgvector.ConnectionPoolConfig(
            max_overflow=-1,
        )

    assert "ensure this value is greater than or equal to 0" in str(excinfo.value)


@pytest.mark.requires("pgvector")
def test_invalid_connection_pool_wait_timeout() -> None:
    """When an invalid wait timeout is set then validation must fails."""
    with pytest.raises(ValidationError) as excinfo:
        pgvector.ConnectionPoolConfig(
            wait_timeout=-1,
        )

    assert "ensure this value is greater than or equal to 0" in str(excinfo.value)


@pytest.mark.requires("pgvector")
@mock.patch("sqlalchemy.create_engine")
def test_default_pool_configuration(create_engine: Mock) -> None:
    """When no pool configuration is set then the default must be used."""
    pgvector.PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embedding_function,
    )
    create_engine.assert_called_with(
        url=CONNECTION_STRING,
        pool_size=pgvector.DEFAULT_CONNECTION_POOL_SIZE,
        max_overflow=pgvector.DEFAULT_CONNECTION_POOL_MAX_OVERFLOW,
        pool_recycle=pgvector.DEFAULT_CONNECTION_POOL_RECYCLE_TIME,
        pool_use_lifo=pgvector.DEFAULT_CONNECTION_POOL_USE_LIFO,
        pool_pre_ping=pgvector.DEFAULT_CONNECTION_POOL_PRE_PING,
        pool_timeout=pgvector.DEFAULT_CONNECTION_POOL_WAIT_TIMEOUT,
    )


@pytest.mark.requires("pgvector")
@mock.patch("sqlalchemy.create_engine")
def test_custom_pool_configuration(create_engine: Mock) -> None:
    """When a custom pool configuration is set then it must be used."""
    pgvector.PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embedding_function,
        connection_pool_config=connection_pool_config,
    )
    create_engine.assert_called_with(
        url=CONNECTION_STRING,
        pool_size=connection_pool_config.size,
        max_overflow=connection_pool_config.max_overflow,
        pool_recycle=connection_pool_config.recycle_time,
        pool_use_lifo=connection_pool_config.use_lifo,
        pool_pre_ping=connection_pool_config.pre_ping,
        pool_timeout=connection_pool_config.wait_timeout,
    )


@pytest.mark.requires("pgvector")
@mock.patch("langchain.vectorstores.pgvector.PGVector.add_embeddings")
@mock.patch("sqlalchemy.create_engine")
def test_from_texts_with_custom_pool_configuration(
    create_engine: Mock, add_embeddings: Mock
) -> None:
    """When a custom pool configuration is set then it must be used."""
    pgvector.PGVector.from_texts(
        connection_string=CONNECTION_STRING,
        texts=texts,
        embedding=embedding_function,
        connection_pool_config=connection_pool_config,
    )
    create_engine.assert_called_with(
        url=CONNECTION_STRING,
        pool_size=connection_pool_config.size,
        max_overflow=connection_pool_config.max_overflow,
        pool_recycle=connection_pool_config.recycle_time,
        pool_use_lifo=connection_pool_config.use_lifo,
        pool_pre_ping=connection_pool_config.pre_ping,
        pool_timeout=connection_pool_config.wait_timeout,
    )
    add_embeddings.assert_called_once()


@pytest.mark.requires("pgvector")
@mock.patch("langchain.vectorstores.pgvector.PGVector.add_embeddings")
@mock.patch("sqlalchemy.create_engine")
def test_from_embeddings_with_custom_pool_configuration(
    create_engine: Mock, add_embeddings: Mock
) -> None:
    """When a custom pool configuration is set then it must be used."""
    embeddings = embedding_function.embed_documents(texts)
    text_embeddings = list(zip(texts, embeddings))
    pgvector.PGVector.from_embeddings(
        connection_string=CONNECTION_STRING,
        text_embeddings=text_embeddings,
        embedding=embedding_function,
        connection_pool_config=connection_pool_config,
    )
    create_engine.assert_called_with(
        url=CONNECTION_STRING,
        pool_size=connection_pool_config.size,
        max_overflow=connection_pool_config.max_overflow,
        pool_recycle=connection_pool_config.recycle_time,
        pool_use_lifo=connection_pool_config.use_lifo,
        pool_pre_ping=connection_pool_config.pre_ping,
        pool_timeout=connection_pool_config.wait_timeout,
    )
    add_embeddings.assert_called_once()


@pytest.mark.requires("pgvector")
@mock.patch("sqlalchemy.create_engine")
def test_from_existing_index_with_custom_pool_configuration(
    create_engine: Mock,
) -> None:
    """When a custom pool configuration is set then it must be used."""
    pgvector.PGVector.from_existing_index(
        connection_string=CONNECTION_STRING,
        embedding=embedding_function,
        connection_pool_config=connection_pool_config,
    )
    create_engine.assert_called_with(
        url=CONNECTION_STRING,
        pool_size=connection_pool_config.size,
        max_overflow=connection_pool_config.max_overflow,
        pool_recycle=connection_pool_config.recycle_time,
        pool_use_lifo=connection_pool_config.use_lifo,
        pool_pre_ping=connection_pool_config.pre_ping,
        pool_timeout=connection_pool_config.wait_timeout,
    )


@pytest.mark.requires("pgvector")
@mock.patch("langchain.vectorstores.pgvector.PGVector.add_embeddings")
@mock.patch("sqlalchemy.create_engine")
def test_from_documents_with_custom_pool_configuration(
    create_engine: Mock, add_embeddings: Mock
) -> None:
    """When a custom pool configuration is set then it must be used."""
    documents = [Document(page_content=text) for text in texts]
    pgvector.PGVector.from_documents(
        connection_string=CONNECTION_STRING,
        documents=documents,
        embedding=embedding_function,
        connection_pool_config=connection_pool_config,
    )
    create_engine.assert_called_with(
        url=CONNECTION_STRING,
        pool_size=connection_pool_config.size,
        max_overflow=connection_pool_config.max_overflow,
        pool_recycle=connection_pool_config.recycle_time,
        pool_use_lifo=connection_pool_config.use_lifo,
        pool_pre_ping=connection_pool_config.pre_ping,
        pool_timeout=connection_pool_config.wait_timeout,
    )
    add_embeddings.assert_called_once()
