"""Test PGVector functionality."""

from unittest import mock
from unittest.mock import Mock

import pytest

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import pgvector

_CONNECTION_STRING = pgvector.PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="postgres",
)

_EMBEDDING_FUNCTION = FakeEmbeddings(size=1536)


@pytest.mark.requires("pgvector")
@mock.patch("sqlalchemy.create_engine")
def test_given_a_connection_is_provided_then_no_engine_should_be_created(
    create_engine: Mock,
) -> None:
    """When a connection is provided then no engine should be created."""
    pgvector.PGVector(
        connection_string=_CONNECTION_STRING,
        embedding_function=_EMBEDDING_FUNCTION,
        connection=mock.MagicMock(),
    )
    create_engine.assert_not_called()


@pytest.mark.requires("pgvector")
@mock.patch("sqlalchemy.create_engine")
def test_given_no_connection_or_engine_args_provided_default_engine_should_be_used(
    create_engine: Mock,
) -> None:
    """When no connection or engine arguments are provided then the default configuration must be used."""  # noqa: E501
    pgvector.PGVector(
        connection_string=_CONNECTION_STRING,
        embedding_function=_EMBEDDING_FUNCTION,
    )
    create_engine.assert_called_with(
        url=_CONNECTION_STRING,
    )


@pytest.mark.requires("pgvector")
@mock.patch("sqlalchemy.create_engine")
def test_given_engine_args_are_provided_then_they_should_be_used(
    create_engine: Mock,
) -> None:
    """When engine arguments are provided then they must be used to create the underlying engine."""  # noqa: E501
    engine_args = {
        "pool_size": 5,
        "max_overflow": 10,
        "pool_recycle": -1,
        "pool_use_lifo": False,
        "pool_pre_ping": False,
        "pool_timeout": 30,
    }
    pgvector.PGVector(
        connection_string=_CONNECTION_STRING,
        embedding_function=_EMBEDDING_FUNCTION,
        engine_args=engine_args,
    )
    create_engine.assert_called_with(
        url=_CONNECTION_STRING,
        **engine_args,
    )
