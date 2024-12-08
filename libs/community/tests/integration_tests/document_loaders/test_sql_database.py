"""
Test SQLAlchemy document loader functionality on behalf of SQLite and PostgreSQL.

To run the tests for SQLite, you need to have the `sqlite3` package installed.

To run the tests for PostgreSQL, you need to have the `psycopg2` package installed.
In addition, to launch the PostgreSQL instance, you can use the docker compose file
located at the root of the repo, `langchain/docker/docker-compose.yml`. Use the
command `docker compose up postgres` to start the instance. It will have the
appropriate credentials set up including being exposed on the appropriate port.
"""

import functools
import logging
import typing
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import sqlalchemy as sa

from langchain_community.utilities.sql_database import SQLDatabase

if typing.TYPE_CHECKING:
    from _pytest.python import Metafunc

from langchain_community.document_loaders.sql_database import SQLDatabaseLoader
from tests.data import MLB_TEAMS_2012_SQL

logging.basicConfig(level=logging.DEBUG)


try:
    import sqlite3  # noqa: F401

    sqlite3_installed = True
except ImportError:
    warnings.warn("sqlite3 not installed, skipping corresponding tests", UserWarning)
    sqlite3_installed = False

try:
    import psycopg2  # noqa: F401

    psycopg2_installed = True
except ImportError:
    warnings.warn("psycopg2 not installed, skipping corresponding tests", UserWarning)
    psycopg2_installed = False

try:
    import sqlalchemy_cratedb  # noqa: F401

    cratedb_installed = True
except ImportError:
    warnings.warn("cratedb not installed, skipping corresponding tests", UserWarning)
    cratedb_installed = False


@pytest.fixture()
def engine(db_uri: str) -> sa.Engine:
    """
    Return an SQLAlchemy engine object.
    """
    return sa.create_engine(db_uri, echo=False)


@pytest.fixture()
def db(engine: sa.Engine) -> SQLDatabase:
    return SQLDatabase(engine=engine)


@pytest.fixture()
def provision_database(engine: sa.Engine) -> None:
    """
    Provision database with table schema and data.
    """
    sql_statements = MLB_TEAMS_2012_SQL.read_text()
    with engine.connect() as connection:
        connection.execute(sa.text("DROP TABLE IF EXISTS mlb_teams_2012;"))
        for statement in sql_statements.split(";"):
            statement = statement.strip()
            if not statement:
                continue
            connection.execute(sa.text(statement))
            connection.commit()
        if engine.dialect.name.startswith("crate"):
            connection.execute(sa.text("REFRESH TABLE mlb_teams_2012;"))
            connection.commit()


tmpdir = TemporaryDirectory()


def pytest_generate_tests(metafunc: "Metafunc") -> None:
    """
    Dynamically parameterize test cases to verify both SQLite and PostgreSQL.
    """
    if "db_uri" in metafunc.fixturenames:
        urls = []
        ids = []
        if sqlite3_installed:
            db_path = Path(tmpdir.name).joinpath("testdrive.sqlite")
            urls.append(f"sqlite:///{db_path}")
            ids.append("sqlite")
        if psycopg2_installed:
            # We use non-standard port for testing purposes.
            # The easiest way to spin up the PostgreSQL instance is to use
            # the docker compose file at the root of the repo located at
            # langchain/docker/docker-compose.yml
            # use `docker compose up postgres` to start the instance
            # it will have the appropriate credentials set up including
            # being exposed on the appropriate port.
            urls.append(
                "postgresql+psycopg2://langchain:langchain@localhost:6023/langchain"
            )
            ids.append("postgresql")
        if cratedb_installed:
            # We use non-standard port for testing purposes.
            # The easiest way to spin up the PostgreSQL instance is to use
            # the docker compose file at the root of the repo located at
            # langchain/docker/docker-compose.yml
            # use `docker compose up postgres` to start the instance
            # it will have the appropriate credentials set up including
            # being exposed on the appropriate port.
            urls.append("crate://crate@localhost/?schema=testdrive")
            ids.append("cratedb")

        metafunc.parametrize("db_uri", urls, ids=ids)


def test_sqldatabase_loader_no_options(db: SQLDatabase) -> None:
    """Test SQLAlchemy loader basics."""

    loader = SQLDatabaseLoader("SELECT 1 AS a, 2 AS b", db=db)
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {}


def test_sqldatabase_loader_include_rownum_into_metadata(db: SQLDatabase) -> None:
    """Test SQLAlchemy loader with row number in metadata."""

    loader = SQLDatabaseLoader(
        "SELECT 1 AS a, 2 AS b",
        db=db,
        include_rownum_into_metadata=True,
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {"row": 0}


def test_sqldatabase_loader_include_query_into_metadata(db: SQLDatabase) -> None:
    """Test SQLAlchemy loader with query in metadata."""

    loader = SQLDatabaseLoader(
        "SELECT 1 AS a, 2 AS b", db=db, include_query_into_metadata=True
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {"query": "SELECT 1 AS a, 2 AS b"}


def test_sqldatabase_loader_page_content_columns(db: SQLDatabase) -> None:
    """Test SQLAlchemy loader with defined page content columns."""

    # Define a custom callback function to convert a row into a "page content" string.
    row_to_content = functools.partial(
        SQLDatabaseLoader.page_content_default_mapper, column_names=["a"]
    )

    loader = SQLDatabaseLoader(
        "SELECT 1 AS a, 2 AS b UNION SELECT 3 AS a, 4 AS b",
        db=db,
        page_content_mapper=row_to_content,
    )
    docs = loader.load()

    assert len(docs) == 2
    assert docs[0].page_content == "a: 1"
    assert docs[0].metadata == {}

    assert docs[1].page_content == "a: 3"
    assert docs[1].metadata == {}


def test_sqldatabase_loader_metadata_columns(db: SQLDatabase) -> None:
    """Test SQLAlchemy loader with defined metadata columns."""

    # Define a custom callback function to convert a row into a "metadata" dictionary.
    row_to_metadata = functools.partial(
        SQLDatabaseLoader.metadata_default_mapper, column_names=["b"]
    )

    loader = SQLDatabaseLoader(
        "SELECT 1 AS a, 2 AS b",
        db=db,
        metadata_mapper=row_to_metadata,
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata == {"b": 2}


def test_sqldatabase_loader_real_data_with_sql_no_parameters(
    db: SQLDatabase, provision_database: None
) -> None:
    """Test SQLAlchemy loader with real data, querying by SQL statement."""

    loader = SQLDatabaseLoader(
        query='SELECT * FROM mlb_teams_2012 ORDER BY "Team";',
        db=db,
    )
    docs = loader.load()

    assert len(docs) == 30
    assert docs[0].page_content == "Team: Angels\nPayroll (millions): 154.49\nWins: 89"
    assert docs[0].metadata == {}


def test_sqldatabase_loader_real_data_with_sql_and_parameters(
    db: SQLDatabase, provision_database: None
) -> None:
    """Test SQLAlchemy loader, querying by SQL statement and parameters."""

    loader = SQLDatabaseLoader(
        query='SELECT * FROM mlb_teams_2012 WHERE "Team" LIKE :search ORDER BY "Team";',
        parameters={"search": "R%"},
        db=db,
    )
    docs = loader.load()

    assert len(docs) == 6
    assert docs[0].page_content == "Team: Rangers\nPayroll (millions): 120.51\nWins: 93"
    assert docs[0].metadata == {}


def test_sqldatabase_loader_real_data_with_selectable(
    db: SQLDatabase, provision_database: None
) -> None:
    """Test SQLAlchemy loader with real data, querying by SQLAlchemy selectable."""

    # Define an SQLAlchemy table.
    mlb_teams_2012 = sa.Table(
        "mlb_teams_2012",
        sa.MetaData(),
        sa.Column("Team", sa.VARCHAR),
        sa.Column("Payroll (millions)", sa.FLOAT),
        sa.Column("Wins", sa.BIGINT),
    )

    # Query the database table using an SQLAlchemy selectable.
    select = sa.select(mlb_teams_2012).order_by(mlb_teams_2012.c.Team)
    loader = SQLDatabaseLoader(
        query=select,
        db=db,
        include_query_into_metadata=True,
    )
    docs = loader.load()

    assert len(docs) == 30
    assert docs[0].page_content == "Team: Angels\nPayroll (millions): 154.49\nWins: 89"
    assert docs[0].metadata == {
        "query": 'SELECT mlb_teams_2012."Team", mlb_teams_2012."Payroll (millions)", '
        'mlb_teams_2012."Wins" \nFROM mlb_teams_2012 '
        'ORDER BY mlb_teams_2012."Team"'
    }
