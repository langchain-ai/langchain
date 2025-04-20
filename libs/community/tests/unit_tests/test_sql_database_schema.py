# flake8: noqa
"""Test SQL database wrapper with schema support.

Using DuckDB as SQLite does not support schemas.
"""

import pytest

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    Sequence,
    String,
    Table,
    create_engine,
    event,
    insert,
    schema,
)
import sqlalchemy as sa

from packaging import version

from langchain_community.utilities.sql_database import SQLDatabase

metadata_obj = MetaData()

event.listen(metadata_obj, "before_create", schema.CreateSchema("schema_a"))
event.listen(metadata_obj, "before_create", schema.CreateSchema("schema_b"))

user = Table(
    "user",
    metadata_obj,
    Column("user_id", Integer, Sequence("user_id_seq"), primary_key=True),
    Column("user_name", String, nullable=False),
    schema="schema_a",
)

company = Table(
    "company",
    metadata_obj,
    Column("company_id", Integer, Sequence("company_id_seq"), primary_key=True),
    Column("company_location", String, nullable=False),
    schema="schema_b",
)


@pytest.mark.xfail(
    version.parse(sa.__version__).major == 1, reason="SQLAlchemy 1.x issues"
)
def test_table_info() -> None:
    """Test that table info is constructed properly."""
    engine = create_engine("duckdb:///:memory:")
    metadata_obj.create_all(engine)

    db = SQLDatabase(engine, schema="schema_a", metadata=metadata_obj)
    output = db.table_info
    expected_output = """
    CREATE TABLE schema_a."user" (
        user_id INTEGER NOT NULL,
        user_name VARCHAR NOT NULL,
        PRIMARY KEY (user_id)
    )
    /*
    3 rows from user table:
    user_id user_name
    */
    """

    assert sorted(" ".join(output.split())) == sorted(" ".join(expected_output.split()))


@pytest.mark.xfail(
    version.parse(sa.__version__).major == 1, reason="SQLAlchemy 1.x issues"
)
def test_sql_database_run() -> None:
    """Test that commands can be run successfully and returned in correct format."""
    engine = create_engine("duckdb:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison")
    with engine.begin() as conn:
        conn.execute(stmt)

    with pytest.warns(Warning) as records:
        db = SQLDatabase(engine, schema="schema_a")

    # Metadata creation with duckdb raises 3 warnings at the moment about reflection.
    # As a stop-gap to increase strictness of pytest to fail on warnings, we'll
    # explicitly catch the warnings and assert that it's the one we expect.
    # We may need to revisit at a later stage and determine why a warning is being
    # raised here.
    for record in records:
        assert isinstance(record.message, Warning)
    assert any(
        record.message.args[0]  # type: ignore[union-attr]
        == "duckdb-engine doesn't yet support reflection on indices"
        for record in records
    )

    command = 'select user_name from "user" where user_id = 13'
    output = db.run(command)
    expected_output = "[('Harrison',)]"
    assert output == expected_output
