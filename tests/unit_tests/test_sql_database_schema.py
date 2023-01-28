"""Test SQL database wrapper with schema support.

Using DuckDB as SQLite does not support schemas.
"""

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

from langchain.sql_database import SQLDatabase

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


def test_table_info() -> None:
    """Test that table info is constructed properly."""
    engine = create_engine("duckdb:///:memory:")
    metadata_obj.create_all(engine)
    db = SQLDatabase(engine, schema="schema_a")
    output = db.table_info
    expected_output = (
        "Table 'user' has columns: user_id (INTEGER), user_name (VARCHAR).",
    )
    assert sorted(output.split("\n")) == sorted(expected_output)


def test_sql_database_run() -> None:
    """Test that commands can be run successfully and returned in correct format."""
    engine = create_engine("duckdb:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison")
    with engine.begin() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine, schema="schema_a")
    command = 'select user_name from "user" where user_id = 13'
    output = db.run(command)
    expected_output = "[('Harrison',)]"
    assert output == expected_output
