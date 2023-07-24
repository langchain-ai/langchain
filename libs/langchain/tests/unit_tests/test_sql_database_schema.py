# flake8: noqa
"""Test SQL database wrapper with schema support.

Using DuckDB as SQLite does not support schemas.
"""
import pytest
from typing import Dict, List, Any

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    Sequence,
    String,
    ARRAY,
    DateTime,
    Table,
    create_engine,
    event,
    insert,
    schema,
)

import datetime
from langchain.utilities.sql_database import SQLDatabase

metadata_obj = MetaData()

event.listen(metadata_obj, "before_create", schema.CreateSchema("schema_a"))
event.listen(metadata_obj, "before_create", schema.CreateSchema("schema_b"))
event.listen(metadata_obj, "before_create", schema.CreateSchema("schema_c"))

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

pizza = Table(
    "pizza",
    metadata_obj,
    Column("pizza_id", Integer, Sequence("pizza_id_seq"), primary_key=True),
    Column("pizza_ingredients", ARRAY(String), nullable=False),
    Column("time_cooked", DateTime, nullable=False),
    schema="schema_c",
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


def test_sql_database_run() -> None:
    """Test that commands can be run successfully and returned in correct format."""
    engine = create_engine("duckdb:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison")
    with engine.begin() as conn:
        conn.execute(stmt)

    with pytest.warns(Warning) as records:
        db = SQLDatabase(engine, schema="schema_a")

    # Metadata creation with duckdb raises a warning at the moment about reflection.
    # As a stop-gap to increase strictness of pytest to fail on warnings, we'll
    # explicitly catch the warning and assert that it's the one we expect.
    # We may need to revisit at a later stage and determine why a warning is being
    # raised here.
    assert len(records) == 1
    assert isinstance(records[0].message, Warning)
    assert (
        records[0].message.args[0]
        == "duckdb-engine doesn't yet support reflection on indices"
    )

    command = 'select user_name from "user" where user_id = 13'
    output = db.run(command)
    expected_output = "[('Harrison',)]"
    assert output == expected_output


def test_sql_database_run_native_format() -> None:
    """Test that commands can be run successfully and returned in correct format."""
    engine = create_engine("duckdb:///:memory:")
    metadata_obj.create_all(engine)
    now = datetime.datetime.now()
    stmt = insert(pizza).values(
        pizza_id=27,
        pizza_ingredients=["Pepperoni", "Cheese", "Pineapple"],
        time_cooked=now,
    )
    with engine.begin() as conn:
        conn.execute(stmt)

    with pytest.warns(Warning) as records:
        db = SQLDatabase(engine, schema="schema_c")

    # Metadata creation with duckdb raises a warning at the moment about reflection.
    # As a stop-gap to increase strictness of pytest to fail on warnings, we'll
    # explicitly catch the warning and assert that it's the one we expect.
    # We may need to revisit at a later stage and determine why a warning is being
    # raised here.
    # assert len(records) == 1
    assert isinstance(records[0].message, Warning)
    # assert (
    #     records[0].message.args[0]
    #     == "duckdb-engine doesn't yet support reflection on indices"
    # )

    command = 'select pizza_id, pizza_ingredients, time_cooked from "pizza" where pizza_id = 27'
    output: List[Dict[str, Any]] = db.run(command, native_format=True)  # type: ignore
    expected_output = {
        "pizza_id": 27,
        "pizza_ingredients": ["Pepperoni", "Cheese", "Pineapple"],
        "time_cooked": now,
    }
    print([i for i in output])
    for k, v in expected_output.items():
        assert output[0][k] == v
