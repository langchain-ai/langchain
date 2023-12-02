"""Test SQL Database Chain."""
from langchain.llms.openai import OpenAI
from langchain.utilities.sql_database import SQLDatabase
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, insert

from langchain_experimental.sql.base import (
    SQLDatabaseChain,
    SQLDatabaseSequentialChain,
)

metadata_obj = MetaData()

user = Table(
    "user",
    metadata_obj,
    Column("user_id", Integer, primary_key=True),
    Column("user_name", String(16), nullable=False),
    Column("user_company", String(16), nullable=False),
)


def test_sql_database_run() -> None:
    """Test that commands can be run successfully and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison", user_company="Foo")
    with engine.connect() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine)
    db_chain = SQLDatabaseChain.from_llm(OpenAI(temperature=0), db)
    output = db_chain.run("What company does Harrison work at?")
    expected_output = " Harrison works at Foo."
    assert output == expected_output


def test_sql_database_run_update() -> None:
    """Test that update commands run successfully and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison", user_company="Foo")
    with engine.connect() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine)
    db_chain = SQLDatabaseChain.from_llm(OpenAI(temperature=0), db)
    output = db_chain.run("Update Harrison's workplace to Bar")
    expected_output = " Harrison's workplace has been updated to Bar."
    assert output == expected_output
    output = db_chain.run("What company does Harrison work at?")
    expected_output = " Harrison works at Bar."
    assert output == expected_output


def test_sql_database_sequential_chain_run() -> None:
    """Test that commands can be run successfully SEQUENTIALLY
    and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison", user_company="Foo")
    with engine.connect() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine)
    db_chain = SQLDatabaseSequentialChain.from_llm(OpenAI(temperature=0), db)
    output = db_chain.run("What company does Harrison work at?")
    expected_output = " Harrison works at Foo."
    assert output == expected_output


def test_sql_database_sequential_chain_intermediate_steps() -> None:
    """Test that commands can be run successfully SEQUENTIALLY and returned
    in correct format. switch Intermediate steps"""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison", user_company="Foo")
    with engine.connect() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine)
    db_chain = SQLDatabaseSequentialChain.from_llm(
        OpenAI(temperature=0), db, return_intermediate_steps=True
    )
    output = db_chain("What company does Harrison work at?")
    expected_output = " Harrison works at Foo."
    assert output["result"] == expected_output

    query = output["intermediate_steps"][0]
    expected_query = (
        " SELECT user_company FROM user WHERE user_name = 'Harrison' LIMIT 1;"
    )
    assert query == expected_query

    query_results = output["intermediate_steps"][1]
    expected_query_results = "[('Foo',)]"
    assert query_results == expected_query_results
