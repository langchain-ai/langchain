"""Test SQL Database Chain."""
import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, insert

from langchain.chains.sql_database.base import (
    SQLDatabaseChain,
    SQLDatabaseSequentialChain,
    SQLValidation,
)
from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase

metadata_obj = MetaData()

user = Table(
    "user",
    metadata_obj,
    Column("user_id", Integer, primary_key=True),
    Column("user_name", String(16), nullable=False),
    Column("user_company", String(16), nullable=False),
)


@pytest.mark.requires("sqlfluff")
def test_sql_database_run() -> None:
    """Test that commands can be run successfully and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    db = SQLDatabase(engine)
    db.run(f"INSERT INTO user VALUES (13, 'Harrison', 'Foo')")
    db_chain = SQLDatabaseChain.from_llm(OpenAI(temperature=0), db)
    output = db_chain.run("What company does Harrison work at?")
    expected_output = "Harrison works at Foo."
    assert output == expected_output


@pytest.mark.requires("sqlfluff")
def test_sql_database_run_update() -> None:
    """Test that update commands run successfully and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    db = SQLDatabase(engine)
    db.run(f"INSERT INTO user VALUES (13, 'Harrison', 'Foo')")
    db_chain = SQLDatabaseChain.from_llm(
        OpenAI(temperature=0),
        db,
        sql_validation=SQLValidation(allow_non_select_statements=True),
    )
    output = db_chain.run("Update Harrison's workplace to Bar")
    expected_output = "Harrison's workplace has been updated to Bar."
    assert output == expected_output
    output = db_chain.run("What company does Harrison work at?")
    expected_output = "Harrison works at Bar."
    assert output == expected_output


@pytest.mark.requires("sqlfluff")
def test_sql_database_sequential_chain_run() -> None:
    """Test that commands can be run successfully SEQUENTIALLY
    and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    db = SQLDatabase(engine)
    db.run(f"INSERT INTO user VALUES (13, 'Harrison', 'Foo')")
    db_chain = SQLDatabaseSequentialChain.from_llm(OpenAI(temperature=0), db)
    output = db_chain.run("What company does Harrison work at?")
    expected_output = "Harrison works at Foo."
    assert output == expected_output


@pytest.mark.requires("sqlfluff")
def test_sql_database_sequential_chain_intermediate_steps() -> None:
    """Test that commands can be run successfully SEQUENTIALLY and returned
    in correct format. sWith Intermediate steps"""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    db = SQLDatabase(engine)
    db.run(f"INSERT INTO user VALUES (13, 'Harrison', 'Foo')")
    db_chain = SQLDatabaseSequentialChain.from_llm(
        OpenAI(temperature=0), db, return_intermediate_steps=True
    )
    output = db_chain("What company does Harrison work at?")
    expected_output = "Harrison works at Foo."
    assert output["result"] == expected_output

    query = output["intermediate_steps"][1]
    expected_query = "SELECT user_company FROM user WHERE user_name = 'Harrison';"
    assert query == expected_query

    query_results = output["intermediate_steps"][3]
    expected_query_results = "[('Foo',)]"
    assert query_results == expected_query_results
