"""Test SQL Database Chain."""
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, insert

from langchain.chains.sql_database.base import SQLDatabaseChain
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


def test_sql_database_run() -> None:
    """Test that commands can be run successfully and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison", user_company="Foo")
    with engine.connect() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine)
    db_chain = SQLDatabaseChain(llm=OpenAI(temperature=0), database=db)
    output = db_chain.query("What company does Harrison work at?")
    expected_output = " Harrison works at Foo."
    assert output == expected_output
