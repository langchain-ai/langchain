# flake8: noqa=E501
"""Test SQL database wrapper."""

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
)
from typing import List, Dict, Any

from langchain_experimental.utilities.sql_database import SQLDatabase, truncate_word

metadata_obj = MetaData()

user = Table(
    "user",
    metadata_obj,
    Column("user_id", Integer, primary_key=True),
    Column("user_name", String(16), nullable=False),
    Column("user_bio", Text, nullable=True),
)

company = Table(
    "company",
    metadata_obj,
    Column("company_id", Integer, primary_key=True),
    Column("company_location", String, nullable=False),
)


def test_sql_database_run_native_format() -> None:
    """Test that commands can be run successfully and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(
        user_id=13, user_name="Harrison", user_bio="That is my Bio " * 24
    )
    with engine.begin() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine)
    command = "select user_id, user_name, user_bio from user where user_id = 13"
    output: List[Dict[str, Any]] = db.run(command, native_format=True)  # type: ignore
    user_bio = "That is my Bio " * 19 + "That is my..."
    expected_output = {
        "user_id": 13,
        "user_name": "Harrison",
        "user_bio": "That is my Bio " * 24,
    }
    for k, v in expected_output.items():
        assert output[0][k] == v
