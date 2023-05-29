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

from langchain.sql_database import SQLDatabase, truncate_word

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


def test_table_info() -> None:
    """Test that table info is constructed properly."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    db = SQLDatabase(engine)
    output = db.table_info
    expected_output = """
    CREATE TABLE user (
            user_id INTEGER NOT NULL,
            user_name VARCHAR(16) NOT NULL,
            user_bio TEXT,
            PRIMARY KEY (user_id)
    )
    /*
    3 rows from user table:
    user_id user_name user_bio
    /*


    CREATE TABLE company (
            company_id INTEGER NOT NULL,
            company_location VARCHAR NOT NULL,
            PRIMARY KEY (company_id)
    )
    /*
    3 rows from company table:
    company_id company_location
    */
    """

    assert sorted(" ".join(output.split())) == sorted(" ".join(expected_output.split()))


def test_table_info_w_sample_rows() -> None:
    """Test that table info is constructed properly."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    values = [
        {"user_id": 13, "user_name": "Harrison", "user_bio": "bio"},
        {"user_id": 14, "user_name": "Chase", "user_bio": "bio"},
    ]
    stmt = insert(user).values(values)
    with engine.begin() as conn:
        conn.execute(stmt)

    db = SQLDatabase(engine, sample_rows_in_table_info=2)

    output = db.table_info

    expected_output = """
        CREATE TABLE company (
        company_id INTEGER NOT NULL,
        company_location VARCHAR NOT NULL,
        PRIMARY KEY (company_id)
)
        /*
        2 rows from company table:
        company_id company_location
        */

        CREATE TABLE user (
        user_id INTEGER NOT NULL,
        user_name VARCHAR(16) NOT NULL,
        user_bio TEXT,
        PRIMARY KEY (user_id)
        )
        /*
        2 rows from user table:
        user_id user_name user_bio
        13 Harrison bio
        14 Chase bio
        */
        """

    assert sorted(output.split()) == sorted(expected_output.split())


def test_sql_database_run() -> None:
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
    output = db.run(command)
    user_bio = "That is my Bio " * 19 + "That is my..."
    expected_output = f"[(13, 'Harrison', '{user_bio}')]"
    assert output == expected_output


def test_sql_database_run_update() -> None:
    """Test commands which return no rows return an empty string."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison")
    with engine.begin() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine)
    command = "update user set user_name='Updated' where user_id = 13"
    output = db.run(command)
    expected_output = ""
    assert output == expected_output


def test_truncate_word() -> None:
    assert truncate_word("Hello World", length=5) == "He..."
    assert truncate_word("Hello World", length=0) == "Hello World"
    assert truncate_word("Hello World", length=-10) == "Hello World"
    assert truncate_word("Hello World", length=5, suffix="!!!") == "He!!!"
    assert truncate_word("Hello World", length=12, suffix="!!!") == "Hello World"
