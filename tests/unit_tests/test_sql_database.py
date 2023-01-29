"""Test SQL database wrapper."""

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, insert

from langchain.sql_database import SQLDatabase

metadata_obj = MetaData()

user = Table(
    "user",
    metadata_obj,
    Column("user_id", Integer, primary_key=True),
    Column("user_name", String(16), nullable=False),
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
    expected_output = (
        "Table 'company' has columns: company_id (INTEGER), "
        "company_location (VARCHAR).",
        "Table 'user' has columns: user_id (INTEGER), user_name (VARCHAR(16)).",
    )
    assert sorted(output.split("\n")) == sorted(expected_output)


def test_table_info_w_sample_row() -> None:
    """Test that table info is constructed properly."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison")
    with engine.begin() as conn:
        conn.execute(stmt)

    db = SQLDatabase(engine, sample_row_in_table_info=True)

    output = db.table_info
    expected_output = (
        "Table 'company' has columns: company_id (INTEGER), "
        "company_location (VARCHAR).\n"
        "Table 'user' has columns: user_id (INTEGER), "
        "user_name (VARCHAR(16)). Here is an example row "
        "for this table (long strings are truncated): 13 Harrison."
    )
    assert sorted(output.split("\n")) == sorted(expected_output.split("\n"))


def test_sql_database_run() -> None:
    """Test that commands can be run successfully and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison")
    with engine.begin() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine)
    command = "select user_name from user where user_id = 13"
    output = db.run(command)
    expected_output = "[('Harrison',)]"
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
