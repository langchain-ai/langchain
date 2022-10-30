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
        "The 'company' table has columns: company_id (INTEGER), "
        "company_location (VARCHAR).\n"
        "The 'user' table has columns: user_id (INTEGER), user_name (VARCHAR(16))."
    )
    assert output == expected_output


def test_sql_database_run() -> None:
    """Test that commands can be run successfully and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison")
    with engine.connect() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine)
    command = "select user_name from user where user_id = 13"
    output = db.run(command)
    expected_output = "[('Harrison',)]"
    assert output == expected_output
