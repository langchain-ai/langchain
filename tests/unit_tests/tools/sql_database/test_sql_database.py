import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, insert

from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.tools import QuerySQLDataBaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool
from tests.unit_tests.llms.fake_llm import FakeLLM

metadata_obj = MetaData()

user = Table(
    "user",
    metadata_obj,
    Column("user_id", Integer, primary_key=True),
    Column("user_name", String(16), nullable=False),
    Column("user_company", String(16), nullable=False),
)


@pytest.fixture(scope="module")
def db():
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison", user_company="Foo")
    with engine.connect() as conn:
        conn.execute(stmt)
        conn.commit()
    db = SQLDatabase(engine)
    return db


@pytest.fixture(scope="module")
def query_sql_database_tool(db):
    query_sql_database_tool = QuerySQLDataBaseTool(
        db=db, description='test'
    )
    return query_sql_database_tool


@pytest.fixture(scope="module")
def info_sql_database_tool(db):
    info_sql_database_tool = InfoSQLDatabaseTool(
        db=db, description='test'
    )
    return info_sql_database_tool


@pytest.fixture(scope="module")
def list_sql_database_tool(db):
    list_sql_database_tool = ListSQLDatabaseTool(
        db=db, description='test'
    )
    return list_sql_database_tool


@pytest.fixture(scope="module")
def sql_toolkit(db):
    llm = FakeLLM()
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return sql_toolkit


def test_query_sql_database_tool_run(query_sql_database_tool) -> None:
    result = query_sql_database_tool._run(query="SELECT * FROM user")
    assert result == "[(13, 'Harrison', 'Foo')]"


@pytest.mark.asyncio
async def test_query_database_tool_arun(query_sql_database_tool) -> None:
    result = await query_sql_database_tool._arun(query="SELECT * FROM user")
    assert result == "[(13, 'Harrison', 'Foo')]"


def test_info_database_tool_run(info_sql_database_tool) -> None:
    result = info_sql_database_tool._run(table_names="user")
    expected_result = """
CREATE TABLE user (
	user_id INTEGER NOT NULL, 
	user_name VARCHAR(16) NOT NULL, 
	user_company VARCHAR(16) NOT NULL, 
	PRIMARY KEY (user_id)
)

/*
3 rows from user table:
user_id	user_name	user_company
13	Harrison	Foo
*/"""
    assert result == expected_result


@pytest.mark.asyncio
async def test_info_database_tool_arun(info_sql_database_tool) -> None:
    result = await info_sql_database_tool._arun(table_names="user")
    expected_result = """
CREATE TABLE user (
	user_id INTEGER NOT NULL, 
	user_name VARCHAR(16) NOT NULL, 
	user_company VARCHAR(16) NOT NULL, 
	PRIMARY KEY (user_id)
)

/*
3 rows from user table:
user_id	user_name	user_company
13	Harrison	Foo
*/"""
    assert result == expected_result


def test_list_sql_database_tool_run(list_sql_database_tool) -> None:
    result = list_sql_database_tool._run()
    assert result == "user"


@pytest.mark.asyncio
async def test_list_sql_database_tool_arun(list_sql_database_tool) -> None:
    result = await list_sql_database_tool._arun()
    assert result == "user"


def test_sql_toolkit_run(sql_toolkit) -> None:
    results = sql_toolkit.get_tools()
    tools = [result.name for result in results]
    expected_tools = ['sql_db_query', 'sql_db_schema', 'sql_db_list_tables', 'sql_db_query_checker']
    assert tools == expected_tools
