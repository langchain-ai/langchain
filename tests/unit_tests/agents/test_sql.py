from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_create_sql_agent() -> None:
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
    )

    assert agent_executor.run("hello") == "baz"
